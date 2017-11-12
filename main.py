import tensorlayer as tl
import numpy as np
import math
from config import config, log_config
from utils import *
from model import *
from scipy.ndimage.filters import gaussian_filter

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1


n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(math.ceil(np.sqrt(batch_size)))

def read_all_imgs(img_list, path='', n_threads=32, mode = 'RGB'):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGB_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def train():
    checkpoint_dir = "/data2/junyonglee/sharpness_assessment/checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    save_dir_gan = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_gan)

    train_sharp_img_list = sorted(tl.files.load_file_list(path = config.TRAIN.sharp_img_path, regx = '.*.png', printable = False))
    test_sharp_img_list = sorted(tl.files.load_file_list(path = config.TEST.sharp_img_path, regx = '.*.png', printable = False))

    train_sharp_imgs = read_all_imgs(train_sharp_img_list, path = config.TRAIN.sharp_img_path, n_threads = 32)
    test_sharp_imgs = read_all_imgs(test_sharp_img_list, path = config.TEST.sharp_img_path, n_threads = 32)

    ### DEFINE MODEL ###
    patches_blurred = tf.placeholder('float32', [batch_size, 224, 224, 3], name = 'input_patches')
    labels_sigma = tf.placeholder('float32', [batch_size, 1], name = 'lables')

    with tf.variable_scope('resNet') as scope:
        net_regression = resNet(patches_blurred, is_train = True, reuse = False, scope = scope)

    ### DEFINE LOSS ###
    mse_loss = tl.cost.mean_squared_error(net_regression.outputs * 2., labels_sigma, is_mean = True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable = False)

    ### DEFINE OPTIMIZER ###
    t_vars = tl.layers.get_variables_with_name('resNet', True, True)
    optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list = t_vars)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
    print "initializing global variable..."
    tl.layers.initialize_global_variables(sess)
    print "initializing global variable...DONE"

    ### START TRAINING ###
    sess.run(tf.assign(lr_v, lr_init))
    global_step = 0
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_loss, n_iter = 0, 0

        for idx in range(0, len(train_sharp_imgs), batch_size):
            step_time = time.time()

            images_crop = tl.prepro.threading_data(train_sharp_imgs[idx : idx + batch_size], fn = crop_sub_img_fn, is_random = True)

            sigma_random = np.expand_dims(np.around(np.random.uniform(low = 0.0, high = 2.0, size = batch_size), 2), 1)
            images_blur = []
            for i in range(0, len(images_crop)):
                image_blur = gaussian_filter(images_crop[i], sigma_random[i][0])
                images_blur.append(image_blur)

            err, _ = sess.run([mse_loss, optim], {patches_blurred: images_blur, labels_sigma: sigma_random})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse_err: %.6f" % (epoch, n_epoch, n_iter, time.time() - step_time, err))
            total_loss += err
            n_iter += 1
            global_step += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_loss/n_iter)
        print(log)

        ## save model
        if epoch % 100 == 0:
            tl.files.save_ckpt(sess=sess, mode_name='SA_net_{}_init.ckpt'.format(tl.global_flag['mode']), save_dir = checkpoint_dir, var_list = t_vars, global_step = global_step, printable = False)

def evaluate():
    print "Evaluation Start"
    checkpoint_dir = "/data2/junyonglee/sharpness_assessment/checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv

    # Input
    test_sharp_img_list = sorted(tl.files.load_file_list(path = config.TEST.sharp_img_path, regx = '.*.png', printable = False))
    test_sharp_imgs = read_all_imgs(test_sharp_img_list, path = config.TEST.sharp_img_path, n_threads = 32)

    # Model
    patches_blurred = tf.placeholder('float32', [len(test_sharp_imgs), 224, 224, 3], name = 'input_patches')
    with tf.variable_scope('resNet') as scope:
        net_regression = resNet(patches_blurred, is_train = True, reuse = False, scope = scope)
        sigma_value = (net_regression.outputs) * 1.5 + 0.5

    t_vars = tl.layers.get_variables_with_name('resNet', True, True)

    # Initi Session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
    tl.layers.initialize_global_variables(sess)

    # Load checkpoint
    tl.files.load_ckpt(sess=sess, mode_name='SA_net_{}_init.ckpt'.format(tl.global_flag['mode']), save_dir=checkpoint_dir, var_list=t_vars)
    if tl.files.file_exists(checkpoint_dir + '/checkpoint'):
        print "exist"

    # Evalute
    images_crop = tl.prepro.threading_data(test_sharp_imgs[0 : len(test_sharp_imgs)], fn = crop_sub_img_fn, is_random = True)

    sigma_random = np.expand_dims(np.around(np.random.uniform(low = 0.0, high = 2.0, size = len(test_sharp_imgs)), 2), 1)
    images_blur = []
    for i in range(0, len(images_crop)):
        image_blur = gaussian_filter(images_crop[i], sigma_random[i][0])
        images_blur.append(image_blur)

    sigma_out = sess.run(sigma_value, {patches_blurred: images_blur})

    for i in np.arange(len(sigma_out)):
        print "sigma: {}, expected: {}".format(sigma_random[i], sigma_out[i])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='sharp_ass', help='model name')
    parser.add_argument('--is_train', type=str , default='true', help='whether train or not')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['is_train'] = t_or_f(args.is_train)

    if tl.global_flag['is_train']:
        train()
    else:
        evaluate()
