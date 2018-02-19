import tensorlayer as tl
import numpy as np
import math
from config import config, log_config
from utils import *
from model import *
import matplotlib
import datetime

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

h = config.TRAIN.height
w = config.TRAIN.width

ni = int(math.ceil(np.sqrt(batch_size)))

def read_all_imgs(img_list, path='', n_threads=32, mode = 'RGB'):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        if mode is 'RGB':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGB_fn, path=path)
        elif mode is 'GRAY':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_GRAY_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def train():
    checkpoint_dir = "/data2/junyonglee/sharpness_assessment/checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_config(checkpoint_dir + '/config', config)

    save_dir_sample = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)

    train_sharp_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.sharp_img_path, regx='.*', printable=False))
    train_sharp_imgs = read_all_imgs(train_sharp_img_list, path=config.TRAIN.sharp_img_path, n_threads=batch_size, mode = 'RGB')

    ### DEFINE MODEL ###
    patches_blurred = tf.placeholder('float32', [batch_size, h, w, 3], name = 'input_patches')
    labels_sigma = tf.placeholder('float32', [batch_size, h, w, 1], name = 'lables')

    with tf.variable_scope('unet') as scope:
        output = unet(patches_blurred, training=False, reuse = False, scope = scope)

    ### DEFINE LOSS ###
    loss = tl.cost.mean_squared_error(output, labels_sigma, is_mean = True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable = False)

    ### DEFINE OPTIMIZER ###
    t_vars = tf.trainable_variables()
    optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list = t_vars)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
    print "initializing global variable..."
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
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

            sigma_random = np.expand_dims(np.around(np.random.uniform(low = 0.0, high = 1.0, size = batch_size), 2), 1)
            images_blur = tl.prepro.threading_data(
                [_ for _ in zip(train_sharp_imgs[idx : idx + batch_size], sigma_random * 15.)], fn=blur_crop_edge_sub_imgs_fn)
            
            '''
            images_blur, images_sharp, images_edge = images_blur.transpose((1, 0, 2, 3, 4))
            for i in np.arange(len(images_blur)):
                scipy.misc.imsave(save_dir_sample+"/sample_{}_2_blur.png".format(i), images_blur[i])
                scipy.misc.imsave(save_dir_sample+"/sample_{}_1_sharp.png".format(i), images_sharp[i])
                scipy.misc.imsave(save_dir_sample+"/sample_{}_4_sub.png".format(i), images_blur[i] - images_sharp[i])
                scipy.misc.imsave(save_dir_sample+"/sample_{}_3_edge.png".format(i), np.squeeze(images_edge[i, :, :, 0].astype(np.float)))

            return
            '''

            sigma_image = np.zeros([batch_size, h, w, 1])
            for i in range(batch_size):
                sigma_image[i, :] = sigma_random[i]

            err, _ = sess.run([loss, optim], {patches_blurred: images_blur, labels_sigma: sigma_image})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, err: %.6f" % (epoch, n_epoch, n_iter, time.time() - step_time, err))
            total_loss += err
            n_iter += 1
            global_step += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_loss/n_iter)
        total_loss, n_iter = 0, 0
        print(log)

        ## save model
        if epoch % 100 == 0:
            tl.files.save_ckpt(sess=sess, mode_name='SA_net_{}_init.ckpt'.format(tl.global_flag['mode']), save_dir = checkpoint_dir, var_list = t_vars, global_step = global_step, printable = False)

def evaluate():
    print "Evaluation Start"
    checkpoint_dir = "/data2/junyonglee/sharpness_assessment/checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    date = datetime.datetime.now().strftime("%y.%m.%d")
    time = datetime.datetime.now().strftime("%H%M")
    save_dir_sample = "samples/{}/{}/{}".format(tl.global_flag['mode'], date, time)
    tl.files.exists_or_mkdir(save_dir_sample)

    # Input
    test_blur_img_list = sorted(tl.files.load_file_list(path=config.TEST.blur_img_path, regx='.*', printable=False))
    test_blur_imgs = read_all_imgs(test_blur_img_list, path=config.TEST.blur_img_path, n_threads=batch_size, mode = 'RGB')

    # Model
    patches_blurred = tf.placeholder('float32', [1, None, None, 3], name = 'input_patches')
    with tf.variable_scope('unet') as scope:
        sigma_value = unet(patches_blurred, training = False, reuse = False, scope = scope)

    t_vars = tf.trainable_variables()

    # Init Session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Load checkpoint
    tl.files.load_ckpt(sess=sess, mode_name='SA_net_{}_init.ckpt'.format(tl.global_flag['mode']), save_dir=checkpoint_dir, var_list=t_vars)
    if tl.files.file_exists(checkpoint_dir + '/checkpoint'):
        print "****************"
        print "checkpoint exist"
        print "****************"

    # Evalute
    '''
    # sigma regression
    images_crop = tl.prepro.threading_data(test_blur_imgs[0 : len(test_blur_imgs)], fn = crop_sub_img_fn, is_random = True)

    sigma_random = np.expand_dims(np.around(np.random.uniform(low = 0.0, high = 2.0, size = len(test_blur_imgs)), 2), 1)
    images_blur = []
    for i in range(0, len(images_crop)):
        image_blur = gaussian_filter(images_crop[i], sigma_random[i][0])
        images_blur.append(image_blur)

    sigma_out = sess.run(sigma_value, {patches_blurred: images_blur})

    for i in np.arange(len(sigma_out)):
        print "sigma: {}, expected: {}".format(sigma_random[i], sigma_out[i])
    '''
    # Blur map
    for i in np.arange(len(test_blur_imgs)):
        print "processing {} ...".format(test_blur_img_list[i])
        shape = test_blur_imgs[i].shape

        if shape[0] % 2 is not 0:
            #test_blur_imgs[i] = test_blur_imgs[i, :-1, :, :]
            continue
        elif (shape[0] / 2) % 2 is not 0:
            continue
        elif (shape[0] / 2 /2) % 2 is not 0:
            continue
        elif (shape[0] / 2 /2/2) % 2 is not 0:
            continue

        if shape[1] % 2 is not 0:
            #test_blur_imgs[i] = test_blur_imgs[i, :, :-1, :]
            continue
        elif (shape[1] / 2) % 2 is not 0:
            continue
        elif (shape[1] / 2 /2) % 2 is not 0:
            continue
        elif (shape[1] / 2 /2/2) % 2 is not 0:
            continue
        # blur_map
        blur_map = sess.run(sigma_value, {patches_blurred: np.expand_dims(test_blur_imgs[i], axis=0)})
        blur_map = np.squeeze(blur_map)
        blur_map = 1. - blur_map
        blur_map_norm = (blur_map - blur_map.min(axis = None))
        blur_map_norm = blur_map_norm / blur_map_norm.max(axis = None)

        print "processing {} ... Done".format(test_blur_img_list[i])
        scipy.misc.toimage(blur_map, cmin=0., cmax=1.).save(save_dir_sample + "/{}_blur.png".format(i))
        scipy.misc.toimage(blur_map_norm, cmin=0., cmax=1.).save(save_dir_sample + "/{}_blur_norm.png".format(i))
        #scipy.misc.imsave(save_dir_sample + "/{}_blur.png".format(i), blur_map)
        #scipy.misc.imsave(save_dir_sample + "/{}_blur_norm.png".format(i), blur_map_norm)
        scipy.misc.imsave(save_dir_sample + "/{}_gt.png".format(i), test_blur_imgs[i])

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
