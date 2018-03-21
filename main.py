import tensorlayer as tl
import numpy as np
import math
from config import config, log_config
from utils import *
from unet import *
from random import shuffle
import matplotlib
import datetime
import time

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
    for idx in range(0, len(img_list), n_threads):
        if idx + n_threads > len(img_list):
            break;
            
        b_imgs_list = img_list[idx : idx + n_threads]
        if mode is 'RGB':
            if idx == 0:
                b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGB_fn, path=path)
            else:
                imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGB_fn, path=path)
                b_imgs = np.concatenate((b_imgs, imgs), axis = 0)
                
        elif mode is 'GRAY':
            if idx == 0:
                b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_GRAY_fn, path=path)
            else:
                imgs= tl.prepro.threading_data(b_imgs_list, fn=get_imgs_GRAY_fn, path=path)
                b_imgs = np.concatenate((b_imgs, imgs), axis = 0)
        
        #print('read %d from %s' % (len(b_imgs), path))
        
    return b_imgs

def train():
    ## LOG
    checkpoint_dir = "/data2/junyonglee/sharpness_assessment/checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_config(checkpoint_dir + '/config', config)

    date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_training_output = "samples/{}/{}/0_training".format(tl.global_flag['mode'], date)
    tl.files.exists_or_mkdir(save_dir_training_output)

    ## DATA
    train_synthetic_img_list = np.array(sorted(tl.files.load_file_list(path=config.TRAIN.synthetic_img_path, regx='.*', printable=False)))
    train_defocus_map_list = np.array(sorted(tl.files.load_file_list(path=config.TRAIN.defocus_map_path, regx='.*', printable=False)))
    
    train_real_img_list = np.array(sorted(tl.files.load_file_list(path=config.TRAIN.real_img_path, regx='.*', printable=False)))
    train_binary_map_list = np.array(sorted(tl.files.load_file_list(path=config.TRAIN.binary_map_path, regx='.*', printable=False)))
    
    shuffle_index = np.arange(len(train_synthetic_img_list))
    np.random.shuffle(shuffle_index)

    train_synthetic_img_list = train_synthetic_img_list[shuffle_index]
    train_defocus_map_list = train_defocus_map_list[shuffle_index]

    ### DEFINE MODEL ###
    patches_synthetic = tf.placeholder('float32', [batch_size, h, w, 3], name = 'input_synthetic')
    labels_synthetic = tf.placeholder('float32', [batch_size, h, w, 1], name = 'lables_synthetic')
    
    patches_real = tf.placeholder('float32', [batch_size, h, w, 3], name = 'input_real')
    labels_real = tf.placeholder('float32', [batch_size, h, w, 1], name = 'lables_real')

    with tf.variable_scope('unet') as scope:
        _, output_synthetic = UNet(patches_synthetic, is_train=True, reuse = False, scope = scope)
        _, output_real = UNet(patches_real, is_train=True, reuse = True, scope = scope)

    ### DEFINE LOSS ###
    loss_synthetic = tl.cost.mean_squared_error(output_synthetic, labels_synthetic, is_mean = True)
    loss_real = 1. - tf_ssim(output_real, labels_real)
    #loss_real = 1. - tf_ms_ssim(output_real, labels_real, batch_size)
    loss = (loss_synthetic + loss_real) / 2.

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable = False)

    ### DEFINE OPTIMIZER ###
    t_vars = tl.layers.get_variables_with_name('unet', True, True)
    a_vars = tl.layers.get_variables_with_name('unet', False, True)

    optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list = t_vars)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
    print "initializing global variable..."
    tl.layers.initialize_global_variables(sess)
    print "initializing global variable...DONE"

    ### START TRAINING ###
    sess.run(tf.assign(lr_v, lr_init))
    global_step = 0
    for epoch in range(0, n_epoch + 1):
        total_loss, n_iter = 0, 0
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

        for idx in range(0, len(train_synthetic_img_list), batch_size):
            step_time = time.time()
            # read synthetic data
            b_idx = (idx + np.arange(batch_size)) % len(train_synthetic_img_list)
            synthetic_images_blur = read_all_imgs(train_synthetic_img_list[b_idx], path=config.TRAIN.synthetic_img_path, n_threads=batch_size, mode = 'RGB')
            defocus_maps = read_all_imgs(train_defocus_map_list[b_idx], path=config.TRAIN.defocus_map_path, n_threads=batch_size, mode = 'GRAY')

            concatenated_images = np.concatenate((synthetic_images_blur, defocus_maps), axis = 3)
            images = tl.prepro.crop_multi(concatenated_images, wrg=h, hrg=w, is_random=True)
            synthetic_images_blur = (images[:, :, :, 0:3] / 127.5) - 1.
            defocus_maps = np.expand_dims(images[:, :, :, 3], axis = 3)

            # read real data #
            b_idx = (idx % len(train_real_img_list) + np.arange(batch_size)) % len(train_real_img_list)
            real_images_blur = read_all_imgs(train_real_img_list[b_idx], path=config.TRAIN.real_img_path, n_threads=batch_size, mode = 'RGB')
            binary_maps = read_all_imgs(train_binary_map_list[b_idx], path=config.TRAIN.binary_map_path, n_threads=batch_size, mode = 'GRAY')
            
            real_image_blur_list = None
            binary_map_list = None
            for i in np.arange(len(real_images_blur)):
                concatenated_images = np.concatenate((real_images_blur[i], binary_maps[i]), axis = 2)
                images = tl.prepro.crop(concatenated_images, wrg=h, hrg=w, is_random=True)
                real_image_blur = np.expand_dims((images[:, :, 0:3] / 127.5 - 1.), axis=0)
                binary_map = np.expand_dims(np.expand_dims(images[:, :, 3], axis=3), axis=0)
                
                real_image_blur_list = np.copy(real_image_blur) if i == 0 else np.concatenate((real_image_blur_list, real_image_blur), axis = 0)
                binary_map_list = np.copy(binary_map) if i == 0 else np.concatenate((binary_map_list, binary_map), axis = 0)
            ###################
                    
            defocus_map_output, binary_map_output, err_synthetic, err_real, _, lr = sess.run([output_synthetic, output_real, loss_synthetic, loss_real, optim, lr_v], {
                patches_synthetic: synthetic_images_blur, labels_synthetic: defocus_maps, patches_real: real_image_blur_list, labels_real: binary_map_list})

            print("Epoch [%2d/%2d] %4d/%4d time: %4.4fs, err_synthetic: %.6f, err_real: %.6f, lr: %.8f" % (epoch, n_epoch, n_iter, len(train_synthetic_img_list)/batch_size, time.time() - step_time, err_synthetic, err_real, lr))
            total_loss += (err_synthetic + err_real) / 2.
            n_iter += 1
            global_step += 1
            
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_loss/n_iter)
        print(log)

        ## save model
        if epoch % 1 == 0:
            tl.files.save_ckpt(sess=sess, mode_name='SA_net_{}_init.ckpt'.format(tl.global_flag['mode']), save_dir = checkpoint_dir, var_list = a_vars, global_step = global_step, printable = False)
            tl.visualize.save_images((synthetic_images_blur[:9] + 1) * 127.5, [3, 3], save_dir_training_output + '/{}_1_synthetic_image.png'.format(epoch))
            tl.visualize.save_images(defocus_map_output[:9], [3, 3], save_dir_training_output + '/{}_2_disp_out.png'.format(epoch))
            tl.visualize.save_images(defocus_maps[:9], [3, 3], save_dir_training_output + '/{}_3_disp_gt.png'.format(epoch))
            tl.visualize.save_images((real_image_blur_list[:9] + 1) * 127.5, [3, 3], save_dir_training_output + '/{}_4_real_image.png'.format(epoch))
            tl.visualize.save_images(binary_map_output[:9], [3, 3], save_dir_training_output + '/{}_5_binary_out.png'.format(epoch))
            tl.visualize.save_images(binary_map_list[:9], [3, 3], save_dir_training_output + '/{}_6_binary_gt.png'.format(epoch))
            
def evaluate():
    print "Evaluation Start"
    checkpoint_dir = "/data2/junyonglee/sharpness_assessment/checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    date = datetime.datetime.now().strftime("%y.%m.%d")
    time = datetime.datetime.now().strftime("%H%M")
    save_dir_sample = "samples/{}/{}/{}".format(tl.global_flag['mode'], date, time)

    # Input
    test_blur_img_list = sorted(tl.files.load_file_list(path=config.TEST.blur_img_path, regx='.*', printable=False))
    test_gt_list = sorted(tl.files.load_file_list(path=config.TEST.binary_map_path, regx='.*', printable=False))
    
    test_blur_imgs = read_all_imgs(test_blur_img_list, path=config.TEST.blur_img_path, n_threads=len(test_blur_img_list), mode = 'RGB')
    test_gt_imgs = read_all_imgs(test_gt_list, path=config.TEST.binary_map_path, n_threads=len(test_gt_list), mode = 'GRAY')
    
    reuse = False
    for i in np.arange(len(test_blur_imgs)):
        # Model

        test_blur_img = np.copy(test_blur_imgs[i])
        test_blur_img = refine_image(test_blur_img)
        shape = test_blur_img.shape

        patches_blurred = tf.placeholder('float32', [1, shape[0], shape[1], 3], name = 'input_patches')
        with tf.variable_scope('unet') as scope:
            _, sigma_value = UNet(patches_blurred, is_train=False, reuse = reuse, scope = scope)

        a_vars = tl.layers.get_variables_with_name('unet', False, False)

        # Init Session
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
        tl.layers.initialize_global_variables(sess)

        # Load checkpoint
        tl.files.load_ckpt(sess=sess, mode_name='SA_net_{}_init.ckpt'.format(tl.global_flag['mode']), save_dir=checkpoint_dir, var_list=a_vars)
        if tl.files.file_exists(checkpoint_dir + '/checkpoint'):
            print "****************"
            print "checkpoint exist"
            print "****************"

        # Blur map
        print "processing {} ...".format(test_blur_img_list[i])
        blur_map = np.squeeze(sess.run(sigma_value, {patches_blurred: np.expand_dims((test_blur_img / 127.5) - 1, axis=0)}))
        print "processing {} ... Done".format(test_blur_img_list[i])
        
        tl.files.exists_or_mkdir(save_dir_sample)
        scipy.misc.imsave(save_dir_sample + "/{}_1_input.png".format(i), test_blur_img)
        scipy.misc.toimage(blur_map, cmin=0., cmax=1.).save(save_dir_sample + "/{}_2_blur.png".format(i))
        scipy.misc.toimage(np.squeeze(test_gt_imgs[i]), cmin=0., cmax=1.).save(save_dir_sample + "/{}_3_blur_gt.png".format(i))

        sess.close()
        reuse = True

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
