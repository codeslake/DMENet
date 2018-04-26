from config import config, log_config
from utils import *
from model import *

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from random import shuffle
import matplotlib
import datetime
import time
import shutil
from flip_gradient import flip_gradient            

batch_size = config.TRAIN.batch_size
batch_size_init = config.TRAIN.batch_size_init
lr_init = config.TRAIN.lr_init
lr_init_init = config.TRAIN.lr_init_init
beta1 = config.TRAIN.beta1

n_epoch = config.TRAIN.n_epoch
n_epoch_init = config.TRAIN.n_epoch_init
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

#lambda_tv = config.TRAIN.lambda_tv
max_coc = config.TRAIN.max_coc

h = config.TRAIN.height
w = config.TRAIN.width

ni = int(np.ceil(np.sqrt(batch_size)))

def train():
    ## CREATE DIRECTORIES
    mode_dir = config.TRAIN.root_dir + '{}'.format(tl.global_flag['mode'])

    ckpt_dir = mode_dir + '/checkpoint'
    init_dir = mode_dir + '/init'
    log_dir_scalar_init = mode_dir + '/log/scalar_init'
    log_dir_image_init = mode_dir + '/log/image_init'
    log_dir_scalar = mode_dir + '/log/scalar'
    log_dir_image = mode_dir + '/log/image'
    sample_dir = mode_dir + '/samples/0_train'
    config_dir = mode_dir + '/config'

    if tl.global_flag['delete_log']:
        shutil.rmtree(ckpt_dir, ignore_errors = True)
        shutil.rmtree(log_dir_scalar_init, ignore_errors = True)
        shutil.rmtree(log_dir_image_init, ignore_errors = True)
        shutil.rmtree(log_dir_scalar, ignore_errors = True)
        shutil.rmtree(log_dir_image, ignore_errors = True)
        shutil.rmtree(sample_dir, ignore_errors = True)
        shutil.rmtree(config_dir, ignore_errors = True)

    tl.files.exists_or_mkdir(ckpt_dir)
    tl.files.exists_or_mkdir(init_dir)
    tl.files.exists_or_mkdir(log_dir_scalar_init)
    tl.files.exists_or_mkdir(log_dir_image_init)
    tl.files.exists_or_mkdir(log_dir_scalar)
    tl.files.exists_or_mkdir(log_dir_image)
    tl.files.exists_or_mkdir(sample_dir)
    tl.files.exists_or_mkdir(config_dir)
    log_config(config_dir, config)

    ## DEFINE SESSION
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
    
    ## READ DATASET LIST
    # train_synthetic_img_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.synthetic_img_path, regx = '.*', printable = False)))
    # train_defocus_map_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.defocus_map_path, regx = '.*', printable = False)))
    #train_synthetic_binary_map_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.synthetic_binary_map_path, regx = '.*', printable = False)))
    
    train_real_img_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.real_img_path, regx = '.*', printable = False)))
    train_real_binary_map_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.real_binary_map_path, regx = '.*', printable = False)))

    ## DEFINE MODEL
    # input
    with tf.variable_scope('input'):
        patches_synthetic = tf.placeholder('float32', [None, h, w, 3], name = 'input_synthetic')
        labels_synthetic_defocus = tf.placeholder('float32', [None, h, w, 1], name = 'labels_synthetic_defocus')
        labels_synthetic_binary = tf.placeholder('float32', [None, h, w, 1], name = 'labels_synthetic_bianry')

        patches_real = tf.placeholder('float32', [None, h, w, 3], name = 'input_real')
        labels_real_binary = tf.placeholder('float32', [None, h, w, 1], name = 'labels_real_binary')

    # model

    with tf.variable_scope('defocus_net') as scope:
        with tf.variable_scope('unet') as scope:
            with tf.variable_scope('unet_down') as scope:
                feats_synthetic = UNet_down(patches_synthetic, is_train = True, reuse = False, scope = scope)
                feats_real = UNet_down(patches_real, is_train = True, reuse = True, scope = scope)

    with tf.variable_scope('discriminator') as scope:
        d_logits_synthetic = SRGAN_d(feats_synthetic, is_train = True, reuse = False, scope = scope)
        d_logits_real = SRGAN_d(feats_real, is_train = True, reuse = True, scope = scope)

    with tf.variable_scope('defocus_net') as scope:
        with tf.variable_scope('unet') as scope:
            with tf.variable_scope('unet_up_defocus_map') as scope:
                output_synthetic_defocus_logits, output_synthetic_defocus = UNet_up(feats_synthetic, is_train = True, reuse = False, scope = scope)
                output_real_defocus_logits, output_real_defocus = UNet_up(feats_real, is_train = True, reuse = True, scope = scope)

        with tf.variable_scope('binary_net') as scope:
            output_synthetic_binary_logits, output_synthetic_binary = Binary_Net(output_synthetic_defocus, is_train = True, reuse = False, scope = scope)
            output_real_binary_logits, output_real_binary = Binary_Net(output_real_defocus, is_train = True, reuse = True, scope = scope)
    
    ## DEFINE LOSS
    with tf.variable_scope('loss'):
        with tf.variable_scope('discriminator'):
            loss_synthetic_d = tl.cost.sigmoid_cross_entropy(d_logits_synthetic, tf.zeros_like(d_logits_synthetic), name = 'synthetic')
            loss_real_d = tl.cost.sigmoid_cross_entropy(d_logits_real, tf.ones_like(d_logits_real), name = 'real')
            loss_d = tf.identity((loss_synthetic_d + loss_real_d), name = 'total')

        with tf.variable_scope('generator'):
            loss_synthetic_g = tl.cost.sigmoid_cross_entropy(d_logits_synthetic, tf.ones_like(d_logits_synthetic), name = 'synthetic')
            #loss_real_g = tl.cost.sigmoid_cross_entropy(d_logits_real, tf.zeros_like(d_logits_real), name = 'real')
            loss_gan = loss_synthetic_g * 1e-3

        with tf.variable_scope('defocus'):
            # loss_defocus = tl.cost.mean_squared_error(output_synthetic_defocus, labels_synthetic_defocus, is_mean = True, name = 'synthetic') * 10.
            loss_defocus = tl.cost.absolute_difference_error(output_synthetic_defocus, labels_synthetic_defocus, is_mean = True)
        with tf.variable_scope('binary'):
            loss_synthetic_binary = tl.cost.sigmoid_cross_entropy(output_synthetic_binary_logits, labels_synthetic_binary, name = 'synthetic')
            loss_real_binary = tl.cost.sigmoid_cross_entropy(output_real_binary_logits, labels_real_binary, name = 'real')
            loss_binary = tf.identity((loss_synthetic_binary + loss_real_binary)/2., name = 'total')

        # with tf.variable_scope('total_variation'):
        #     tv_loss_synthetic = lambda_tv * tf.reduce_sum(tf.image.total_variation(output_synthetic_defocus))
        #     tv_loss_real = lambda_tv * tf.reduce_sum(tf.image.total_variation(output_real_defocus))
        #     tv_loss = (tv_loss_real + tv_loss_synthetic) / 2.

        # loss = tf.identity(loss_defocus + loss_binary + loss_domain + tv_loss, name = 'total')
        # loss_init = tf.identity(loss_defocus + tv_loss_synthetic + loss_synthetic_binary, name = 'loss_init')

        loss_g = tf.identity(loss_defocus + loss_binary + loss_gan, name = 'total')
        loss_init = tf.identity(loss_defocus + loss_synthetic_binary, name = 'loss_init')

    ## DEFINE OPTIMIZER
    # variables to save / train
    d_vars = tl.layers.get_variables_with_name('discriminator', True, False)
    g_vars = tl.layers.get_variables_with_name('unet', True, False)
    save_vars = tl.layers.get_variables_with_name('unet', False, False)

    # define optimizer
    with tf.variable_scope('Optimizer'):
        learning_rate = tf.Variable(lr_init, trainable = False)
        learning_rate_init = tf.Variable(lr_init_init, trainable = False)
        optim_d = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(loss_d, var_list = d_vars)
        optim_g = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(loss_g, var_list = g_vars)
        optim_init = tf.train.AdamOptimizer(learning_rate_init, beta1 = beta1).minimize(loss_init, var_list = save_vars)

    ## DEFINE SUMMARY
    # writer
    writer_scalar = tf.summary.FileWriter(log_dir_scalar, sess.graph, flush_secs=30, filename_suffix = '.loss_log')
    writer_image = tf.summary.FileWriter(log_dir_image, sess.graph, flush_secs=30, filename_suffix = '.image_log')
    if tl.global_flag['is_pretrain']:
        writer_scalar_init = tf.summary.FileWriter(log_dir_scalar_init, sess.graph, flush_secs=30, filename_suffix = '.loss_log_init')
        writer_image_init = tf.summary.FileWriter(log_dir_image_init, sess.graph, flush_secs=30, filename_suffix = '.image_log_init')
 
    # for pretrain
    loss_sum_list_init = []
    with tf.variable_scope('loss_init'):
        loss_sum_list_init.append(tf.summary.scalar('1_total_loss_init', loss_init))
        loss_sum_list_init.append(tf.summary.scalar('2_defocus_loss_init', loss_defocus))
        loss_sum_list_init.append(tf.summary.scalar('3_binary_loss_init', loss_synthetic_binary))
        #loss_sum_list_init.append(tf.summary.scalar('4_tv_loss_init', tv_loss_synthetic))
        loss_sum_init = tf.summary.merge(loss_sum_list_init)

    image_sum_list_init = []
    image_sum_list_init.append(tf.summary.image('1_synthetic_input_init', patches_synthetic))
    image_sum_list_init.append(tf.summary.image('2_synthetic_defocus_out_init', fix_image(output_synthetic_defocus, 1.)))
    image_sum_list_init.append(tf.summary.image('3_synthetic_defocus_out_norm_init', norm_image(output_synthetic_defocus)))
    image_sum_list_init.append(tf.summary.image('4_synthetic_defocus_gt_init', fix_image(labels_synthetic_defocus, 1.)))
    image_sum_list_init.append(tf.summary.image('5_synthetic_binary_out_init', fix_image(output_synthetic_binary, 1.)))
    image_sum_list_init.append(tf.summary.image('6_synthetic_binary_gt_init', fix_image(labels_synthetic_binary, 1.)))
    image_sum_init = tf.summary.merge(image_sum_list_init)

    # for train
    loss_sum_g_list = []
    with tf.variable_scope('loss_generator'):
        loss_sum_g_list.append(tf.summary.scalar('1_total_loss', loss_g))
        loss_sum_g_list.append(tf.summary.scalar('2_gan_loss', loss_gan))
        loss_sum_g_list.append(tf.summary.scalar('3_defocus_loss', loss_defocus))
        loss_sum_g_list.append(tf.summary.scalar('4_binary_loss', loss_binary))
        #loss_sum_g_list.append(tf.summary.scalar('5_tv_loss', tv_loss))
    loss_sum_g = tf.summary.merge(loss_sum_g_list)

    loss_sum_d_lit = []
    with tf.variable_scope('loss_discriminator'):
        loss_sum_d_lit.append(tf.summary.scalar('1_loss_d', loss_d))
    loss_sum_d = tf.summary.merge(loss_sum_d_lit)

    image_sum_list = []
    image_sum_list.append(tf.summary.image('1_synthetic_input', patches_synthetic))
    image_sum_list.append(tf.summary.image('2_synthetic_defocus_out', fix_image(output_synthetic_defocus, 1.)))
    image_sum_list.append(tf.summary.image('3_synthetic_defocus_out_norm', fix_image(output_synthetic_defocus, 1.)))
    image_sum_list.append(tf.summary.image('4_synthetic_defocus_gt', fix_image(labels_synthetic_defocus, 1.)))
    image_sum_list.append(tf.summary.image('5_synthetic_binary_out', fix_image(output_synthetic_binary, 1.)))
    image_sum_list.append(tf.summary.image('6_synthetic_binary_gt', fix_image(labels_synthetic_binary, 1.)))
    image_sum_list.append(tf.summary.image('7_real_input', patches_real))
    image_sum_list.append(tf.summary.image('8_real_defocus_out', fix_image(output_real_defocus, 1.)))
    image_sum_list.append(tf.summary.image('9_real_defocus_out_norm', norm_image(output_real_defocus)))
    image_sum_list.append(tf.summary.image('10_real_binary_out', fix_image(output_real_binary, 1.)))
    image_sum_list.append(tf.summary.image('11_real_binary_gt', fix_image(labels_real_binary, 1.)))
    image_sum = tf.summary.merge(image_sum_list)

    ## INITIALIZE SESSION
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz_dict(name = init_dir + '/{}_init.npz'.format(tl.global_flag['mode']), sess = sess) is False and tl.global_flag['is_pretrain']:
        print '*****************************************'
        print '           PRE-TRAINING START'
        print '*****************************************'
        global_step = 0
        for epoch in range(0, n_epoch_init):
            total_loss_init, n_iter = 0, 0
            # reload image list
            train_synthetic_img_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.synthetic_img_path, regx = '.*', printable = False)))
            train_defocus_map_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.defocus_map_path, regx = '.*', printable = False)))
            train_synthetic_binary_map_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.synthetic_binary_map_path, regx = '.*', printable = False)))
            crop_length = min(len(train_synthetic_img_list), len(train_defocus_map_list), len(train_synthetic_binary_map_list))
            train_synthetic_img_list = train_synthetic_img_list[:crop_length]
            train_defocus_map_list = train_defocus_map_list[:crop_length]
            train_synthetic_binary_map_list = train_synthetic_binary_map_list[:crop_length]

            # shuffle datasets
            shuffle_index = np.arange(len(train_synthetic_img_list))
            np.random.shuffle(shuffle_index)
            train_synthetic_img_list = train_synthetic_img_list[shuffle_index]
            train_defocus_map_list = train_defocus_map_list[shuffle_index]
            train_synthetic_binary_map_list = train_synthetic_binary_map_list[shuffle_index]

            shuffle_index = np.arange(len(train_real_img_list))
            np.random.shuffle(shuffle_index)
            train_real_img_list = train_real_img_list[shuffle_index]
            train_real_binary_map_list = train_real_binary_map_list[shuffle_index]

            epoch_time = time.time()
            for idx in range(0, len(train_synthetic_img_list), batch_size_init):
                step_time = time.time()
                ## READ DATA
                # read synthetic data
                b_idx = (idx + np.arange(batch_size_init)) % len(train_synthetic_img_list)
                synthetic_images_blur = read_all_imgs(train_synthetic_img_list[b_idx], path = config.TRAIN.synthetic_img_path, mode = 'RGB')
                synthetic_defocus_maps = read_all_imgs(train_defocus_map_list[b_idx], path = config.TRAIN.defocus_map_path, mode = 'DEPTH')
                synthetic_binary_maps = read_all_imgs(train_synthetic_binary_map_list[b_idx], path = config.TRAIN.synthetic_binary_map_path, mode = 'GRAY')

                synthetic_images_blur, synthetic_defocus_maps, synthetic_binary_maps = \
                crop_pair_with_different_shape_images_3(synthetic_images_blur, synthetic_defocus_maps, synthetic_binary_maps, [h, w])
               
                err_init, synthetic_defocus_out, synthetic_binary_out, lr, summary_loss_init, summary_image_init, _ = \
                sess.run([loss_init, output_synthetic_defocus, output_synthetic_binary, learning_rate_init, loss_sum_init, image_sum_init, optim_init], {
                    patches_synthetic: synthetic_images_blur,
                    labels_synthetic_defocus: synthetic_defocus_maps,
                    labels_synthetic_binary: synthetic_binary_maps,
                    })

                print('[%s] Ep [%2d/%2d] %4d/%4d time: %4.2fs, err_init: %.3f, lr: %.8f' % \
                    (tl.global_flag['mode'], epoch, n_epoch_init, n_iter, len(train_synthetic_img_list)/batch_size_init, time.time() - step_time, err_init, lr))

                if global_step % config.TRAIN.write_log_every == 0:
                    writer_scalar_init.add_summary(summary_loss_init, global_step)
                    writer_image_init.add_summary(summary_image_init, global_step)

                total_loss_init += err_init
                n_iter += 1
                global_step += 1

            if epoch % config.TRAIN.refresh_image_log_every and epoch != n_epoch_init == 0:
                writer_image_init.close()
                remove_file_end_with(log_dir_image_init, '*.image_log')
                writer_image_init.reopen()

            if epoch % 5:
                tl.files.save_npz_dict(save_vars, name = init_dir + '/{}_init.npz'.format(tl.global_flag['mode']), sess = sess)

        tl.files.save_npz_dict(save_vars, name = init_dir + '/{}_init.npz'.format(tl.global_flag['mode']), sess = sess)
        writer_image_init.close()
        writer_scalar_init.close()

    ## START TRAINING
    print '*****************************************'
    print '             TRAINING START'
    print '*****************************************'
    global_step = 0
    for epoch in range(0, n_epoch + 1):
        total_loss, n_iter = 0, 0
        
        # reload synthetic datasets
        train_synthetic_img_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.synthetic_img_path, regx = '.*', printable = False)))
        train_defocus_map_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.defocus_map_path, regx = '.*', printable = False)))
        train_synthetic_binary_map_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.synthetic_binary_map_path, regx = '.*', printable = False)))
        crop_length = min(len(train_synthetic_img_list), len(train_defocus_map_list), len(train_synthetic_binary_map_list))
        train_synthetic_img_list = train_synthetic_img_list[:crop_length]
        train_defocus_map_list = train_defocus_map_list[:crop_length]
        train_synthetic_binary_map_list = train_synthetic_binary_map_list[:crop_length]

        # shuffle datasets
        shuffle_index = np.arange(len(train_synthetic_img_list))
        np.random.shuffle(shuffle_index)

        train_synthetic_img_list = train_synthetic_img_list[shuffle_index]
        train_defocus_map_list = train_defocus_map_list[shuffle_index]
        train_synthetic_binary_map_list = train_synthetic_binary_map_list[shuffle_index]
        
        shuffle_index = np.arange(len(train_real_img_list))
        np.random.shuffle(shuffle_index)
        train_real_img_list = train_real_img_list[shuffle_index]
        train_real_binary_map_list = train_real_binary_map_list[shuffle_index]

        # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(learning_rate, lr_init * new_lr_decay))
        elif epoch == 0:
            sess.run(tf.assign(learning_rate, lr_init))

        epoch_time = time.time()
        for idx in range(0, len(train_synthetic_img_list), batch_size):
            step_time = time.time()
            
            ## READ DATA
            # read synthetic data
            b_idx = (idx + np.arange(batch_size)) % len(train_synthetic_img_list)
            synthetic_images_blur = read_all_imgs(train_synthetic_img_list[b_idx], path = config.TRAIN.synthetic_img_path, mode = 'RGB')
            synthetic_defocus_maps = read_all_imgs(train_defocus_map_list[b_idx], path = config.TRAIN.defocus_map_path, mode = 'DEPTH')
            synthetic_binary_maps = read_all_imgs(train_synthetic_binary_map_list[b_idx], path = config.TRAIN.synthetic_binary_map_path, mode = 'GRAY')

            synthetic_images_blur, synthetic_defocus_maps, synthetic_binary_maps = crop_pair_with_different_shape_images_3(synthetic_images_blur, synthetic_defocus_maps, synthetic_binary_maps, [h, w])

            # read real data #
            b_idx = (idx % len(train_real_img_list) + np.arange(batch_size)) % len(train_real_img_list)
            real_images_blur = read_all_imgs(train_real_img_list[b_idx], path = config.TRAIN.real_img_path, mode = 'RGB')
            real_binary_maps = read_all_imgs(train_real_binary_map_list[b_idx], path = config.TRAIN.real_binary_map_path, mode = 'GRAY')
            real_images_blur, real_binary_maps = crop_pair_with_different_shape_images_2(real_images_blur, real_binary_maps, [h, w])

            ## RUN NETWORK
            # discriminator
            err_d, summary_loss_d, _ = \
            sess.run([loss_d, loss_sum_d, optim_d], {patches_synthetic: synthetic_images_blur, patches_real: real_images_blur})

            # generator
            err_g, err_def, err_bin, synthetic_defocus_out, synthetic_binary_out, real_defocus_out, real_binary_out, lr, summary_loss_g, summary_image, _ = \
            sess.run([loss_g, loss_defocus, loss_binary, output_synthetic_defocus, output_synthetic_binary, output_real_defocus, output_real_binary, learning_rate, loss_sum_g, image_sum, optim_g], 
                {patches_synthetic: synthetic_images_blur,
                labels_synthetic_defocus: synthetic_defocus_maps,
                labels_synthetic_binary: synthetic_binary_maps,
                patches_real: real_images_blur,
                labels_real_binary: real_binary_maps
                })

            print('[%s] Ep [%2d/%2d] %4d/%4d time: %4.2fs, err_g: %.3f, err_d: %.3f, err_def: %.3f, err_bin: %.3f, lr: %.8f' % \
                (tl.global_flag['mode'], epoch, n_epoch, n_iter, len(train_synthetic_img_list)/batch_size, time.time() - step_time, err_g, err_d, err_def, err_bin, lr))
            
            ## SAVE LOGS
            # save loss & image log
            if global_step % config.TRAIN.write_log_every == 0:
                writer_scalar.add_summary(summary_loss_g, global_step)
                writer_scalar.add_summary(summary_loss_d, global_step)
                writer_image.add_summary(summary_image, global_step)
            # save checkpoint
            if global_step % config.TRAIN.write_ckpt_every == 0:
                shutil.rmtree(ckpt_dir, ignore_errors = True)
                tl.files.save_ckpt(sess = sess, mode_name = '{}.ckpt'.format(tl.global_flag['mode']), save_dir = ckpt_dir, var_list = save_vars, global_step = global_step, printable = False)
            # save samples
            if global_step % config.TRAIN.write_sample_every == 0:
                save_images(synthetic_images_blur, [ni, ni], sample_dir + '/{}_{}_1_synthetic_input.png'.format(epoch, global_step))
                save_images(synthetic_defocus_out, [ni, ni], sample_dir + '/{}_{}_2_synthetic_defocus_out.png'.format(epoch, global_step))
                save_images(synthetic_defocus_maps, [ni, ni], sample_dir + '/{}_{}_3_synthetic_defocus_gt.png'.format(epoch, global_step))
                save_images(synthetic_binary_out, [ni, ni], sample_dir + '/{}_{}_4_synthetic_binary_out.png'.format(epoch, global_step))
                save_images(synthetic_binary_maps, [ni, ni], sample_dir + '/{}_{}_5_synthetic_binary_gt.png'.format(epoch, global_step))
                save_images(real_images_blur, [ni, ni], sample_dir + '/{}_{}_6_real_input.png'.format(epoch, global_step))
                save_images(real_defocus_out, [ni, ni], sample_dir + '/{}_{}_7_real_defocus_out.png'.format(epoch, global_step))
                save_images(real_binary_out, [ni, ni], sample_dir + '/{}_{}_8_real_binary_out.png'.format(epoch, global_step))
                save_images(real_binary_maps, [ni, ni], sample_dir + '/{}_{}_9_real_binary_gt.png'.format(epoch, global_step))

            total_loss += err_g
            n_iter += 1
            global_step += 1
            
        print('[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %.8f' % (epoch, n_epoch, time.time() - epoch_time, total_loss/n_iter))
        # reset image log
        if epoch % config.TRAIN.refresh_image_log_every == 0:
            writer_image.close()
            remove_file_end_with(log_dir_image, '*.image_log')
            writer_image.reopen()

def evaluate():
    print 'Evaluation Start'
    date = datetime.datetime.now().strftime('%Y_%m_%d/%H-%M')
    # directories
    mode_dir = config.TRAIN.root_dir + '{}'.format(tl.global_flag['mode'])
    ckpt_dir = mode_dir + '/checkpoint'
    sample_dir = mode_dir + '/samples/1_test/{}'.format(date)
    
    # input
    test_blur_img_list = np.array(sorted(tl.files.load_file_list(path = config.TEST.real_img_path, regx = '.*', printable = False)))
    test_gt_list = np.array(sorted(tl.files.load_file_list(path = config.TEST.real_binary_map_path, regx = '.*', printable = False)))
    
    test_blur_imgs = read_all_imgs(test_blur_img_list, path = config.TEST.real_img_path, mode = 'RGB')
    test_gt_imgs = read_all_imgs(test_gt_list, path = config.TEST.real_binary_map_path, mode = 'GRAY')
    
    reuse = False
    for i in np.arange(len(test_blur_imgs)):
        test_blur_img = np.copy(test_blur_imgs[i])
        test_blur_img = refine_image(test_blur_img)
        shape = test_blur_img.shape

        patches_blurred = tf.placeholder('float32', [1, shape[0], shape[1], 3], name = 'input_patches')
        # define model
        with tf.variable_scope('defocus_net') as scope:
            with tf.variable_scope('unet') as scope:
                with tf.variable_scope('unet_down') as scope:
                    feats = UNet_down(patches_blurred, is_train = False, reuse = reuse, scope = scope)
                with tf.variable_scope('binary_net') as scope:
                    with tf.variable_scope('unet_up_defocus_map') as scope:
                        _, output_defocus = UNet_up(feats, is_train = False, reuse = reuse, scope = scope)

                _, output_binary = Binary_Net(output_defocus, is_train = False, reuse = reuse, scope = scope)
                        
        save_vars = tl.layers.get_variables_with_name('unet', False, False)

        # init session
        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
        tl.layers.initialize_global_variables(sess)

        # load checkpoint
        tl.files.load_ckpt(sess = sess, mode_name = '{}.ckpt'.format(tl.global_flag['mode']), save_dir = ckpt_dir, var_list = save_vars)

        # run network
        print 'processing {} ...'.format(test_blur_img_list[i])
        processing_time = time.time()
        defocus_map, binary_map = sess.run([output_defocus, output_binary], {patches_blurred: np.expand_dims(test_blur_img, axis = 0)})
        defocus_map = np.squeeze(defocus_map)
        defocus_map_norm = defocus_map - defocus_map.min()
        defocus_map_norm = defocus_map_norm / defocus_map_norm.max()
        binary_map = np.squeeze(binary_map)
        print 'processing {} ... Done [{:.3f}s]'.format(test_blur_img_list[i], time.time() - processing_time)
        
        tl.files.exists_or_mkdir(sample_dir, verbose = False)
        scipy.misc.toimage(test_blur_img, cmin = 0., cmax = 1.).save(sample_dir + '/{}_1_input.png'.format(i))
        scipy.misc.toimage(defocus_map, cmin = 0., cmax = 1.).save(sample_dir + '/{}_2_defocus_map_out.png'.format(i))
        scipy.misc.toimage(defocus_map_norm, cmin = 0., cmax = 1.).save(sample_dir + '/{}_3_defocus_map_norm_out.png'.format(i))
        scipy.misc.toimage(binary_map, cmin = 0., cmax = 1.).save(sample_dir + '/{}_4_binary_map_out.png'.format(i))
        scipy.misc.toimage(np.squeeze(test_gt_imgs[i]), cmin = 0., cmax = 1.).save(sample_dir + '/{}_5_binary_map_gt.png'.format(i))

        sess.close()
        reuse = True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, default = 'sharp_ass', help = 'model name')
    parser.add_argument('--is_train', type = str , default = 'true', help = 'whether to train or not')
    parser.add_argument('--is_pretrain', type = str , default = 'false', help = 'whether to pretrain or not')
    parser.add_argument('--delete_log', type = str , default = 'false', help = 'whether to delete log or not')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['is_train'] = t_or_f(args.is_train)
    tl.global_flag['is_pretrain'] = t_or_f(args.is_pretrain)
    tl.global_flag['delete_log'] = t_or_f(args.delete_log)
    

    if tl.global_flag['is_train']:
        train()
    else:
        evaluate()
