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

lambda_adv = config.TRAIN.lambda_adv
lambda_lr_d = config.TRAIN.lambda_lr_d
lambda_binary = config.TRAIN.lambda_binary
lambda_perceptual = config.TRAIN.lambda_perceptual
lambda_perceptual_c = config.TRAIN.lambda_perceptual_c

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
        if tl.global_flag['is_pretrain']:
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
    # train_synthetic_binary_map_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.synthetic_binary_map_path, regx = '.*', printable = False)))
    
    train_real_img_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.real_img_path, regx = '.*', printable = False)))
    train_real_binary_map_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.real_binary_map_path, regx = '.*', printable = False)))

    ## DEFINE MODEL
    # input
    with tf.variable_scope('input'):
        patches_synthetic = tf.placeholder('float32', [None, h, w, 3], name = 'input_synthetic')
        labels_synthetic_defocus = tf.placeholder('float32', [None, h, w, 1], name = 'labels_synthetic_defocus')
        labels_synthetic_binary = tf.placeholder('float32', [None, h, w, 1], name = 'labels_synthetic_binary')

        patches_real = tf.placeholder('float32', [None, h, w, 3], name = 'input_real')
        labels_real_binary = tf.placeholder('float32', [None, h, w, 1], name = 'labels_real_binary')

    # model
    with tf.variable_scope('main_net') as scope:
        with tf.variable_scope('defocus_net') as scope:
            with tf.variable_scope('encoder') as scope:
                net_vgg, feats_synthetic_down, _, _ = Vgg19_simple_api(patches_synthetic, reuse = False, scope = scope)
                _, feats_real_down, _, _ = Vgg19_simple_api(patches_real, reuse = True, scope = scope)
            with tf.variable_scope('decoder') as scope:
                output_synthetic_defocus, feats_synthetic_up, _ = UNet_up(patches_synthetic, feats_synthetic_down, is_train = True, reuse = False, scope = scope)
                output_real_defocus, _, _ = UNet_up(patches_real, feats_real_down, is_train = True, reuse = True, scope = scope)
        with tf.variable_scope('binary_net') as scope:
            output_real_binary_logits, output_real_binary = Binary_Net(output_real_defocus, is_train = True, reuse = False, scope = scope)

    with tf.variable_scope('discriminator') as scope:
        with tf.variable_scope('feature') as scope:
            d_feature_logits_synthetic, d_feature_synthetic = feature_discriminator(feats_synthetic_down[4], is_train = True, reuse = False, scope = scope)
            d_feature_logits_real, d_feature_real = feature_discriminator(feats_real_down[4], is_train = True, reuse = True, scope = scope)
        with tf.variable_scope('perceptual') as scope:
            output_synthetic_defocus_3c = tf.concat([output_synthetic_defocus, output_synthetic_defocus, output_synthetic_defocus], axis = 3)
            net_vgg_perceptual, _, perceptual_synthetic_out, logits_perceptual_out = Vgg19_simple_api(output_synthetic_defocus_3c, reuse = False, scope = scope)
            labels_synthetic_defocus_3c = tf.concat([labels_synthetic_defocus, labels_synthetic_defocus, labels_synthetic_defocus], axis = 3)
            _, _, perceptual_synthetic_label, logits_perceptual_label = Vgg19_simple_api(labels_synthetic_defocus_3c, reuse = True, scope = scope)

    ## DEFINE LOSS
    with tf.variable_scope('loss'):
        with tf.variable_scope('discriminator'):
            with tf.variable_scope('feature'):
                loss_synthetic_d_feature = tl.cost.sigmoid_cross_entropy(d_feature_logits_synthetic, tf.zeros_like(d_feature_logits_synthetic), name = 'synthetic')
                loss_real_d_feature = tl.cost.sigmoid_cross_entropy(d_feature_logits_real, tf.ones_like(d_feature_logits_real), name = 'real')
                loss_d_feature = tf.identity((loss_synthetic_d_feature + loss_real_d_feature) / 2. * lambda_adv, name = 'total')

            with tf.variable_scope('perceptual'):
                loss_out_d_perceptual = tl.cost.sigmoid_cross_entropy(logits_perceptual_out, tf.zeros_like(logits_perceptual_out), name = 'out')
                loss_label_d_perceptual = tl.cost.sigmoid_cross_entropy(logits_perceptual_label, tf.ones_like(logits_perceptual_label), name = 'label')
                loss_d_percetpcual = tf.identity((loss_out_d_perceptual + loss_label_d_perceptual) * lambda_perceptual_c, name = 'total')

            loss_d = tf.identity(loss_d_feature + loss_d_percetpcual, name = 'total')

        with tf.variable_scope('generator'):
            with tf.variable_scope('feature'):
                loss_synthetic_g_feature = tl.cost.sigmoid_cross_entropy(d_feature_logits_synthetic, tf.ones_like(d_feature_logits_synthetic), name = 'synthetic')
                loss_real_g_feature = tl.cost.sigmoid_cross_entropy(d_feature_logits_real, tf.zeros_like(d_feature_logits_real), name = 'real')
                loss_g_feature = tf.identity((loss_synthetic_g_feature + loss_real_g_feature) / 2., name = 'total')

            loss_g = tf.identity(loss_g_feature * lambda_adv, name = 'total')

        with tf.variable_scope('defocus'):
            loss_defocus = tl.cost.mean_squared_error(output_synthetic_defocus, labels_synthetic_defocus, is_mean = True, name = 'synthetic')
            # loss_defocus = tl.cost.absolute_difference_error(output_synthetic_defocus, labels_synthetic_defocus, is_mean = True)
        with tf.variable_scope('auxilary'):
            labels_layer = InputLayer(labels_synthetic_defocus)
            loss_aux_1 = tl.cost.mean_squared_error(feats_synthetic_up[0],
                DownSampling2dLayer(labels_layer, (1/16., 1/16.), method = 1, align_corners=True).outputs, is_mean = True, name = 'aux1')
            loss_aux_2 = tl.cost.mean_squared_error(feats_synthetic_up[1],
                DownSampling2dLayer(labels_layer, (1/8., 1/8.), method = 1, align_corners=True).outputs, is_mean = True, name = 'aux2')
            loss_aux_3 = tl.cost.mean_squared_error(feats_synthetic_up[2],
                DownSampling2dLayer(labels_layer, (1/4., 1/4.), method = 1, align_corners=True).outputs, is_mean = True, name = 'aux3')
            loss_aux_4 = tl.cost.mean_squared_error(feats_synthetic_up[3],
                DownSampling2dLayer(labels_layer, (1/2., 1/2.), method = 1, align_corners=True).outputs, is_mean = True, name = 'aux4')
            loss_aux = tf.identity(loss_aux_1 + loss_aux_2 + loss_aux_3 + loss_aux_4, name = 'total')

        with tf.variable_scope('perceptual'):
            loss_synthetic_perceptual = tl.cost.mean_squared_error(perceptual_synthetic_out, perceptual_synthetic_label, is_mean = True, name = 'synthetic')
            loss_perceptual = tf.identity(loss_synthetic_perceptual * lambda_perceptual, name = 'total')

        with tf.variable_scope('binary'):
            loss_real_binary = tl.cost.sigmoid_cross_entropy(output_real_binary_logits, labels_real_binary, name = 'real')
            #loss_real_binary = tl.cost.mean_squared_error(output_real_binary, labels_real_binary, is_mean = True, name = 'real')
            loss_binary = tf.identity(loss_real_binary * lambda_binary, name = 'total')

        loss_main = tf.identity(loss_defocus + loss_binary + loss_perceptual + loss_aux + loss_g, name = 'total')
        loss_init = tf.identity(loss_defocus, name = 'loss_init')

    ## DEFINE OPTIMIZER
    # variables to save / train
    d_vars = tl.layers.get_variables_with_name('discriminator', True, False)
    main_vars = tl.layers.get_variables_with_name('main_net', True, False)
    init_vars = tl.layers.get_variables_with_name('defocus_net', False, False)
    save_vars = tl.layers.get_variables_with_name('main_net', False, False)

    # define optimizer
    with tf.variable_scope('Optimizer'):
        learning_rate = tf.Variable(lr_init, trainable = False)
        learning_rate_init = tf.Variable(lr_init_init, trainable = False)
        optim_d = tf.train.AdamOptimizer(learning_rate * lambda_lr_d, beta1 = beta1).minimize(loss_d, var_list = d_vars)
        optim_main = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(loss_main, var_list = main_vars)
        optim_init = tf.train.AdamOptimizer(learning_rate_init, beta1 = beta1).minimize(loss_init, var_list = init_vars)

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
        loss_sum_init = tf.summary.merge(loss_sum_list_init)

    image_sum_list_init = []
    image_sum_list_init.append(tf.summary.image('1_synthetic_input_init', patches_synthetic))
    image_sum_list_init.append(tf.summary.image('3_synthetic_defocus_out_init', fix_image_tf(output_synthetic_defocus, 1)))
    image_sum_list_init.append(tf.summary.image('4_synthetic_defocus_out_norm_init', fix_image_tf(output_synthetic_defocus, 1)))
    image_sum_list_init.append(tf.summary.image('5_synthetic_defocus_gt_init', fix_image_tf(labels_synthetic_defocus, 1)))
    image_sum_init = tf.summary.merge(image_sum_list_init)

    # for train
    loss_sum_g_list = []
    with tf.variable_scope('loss_generator'):
        loss_sum_g_list.append(tf.summary.scalar('1_total', loss_main))
        loss_sum_g_list.append(tf.summary.scalar('2_g', loss_g))
        loss_sum_g_list.append(tf.summary.scalar('3_defocus', loss_defocus))
        loss_sum_g_list.append(tf.summary.scalar('4_perceptual', loss_perceptual))
        loss_sum_g_list.append(tf.summary.scalar('5_auxilary', loss_aux))
        loss_sum_g_list.append(tf.summary.scalar('6_binary', loss_binary))
    loss_sum_g = tf.summary.merge(loss_sum_g_list)

    loss_sum_d_list = []
    with tf.variable_scope('loss_discriminator'):
        loss_sum_d_list.append(tf.summary.scalar('1_total', loss_d))
        loss_sum_d_list.append(tf.summary.scalar('2_d_feature', loss_d_feature))
        loss_sum_d_list.append(tf.summary.scalar('3_d_perceptual', loss_d_percetpcual))
    loss_sum_d = tf.summary.merge(loss_sum_d_list)

    image_sum_list = []
    image_sum_list.append(tf.summary.image('1_synthetic_input', patches_synthetic))
    image_sum_list.append(tf.summary.image('2_synthetic_defocus_out', fix_image_tf(output_synthetic_defocus, 1)))
    image_sum_list.append(tf.summary.image('3_synthetic_defocus_gt', fix_image_tf(labels_synthetic_defocus, 1)))
    image_sum_list.append(tf.summary.image('4_real_input', patches_real))
    image_sum_list.append(tf.summary.image('5_real_defocus_out', fix_image_tf(output_real_defocus, 1)))
    image_sum_list.append(tf.summary.image('6_real_binary_out', fix_image_tf(output_real_binary, 1)))
    image_sum_list.append(tf.summary.image('7_real_binary_gt', fix_image_tf(labels_real_binary, 1)))
    image_sum = tf.summary.merge(image_sum_list)

    ## INITIALIZE SESSION
    tl.layers.initialize_global_variables(sess)
    ## LOAD VGG
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted( npz.items() ):
        if val[0] == 'fc6':
            break;
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    tl.files.assign_params(sess, params, net_vgg_perceptual)

    tl.files.load_and_assign_npz_dict(name = init_dir + '/{}_init.npz'.format(tl.global_flag['mode']), sess = sess)
    if tl.global_flag['is_pretrain']:
        print '*****************************************'
        print '           PRE-TRAINING START'
        print '*****************************************'
        global_step = 0
        for epoch in range(0, n_epoch_init):
            total_loss_init, n_iter = 0, 0
            # reload image list
            train_synthetic_img_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.synthetic_img_path, regx = '.*', printable = False)))
            train_defocus_map_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.defocus_map_path, regx = '.*', printable = False)))

            # shuffle datasets
            shuffle_index = np.arange(len(train_synthetic_img_list))
            np.random.shuffle(shuffle_index)
            train_synthetic_img_list = train_synthetic_img_list[shuffle_index]
            train_defocus_map_list = train_defocus_map_list[shuffle_index]

            epoch_time = time.time()
            for idx in range(0, len(train_synthetic_img_list), batch_size_init):
                step_time = time.time()
                ## READ DATA
                # read synthetic data
                b_idx = (idx + np.arange(batch_size_init)) % len(train_synthetic_img_list)
                synthetic_images_blur = read_all_imgs(train_synthetic_img_list[b_idx], path = config.TRAIN.synthetic_img_path, mode = 'RGB')
                synthetic_defocus_maps = read_all_imgs(train_defocus_map_list[b_idx], path = config.TRAIN.defocus_map_path, mode = 'DEPTH')

                synthetic_images_blur, synthetic_defocus_maps = \
                         crop_pair_with_different_shape_images_2(synthetic_images_blur, synthetic_defocus_maps, [h, w])
               
                err_init, lr, _ = \
                        sess.run([loss_init, learning_rate_init, optim_init], {patches_synthetic: synthetic_images_blur, labels_synthetic_defocus: synthetic_defocus_maps})

                print('[%s] Ep [%2d/%2d] %4d/%4d time: %4.2fs, err_init: %1.2e, lr: %1.2e' % \
                    (tl.global_flag['mode'], epoch, n_epoch_init, n_iter, len(train_synthetic_img_list)/batch_size_init, time.time() - step_time, err_init, lr))

                if global_step % config.TRAIN.write_log_every == 0:
                    summary_loss_init, summary_image_init = sess.run([loss_sum_init, image_sum_init], {patches_synthetic: synthetic_images_blur, labels_synthetic_defocus: synthetic_defocus_maps})
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
                tl.files.save_npz_dict(init_vars, name = init_dir + '/{}_init.npz'.format(tl.global_flag['mode']), sess = sess)

        tl.files.save_npz_dict(init_vars, name = init_dir + '/{}_init.npz'.format(tl.global_flag['mode']), sess = sess)
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
            #discriminator
            feed_dict = {patches_synthetic: synthetic_images_blur, patches_real: real_images_blur, labels_synthetic_defocus: synthetic_defocus_maps}
            # d_synthetic, d_real = sess.run([d_feature_synthetic, d_feature_real], feed_dict)
            # d_acc = get_disc_accuracy([d_synthetic, d_real], [0, 1])
            d_count = 0
            # while d_acc < 0.5:
            #     _ = sess.run(optim_d, feed_dict)
            #     d_synthetic, d_real = sess.run([d_feature_synthetic, d_feature_real], feed_dict)
            #     d_acc = get_disc_accuracy([d_synthetic, d_real], [0, 1])
            #     d_count = d_count + 1
            _ = sess.run(optim_d, feed_dict)

            #generator
            feed_dict = {patches_synthetic: synthetic_images_blur, labels_synthetic_defocus: synthetic_defocus_maps, labels_synthetic_binary: synthetic_binary_maps, patches_real: real_images_blur, labels_real_binary: real_binary_maps}
            # d_synthetic, d_real = sess.run([d_feature_synthetic, d_feature_real], feed_dict)
            # g_acc = get_disc_accuracy([d_synthetic, d_real], [1, 0])
            g_count = 0
            # while g_count == 0 or g_acc < 0.7:
            #     err_g, _ = sess.run([loss_g, optim_main], feed_dict)
            #     d_synthetic, d_real = sess.run([d_feature_synthetic, d_feature_real], feed_dict)
            #     g_acc = get_disc_accuracy([d_synthetic, d_real], [1, 0])
            #     g_count = g_count + 1
            _ = sess.run(optim_main, feed_dict)

            #log
            err_main, err_g, err_d, d_synthetic, d_real, lr = \
            sess.run([loss_main, loss_g, loss_d, d_feature_synthetic, d_feature_real, learning_rate], feed_dict)
            d_acc = get_disc_accuracy([d_synthetic, d_real], [0, 1])
            g_acc = get_disc_accuracy([d_synthetic, d_real], [1, 0])

            print('[%s] Ep [%2d/%2d] %4d/%4d time: %4.2fs, err[main: %1.2e, g(acc, #): %1.2e(%1.2f, %d), d(acc, #): %1.2e(%1.2f, %d)], lr: %1.2e' % \
                (tl.global_flag['mode'], epoch, n_epoch, n_iter, len(train_synthetic_img_list)/batch_size, time.time() - step_time, err_main, err_g, g_acc, g_count, err_d, d_acc, d_count, lr))

            ## SAVE LOGS
            # save loss & image log
            if global_step % config.TRAIN.write_log_every == 0:
                summary_loss_g, summary_loss_d, summary_image  = sess.run([loss_sum_g, loss_sum_d, image_sum], {patches_synthetic: synthetic_images_blur, labels_synthetic_defocus: synthetic_defocus_maps, patches_real: real_images_blur, labels_synthetic_binary: synthetic_binary_maps, labels_real_binary: real_binary_maps})
                writer_scalar.add_summary(summary_loss_d, global_step)
                writer_scalar.add_summary(summary_loss_g, global_step)
                writer_image.add_summary(summary_image, global_step)
            # save checkpoint
            if global_step != 0 and global_step % config.TRAIN.write_ckpt_every == 0:
                shutil.rmtree(ckpt_dir, ignore_errors = True)
                tl.files.save_ckpt(sess = sess, mode_name = '{}.ckpt'.format(tl.global_flag['mode']), save_dir = ckpt_dir, var_list = save_vars, global_step = global_step, printable = False)
            # save samples
            if global_step != 0 and global_step % config.TRAIN.write_sample_every == 0:
                synthetic_defocus_out, real_defocus_out, real_binary_out = sess.run([output_synthetic_defocus, output_real_defocus, output_real_binary], {patches_synthetic: synthetic_images_blur, patches_real: real_images_blur, labels_real_binary: real_binary_maps })
                save_images(synthetic_images_blur, [ni, ni], sample_dir + '/{}_{}_1_synthetic_input.png'.format(epoch, global_step))
                save_images(norm_image(synthetic_defocus_out), [ni, ni], sample_dir + '/{}_{}_2_synthetic_defocus_out_norm.png'.format(epoch, global_step))
                save_images(norm_image(synthetic_defocus_maps), [ni, ni], sample_dir + '/{}_{}_3_synthetic_defocus_gt.png'.format(epoch, global_step))
                save_images(real_images_blur, [ni, ni], sample_dir + '/{}_{}_4_real_input.png'.format(epoch, global_step))
                save_images(norm_image(real_defocus_out), [ni, ni], sample_dir + '/{}_{}_5_real_defocus_out_norm.png'.format(epoch, global_step))
                save_images(real_binary_out, [ni, ni], sample_dir + '/{}_{}_6_real_binary_out.png'.format(epoch, global_step))
                save_images(real_binary_maps, [ni, ni], sample_dir + '/{}_{}_7_real_binary_gt.png'.format(epoch, global_step))

            total_loss += err_main
            n_iter += 1
            global_step += 1
            
        print('[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %1.2e' % (epoch, n_epoch, time.time() - epoch_time, total_loss/n_iter))
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
        with tf.variable_scope('main_net') as scope:
            with tf.variable_scope('defocus_net') as scope:
                with tf.variable_scope('encoder') as scope:
                    _, feats, _, _, feats_down = Vgg19_simple_api(patches_blurred, reuse = reuse, scope = scope)
                with tf.variable_scope('decoder') as scope:
                    output_defocus, feats_up, refine_lists = UNet_up(patches_blurred, feats, is_train = False, reuse = reuse, scope = scope)
            with tf.variable_scope('binary_net') as scope:
                _, output_binary = Binary_Net(output_defocus, is_train = False, reuse = reuse, scope = scope)

        #save_vars = tl.layers.get_variables_with_name('defocus_net', False, False)
        save_vars = tl.layers.get_variables_with_name('main_net', False, False)

        # init session
        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
        tl.layers.initialize_global_variables(sess)

        # load checkpoint
        tl.files.load_ckpt(sess = sess, mode_name = '{}.ckpt'.format(tl.global_flag['mode']), save_dir = ckpt_dir, var_list = save_vars)

        # run network
        print 'processing {} ...'.format(test_blur_img_list[i])
        processing_time = time.time()
        defocus_map, binary_map, feats_down_out, feats_up_out, refine_lists_out = sess.run([output_defocus, output_binary, feats_down, feats_up, refine_lists], {patches_blurred: np.expand_dims(test_blur_img, axis = 0)})
        defocus_map = np.squeeze(1 - defocus_map)
        defocus_map_norm = defocus_map - defocus_map.min()
        defocus_map_norm = defocus_map_norm / defocus_map_norm.max()
        binary_map = np.squeeze(1 - binary_map)
        print 'processing {} ... Done [{:.3f}s]'.format(test_blur_img_list[i], time.time() - processing_time)
        
        tl.files.exists_or_mkdir(sample_dir, verbose = False)
        scipy.misc.toimage(test_blur_img, cmin = 0., cmax = 1.).save(sample_dir + '/{}_1_input.png'.format(i))
        scipy.misc.toimage(defocus_map, cmin = 0., cmax = 1.).save(sample_dir + '/{}_2_defocus_map_out.png'.format(i))
        scipy.misc.toimage(defocus_map_norm, cmin = 0., cmax = 1.).save(sample_dir + '/{}_3_defocus_map_norm_out.png'.format(i))
        scipy.misc.toimage(binary_map, cmin = 0., cmax = 1.).save(sample_dir + '/{}_4_binary_map_out.png'.format(i))
        scipy.misc.toimage(np.squeeze(1 - test_gt_imgs[i]), cmin = 0., cmax = 1.).save(sample_dir + '/{}_5_binary_map_gt.png'.format(i))

        for j in np.arange(len(feats_down_out)):
            feats_down_out[j] = np.squeeze(feats_down_out[j])
            feats_down_out[j] = np.transpose(feats_down_out[j], [2, 0, 1])
            ni = len(feats_down_out[j])
            ni = int(np.ceil(np.sqrt(ni)))
            save_images(norm_image(feats_down_out[j], (1, 2)), [ni, ni], sample_dir + '/{}_6_feats_down_{}.png'.format(i, j+1))

        for j in np.arange(len(feats_up_out)):
            scipy.misc.toimage(np.squeeze(feats_up_out[j]), cmin = 0., cmax = 1.).save(sample_dir + '/{}_7_feat_up_{}.png'.format(i, j+1))

        for j in np.arange(len(refine_lists_out)):
            refine_lists_out[j] = np.squeeze(refine_lists_out[j])
            refine_lists_out[j] = np.transpose(refine_lists_out[j], [2, 0, 1])
            save_images(norm_image(refine_lists_out[j], (1, 2)), [8, 8], sample_dir + '/{}_8_refine_{}.png'.format(i, j+1))

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
