from easydict import EasyDict as edict
import json
import os

config = edict()
config.TRAIN = edict()
config.TEST = edict()

## Adam
config.TRAIN.batch_size = 3
config.TRAIN.batch_size_init = 8
config.TRAIN.lr_init = 1e-4
config.TRAIN.lr_init_init = 1e-4
config.TRAIN.beta1 = 0.9

# learning rate
config.TRAIN.n_epoch = 10000
config.TRAIN.n_epoch_init = 10
config.TRAIN.lr_decay = 0.8
config.TRAIN.decay_every = 20

## adversarial loss coefficient
config.TRAIN.lambda_adv = 1e-3

## discriminator lr coefficient
config.TRAIN.lambda_lr_d = 1

## binary loss coefficient
config.TRAIN.lambda_binary = 2e-2

## perceptual loss coefficient
config.TRAIN.lambda_perceptual = 1e-4

### TRAIN DATSET PATH
data_offset = './datasets/'
# data_offset = '/data1/junyonglee/defocus_map_estimation/DMENet/'
config.TRAIN.synthetic_img_path = os.path.join(data_offset, 'train/SYNDOF/image/')
config.TRAIN.defocus_map_path = os.path.join(data_offset, 'train/SYNDOF/blur_map/')
config.TRAIN.defocus_map_norm_path = os.path.join(data_offset, 'train/SYNDOF/blur_map_norm/')
config.TRAIN.synthetic_binary_map_path = os.path.join(data_offset, 'train/SYNDOF/blur_map_binary/')
# Real
config.TRAIN.real_img_path = os.path.join(data_offset, 'train/CUHK/image/')
config.TRAIN.real_binary_map_path = os.path.join(data_offset, 'train/CUHK/gt/')
config.TRAIN.real_img_no_label_path = os.path.join(data_offset, 'train/Flickr/')

### TEST DATSET PATH
config.TEST.cuhk_img_path = os.path.join(data_offset, 'test/CUHK/image/')
config.TEST.cuhk_binary_map_path = os.path.join(data_offset, 'test/CUHK/gt/')
config.TEST.SYNDOF_img_path = os.path.join(data_offset, 'test/SYNDOF/image/')
config.TEST.SYNDOF_gt_map_path = os.path.join(data_offset, 'test/SYNDOF/gt/')
config.TEST.RTF0_img_path = os.path.join(data_offset, 'test/RTF/image/0/')
config.TEST.RTF0_gt_map_path = os.path.join(data_offset, 'test/RTF/gt/0/')
config.TEST.RTF1_img_path = os.path.join(data_offset, 'test/RTF/image/1/')
config.TEST.RTF1_gt_map_path = os.path.join(data_offset, 'test/RTF/gt/1/')
config.TEST.RTF1_6_img_path = os.path.join(data_offset, 'test/RTF/image/1.6/')
config.TEST.RTF1_6_gt_map_path = os.path.join(data_offset, 'test/RTF/gt/1.6/')
config.TEST.random_img_path = os.path.join(data_offset, 'test/random/')


## train image size
config.TRAIN.height = 240
config.TRAIN.width = 240

## log & checkpoint & samples
# every global step
config.TRAIN.write_log_every = 100
config.TRAIN.write_ckpt_every = 1
config.TRAIN.write_sample_every = 1000
# every epoch
config.TRAIN.refresh_image_log_every = 20

# save dir
config.root_offset = './logs/'
# config.root_offset = '/Jarvis/logs/junyonglee'
config.TRAIN.root_dir = os.path.join(config.root_offset, 'DMENet_CVPR2019/')

config.TRAIN.max_coc = 15.;

def log_config(path, cfg):
    with open(path + '/config.txt', 'w') as f:
        f.write('================================================\n')
        f.write(json.dumps(cfg, indent=4))
        f.write('\n================================================\n')

def get_eval_path(test_set, cfg):
    if test_set == 'CUHK':
        # return cfg.TEST.cuhk_img_path, cfg.TEST.cuhk_binary_map_path
        return cfg.TEST.cuhk_img_path, None
    elif test_set == 'SYNDOF':
        return cfg.TEST.SYNDOF_img_path, cfg.TEST.SYNDOF_gt_map_path
    elif test_set == 'RTF0':
        return cfg.TEST.RTF0_img_path, cfg.TEST.RTF0_gt_map_path
    elif test_set == 'RTF1':
        return cfg.TEST.RTF1_img_path, cfg.TEST.RTF1_gt_map_path
    elif test_set == 'RTF1_6':
        return cfg.TEST.RTF1_6_img_path, cfg.TEST.RTF1_6_gt_map_path
    elif test_set == 'random':
        return cfg.TEST.random_img_path, None
