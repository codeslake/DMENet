from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()

## Adam
config.TRAIN.batch_size = 3
config.TRAIN.batch_size_init = 8
#config.TRAIN.lr_init = 1e-5
config.TRAIN.lr_init = 1e-4
config.TRAIN.lr_init_init = 1e-4
#config.TRAIN.beta1 = 0.5
config.TRAIN.beta1 = 0.9

# learning rate
config.TRAIN.n_epoch = 10000
config.TRAIN.n_epoch_init = 10
config.TRAIN.lr_decay = 0.8
config.TRAIN.decay_every = 20

## adversarial loss coefficient
config.TRAIN.lambda_adv = 1e-3

## discriminator lr coefficient
#config.TRAIN.lambda_lr_d = 1e-2
config.TRAIN.lambda_lr_d = 1

## binary loss coefficient
#config.TRAIN.lambda_binary = 1e-2
config.TRAIN.lambda_binary = 2e-2

## perceptual loss coefficient
#config.TRAIN.lambda_perceptual = 1e-6
config.TRAIN.lambda_perceptual = 1e-4

## train set location
# config.TRAIN.synthetic_img_path = '/data1/junyonglee/blur_sharp_only/image_dof/'
# config.TRAIN.defocus_map_path = '/data1/junyonglee/blur_sharp_only/defocus_map/'
# config.TRAIN.synthetic_binary_map_path = '/data1/junyonglee/blur_sharp_only/binary_map/'

#offset = '/Mango/Users/JunyongLee/datasets/30_new_better/'
offset = '/Mango/Users/JunyongLee/hub/datasets/'
config.TRAIN.synthetic_img_path = offset + 'SYNDOF/29_gaussian/image/'
config.TRAIN.defocus_map_path = offset + 'SYNDOF/29_gaussian/blur_map/'
config.TRAIN.defocus_map_norm_path = offset + 'SYNDOF/29_gaussian/blur_map_norm/'
config.TRAIN.synthetic_binary_map_path = offset + 'SYNDOF/29_gaussian/blur_map_binary/'

config.TRAIN.real_img_no_label_path = offset + 'dof_real_resized/'

## test set location
offset = '/Mango/Users/JunyongLee/hub/datasets/'
config.TRAIN.real_img_path = offset + 'BlurDetection/train/image/'
config.TRAIN.real_binary_map_path = offset + 'BlurDetection/train/gt/'

config.TEST.real_img_path = offset + 'BlurDetection/test/image/'
config.TEST.real_binary_map_path = offset + 'BlurDetection/test/gt/'
# config.TEST.real_img_path = offset + 'test/RTF/1.6/'
# config.TEST.real_binary_map_path = offset + 'test/RTF/1.6/'
# config.TEST.real_img_path = offset + 'test/syndof/'
# config.TEST.real_binary_map_path = offset + 'test/syndof/'

## train image size
config.TRAIN.height = 240
config.TRAIN.width = 240
# config.TRAIN.height = 299
# config.TRAIN.width = 299

## log & checkpoint & samples
# every global step
config.TRAIN.write_log_every = 5
config.TRAIN.write_ckpt_every = 1000
config.TRAIN.write_sample_every = 1000
# every epoch
config.TRAIN.refresh_image_log_every = 20

# save dir
#config.TRAIN.root_dir = '/data2/junyonglee/sharpness_assessment/'
config.TRAIN.root_dir = '/Jarvis/logs/junyonglee/sharpness_assessment/'

config.TRAIN.max_coc = 15.;

def log_config(path, cfg):
    with open(path + '/config.txt', 'w') as f:
        f.write('================================================\n')
        f.write(json.dumps(cfg, indent=4))
        f.write('\n================================================\n')
