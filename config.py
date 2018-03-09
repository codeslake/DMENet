from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()

## Adam
config.TRAIN.batch_size = 10
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 1000
# config.TRAIN.lr_decay_init = 0.1
# config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 10000
config.TRAIN.lr_decay = 0.8
config.TRAIN.decay_every = 100

## train set location
config.TRAIN.synthetic_img_path = '/data1/stereo/image/'
config.TRAIN.defocus_map_path = '/data1/stereo/defocus_map/'

config.TRAIN.real_img_path = '/data1/BlurDetection/train/image/'
config.TRAIN.binary_map_path = '/data1/BlurDetection/train/gt/'

config.TRAIN.img_path_sample = '/data1/stereo_sample/image/'
config.TRAIN.defocus_map_path_sample = '/data1/stereo_sample/defocus_map/'

config.TEST.blur_img_path = "/data1/SharpImages/test/"
#config.TEST.blur_img_path = '/data1/JunyongLee/monkaa_clean_training_sample/image/'


## train image size
config.TRAIN.height = 256
config.TRAIN.width = 256


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
