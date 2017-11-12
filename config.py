from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()

## Adam
config.TRAIN.batch_size = 80
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 1000
# config.TRAIN.lr_decay_init = 0.1
# config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 3000
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 200

## train set location
config.TRAIN.blur_img_path = '/data1/BlurDetection/train/image/'
config.TRAIN.mask_img_path = '/data1/BlurDetection/train/gt/'
config.TEST.blur_img_path = '/data1/BlurDetection/test/image/'
config.TEST.mask_img_path = '/data1/BlurDetection/test/gt/'


## train image size
config.TRAIN.height = 71
config.TRAIN.width = 71

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
