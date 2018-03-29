from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()

## Adam
config.TRAIN.batch_size = 5
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

# learning rate
config.TRAIN.n_epoch = 100000
config.TRAIN.lr_decay = 0.8
config.TRAIN.decay_every = 5000

## train set location
config.TRAIN.synthetic_img_path = '/data1/blur_sharp_only/image_dof/'
config.TRAIN.defocus_map_path = '/data1/blur_sharp_only/defocus_map/'

config.TRAIN.real_img_path = '/data1/BlurDetection/train/image/'
config.TRAIN.binary_map_path = '/data1/BlurDetection/train/gt/'

config.TEST.blur_img_path = "/data1/BlurDetection/test/image/"
config.TEST.binary_map_path = "/data1/BlurDetection/test/gt/"

## train image size
config.TRAIN.height = 512
config.TRAIN.width = 512

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
