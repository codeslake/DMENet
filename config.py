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

### TRAIN DATSET PATH
#offset = '/Mango/Users/junyonglee/hub/datasets/'
#offset = '/Mango/Users/junyonglee/hub/datasets/DMENet/train/'
offset = '/data1/junyonglee/defocus_map_estimation/DMENet/train/'
#offset = '/root/DMENet/train/'
config.TRAIN.synthetic_img_path = offset + '15_gaussian_many/image/'
config.TRAIN.defocus_map_path = offset + '15_gaussian_many/blur_map/'
config.TRAIN.defocus_map_norm_path = offset + '15_gaussian_many/blur_map_norm/'
config.TRAIN.synthetic_binary_map_path = offset + '15_gaussian_many/blur_map_binary/'
# Real
config.TRAIN.real_img_path = offset + 'CUHK/image/'
config.TRAIN.real_binary_map_path = offset + 'CUHK/gt/'
config.TRAIN.real_img_no_label_path = offset + 'Flickr/'

### TEST DATSET PATH
## CUHK
# offset = '/Mango/Users/junyonglee/hub/datasets/BlurDetection/test/'
# offset = '/Jarvis/workspace/junyonglee/defocus_map_estimation/datasets/test/CUHK/'
# offset = '/Mango/Users/junyonglee/hub/datasets/DMENet/test/CUHK/'
# offset = '/root/DMENet/test/CUHK/'
# config.TEST.real_img_path = offset + 'image/'
# config.TEST.real_binary_map_path = offset + 'gt/'

# current
# offset = '/data1/junyonglee/defocus_map_estimation/DMENet/test/CUHK/'
# config.TEST.real_img_path = offset + 'image/'
# config.TEST.real_binary_map_path = offset + 'gt/'

# SYNDOF
offset = '/Mango/Users/junyonglee/hub/datasets/test/syndof/'
config.TEST.real_img_path = offset
config.TEST.real_binary_map_path = offset

## RDF
# offset = '/Jarvis/workspace/junyonglee/defocus_map_estimation/datasets/test/RTF'
# config.TEST.real_img_path = offset + 'test/RTF/1/'
# config.TEST.real_binary_map_path = offset + 'test/RTF/1/'
# config.TEST.real_img_path = offset + 'test/syndof/'
# config.TEST.real_binary_map_path = offset + 'test/syndof/'

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
config.TRAIN.root_dir = '/Jarvis/logs/junyonglee/DMENet/'

config.TRAIN.max_coc = 15.;

def log_config(path, cfg):
    with open(path + '/config.txt', 'w') as f:
        f.write('================================================\n')
        f.write(json.dumps(cfg, indent=4))
        f.write('\n================================================\n')
