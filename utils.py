import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
from config import config, log_config
from skimage import feature
from skimage import color
from scipy.ndimage.filters import gaussian_filter

import scipy
import numpy as np

def get_imgs_RGB_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def get_imgs_GRAY_fn(file_name, path):
    """ Input an image path and name, return an image array """
    image = scipy.misc.imread(path + file_name, mode='P')/255.
    return np.expand_dims(image, axis = 2)

def blur_crop_edge_sub_imgs_fn(data):
    '''
    h = config.TRAIN.height
    w = config.TRAIN.width
    cropped_image = crop(image, wrg=w, hrg=h, is_random=is_random)
    cropped_image = cropped_image / (255. / 2.) - 1
    '''
    h = config.TRAIN.height
    w = config.TRAIN.width
    r = (int)(h/2.) # 35

    image, sigma = data
    #mask = np.ones_like(mask) - mask

    image_h, image_w = np.asarray(image).shape[0:2]

    '''
    # 1. get edge image in "sharp region"
    # 1-1. elementary wise application between image and mask
    #sharp_image = np.multiply(image, mask)
    # 1-2. get edge image
    #edge_image = feature.canny(color.rgb2gray(sharp_image))
    edge_image = np.squeeze(edge)
    # 2. get points in  edge image
    coordinates = np.transpose(np.where(edge_image == 1), (1, 0))

    condition_y = np.logical_and(coordinates[:, 0] >= r, coordinates[:, 0] < image_h - r)
    condition_x = np.logical_and(coordinates[:, 1] >= r, coordinates[:, 1] < image_w - r)
    condition = np.logical_and(condition_x, condition_y)
    condition = np.transpose(np.expand_dims(condition, axis = 0), (1, 0))
    condition = np.concatenate((condition, condition), axis = 1)

    coordinates = np.reshape(np.extract(condition, coordinates), (-1, 2))

    # 3. crop image with given random edge point at center
    random_index = np.random.randint(0, coordinates.shape[0])
    center_y, center_x = coordinates[random_index, 0:2]

    cropped_image = image[center_y - r : center_y + r + 1, center_x - r : center_x + r + 1]

    # 4. Gaussian Blur
    image_blur = gaussian_filter(cropped_image, (sigma[0], sigma[0], 0))
    image_blur = image_blur + (np.mean(cropped_image) - np.mean(image_blur))
    image_blur[image_blur > 255.] = 255.

    ###
    cropped_edge = edge[center_y - r : center_y + r + 1, center_x - r : center_x + r + 1]
    cropped_edge = np.concatenate((cropped_edge, cropped_edge, cropped_edge), axis = 2)
    return image_blur, cropped_image, cropped_edge
    ###
    '''
    cropped_image = tl.prepro.crop(image, wrg=h, hrg=w, is_random=True)
    image_blur = gaussian_filter(cropped_image, (sigma[0], sigma[0], 0))
    image_blur = image_blur + (np.mean(cropped_image) - np.mean(image_blur))
    image_blur[image_blur > 255.] = 255.

    return image_blur / (255. / 2.) - 1.

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[56, 56], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass 

def activation_map(gray):
    red = np.expand_dims(np.ones_like(gray), axis = 2)
    green = np.expand_dims(np.ones_like(gray), axis = 2)
    blue = np.expand_dims(np.ones_like(gray), axis = 2)
    gray = np.expand_dims(gray, axis = 2)

    red[(gray == 0.)] = 0.
    green[(gray == 0.)] = 0.
    blue[(gray == 0.)] = 0.

    red[(gray <= 1./3)&(gray > 0.)] = 1.
    green[(gray <= 1./3)&(gray > 0.)] = 3. * gray[(gray <= 1./3.)&(gray > 0.)]
    blue[(gray <= 1./3)&(gray > 0.)] = 0.

    red[(gray > 1./3)&(gray <= 2./3)] = -3. * (gray[(gray > 1./3)&(gray <= 2./3)] - 1./3.) + 1.
    green[(gray > 1./3)&(gray <= 2./3)] = 1.
    blue[(gray > 1./3)&(gray <= 2./3)] = 0.

    red[(gray > 2./3)] = 0.
    green[(gray > 2./3)] = -3. * (gray[(gray > 2./3)] - 2./3.) + 1.
    blue[(gray > 2./3)] = 3. * (gray[(gray > 2./3)] - 2./3.)

    return np.concatenate((red, green, blue), axis = 2) * 255.

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, mean_metric = True, cs_map=False, size=9, sigma=0.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    C3 = C2/2.
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2

    l = (2 * mu1 * mu2 + C1)/(mu1_sq + mu2_sq + C1)
    c = (2 * tf.sqrt(sigma1_sq) * tf.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
    #s = (sigma12 + C3)/(tf.sqrt(sigma1_sq) * tf.sqrt(sigma2_sq) + C3)
    s = (sigma12**2 + C3)/(sigma1_sq * sigma2_sq + C3)
    if cs_map:
        value = ( l ** 0, (c ** 0) * s)
    else:
        '''
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
        '''
        value = s

    if mean_metric:
        value = tf.reduce_mean(value)

    return value

def tf_ms_ssim(img1, img2, batch_size, mean_metric=True, level=5):
    #weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weight = [1., 1., 1., 1., 1.]

    ml = []
    mcs = []
    image1 = img1
    image2 = img2
    for l in range(level):
        l_map, cs_map = tf_ssim(image1, image2, cs_map=True, mean_metric=False)

        l_map_mean = tf.reduce_mean(tf.reshape(l_map, [batch_size, -1]), axis=1)
        cs_map_mean = tf.reduce_mean(tf.reshape(cs_map, [batch_size, -1]), axis=1)

        #l_map_mean = tf.Print(l_map_mean, [l_map_mean], message = 'l_map_{}'.format(l), summarize = 10)
        #cs_map_mean = tf.Print(cs_map_mean, [cs_map_mean], message = 'cs_map_{}'.format(l), summarize = 10)

        ml.append(l_map_mean)
        mcs.append(cs_map_mean)

        image1 = tf.nn.avg_pool(image1, [1,2,2,1], [1,2,2,1], padding='SAME')
        image2 = tf.nn.avg_pool(image2, [1,2,2,1], [1,2,2,1], padding='SAME')

    # list to tensor of dim D+1
    ml = tf.stack(ml, axis=0)
    mcs = tf.stack(mcs, axis=0)

    mat = np.copy(weight)
    for i in np.arange(batch_size - 1):
        mat = np.concatenate((mat, weight))
    mat = np.reshape(mat, [batch_size, level])
    mat = np.transpose(mat)

    weight_mat = tf.constant(mat, shape=[level, batch_size], dtype=tf.float32)
    #weight_mat = tf.Print(weight_mat, [weight_mat], message = 'w_mat', summarize = 50)

    value = (ml[level-1]**tf.constant(weight[level-1], shape=[batch_size])) * tf.reduce_prod(mcs[0:level]**weight_mat)

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def refine_image(img):
    image = img
    while True:
        shape = image.shape
        if shape[0] % 2 is not 0:
            image = image[:-1, :, :]
            continue
        elif (shape[0] / 2) % 2 is not 0:
            image = image[:-1, :, :]
            continue
        elif (shape[0] / 2 / 2) % 2 is not 0:
            image = image[:-1, :, :]
            continue
        elif (shape[0] / 2 / 2 / 2) % 2 is not 0:
            image = image[:-1, :, :]
            continue
        else:
            break;
    
    while True:
        shape = image.shape
        if shape[1] % 2 is not 0:
            image = image[:, :-1, :]
            continue
        elif (shape[1] / 2) % 2 is not 0:
            image = image[:, :-1, :]
            continue
        elif (shape[1] / 2 / 2) % 2 is not 0:
            image = image[:, :-1, :]
            continue
        elif (shape[1] / 2 / 2/ 2) % 2 is not 0:
            image = image[:, :-1, :]
            continue
        else:
            break;
    
    return image
