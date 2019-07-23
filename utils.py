import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
from config import config, log_config
from skimage import feature
from skimage import color
from scipy.ndimage.filters import gaussian_filter

import scipy
import numpy as np
import cv2
import math
import random

import os
import fnmatch

def read_all_imgs(file_name_list, path = '', mode = 'RGB'):
    imgs = []
    for idx in range(0, len(file_name_list)):
        imgs.append(get_images(file_name_list[idx], path, mode))
        
    return imgs 

def get_images(file_name, path, mode):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    if mode is 'RGB':
        image = (scipy.misc.imread(path + file_name, mode='RGB')/255.).astype(np.float32)
    elif mode is 'GRAY':
        image = (scipy.misc.imread(path + file_name, mode='P')/255.).astype(np.float32)
        image = np.expand_dims(image, axis = 2)
    elif mode is 'DEPTH':
        image = (np.float32(cv2.imread(path + file_name, cv2.IMREAD_UNCHANGED))/10.)[:, :, 1]
        image = image / 15.
        image = np.expand_dims(image, axis = 2)

    return image

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass 

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

def refine_image(img):
    h, w = img.shape[:2]
    
    return img[0 : h - h % 16, 0 : w - w % 16]

def random_crop(images, resize_shape, is_gaussian_noise = False):
    images_list = None
    h, w = resize_shape[:2]
    max_size_limit = 800
    
    for i in np.arange(len(images)):
        image = np.copy(images[i])
        shape = np.array(image.shape[:2])
        
        if shape.min() <= h:
            ratio = resize_shape[shape.argmin()]/float(shape.min())
            resize_w = int(math.floor(shape[1] * ratio)) + 1
            resize_h = int(math.floor(shape[0] * ratio)) + 1
            image = cv2.resize(image, (resize_w, resize_h))

        if shape.min() > max_size_limit:
            ratio = max_size_limit/float(shape.min())
            resize_w = int(math.floor(shape[1] * ratio)) + 1
            resize_h = int(math.floor(shape[0] * ratio)) + 1
            image = cv2.resize(image, (resize_w, resize_h))

        if is_gaussian_noise:
            image = add_gaussian_noise(image)

        cropped_image = tl.prepro.crop(image, wrg=w, hrg=h, is_random=True)
        augmented_image = _random_flip(cropped_image)
        angles = np.array([1, 2, 3, 4])
        angle = np.random.choice(angles)
        augmented_image = _random_rotation(augmented_image, angle)
        image = np.expand_dims(augmented_image, axis=0)
        
        images_list = np.copy(image) if i == 0 else np.concatenate((images_list, image), axis = 0)

    return images_list

def crop_pair_with_different_shape_images(images, labels, resize_shape, is_gaussian_noise = False):
    images_list = None
    labels_list = None
    h, w = resize_shape[:2]
    max_size_limit = 800
    
    for i in np.arange(len(images)):
        image = np.copy(images[i])
        label = np.copy(labels[i])
        shape = np.array(image.shape[:2])
        
        if shape.min() <= h:
            ratio = resize_shape[shape.argmin()]/float(shape.min())
            resize_w = int(math.floor(shape[1] * ratio)) + 1
            resize_h = int(math.floor(shape[0] * ratio)) + 1
            image = cv2.resize(image, (resize_w, resize_h))
            label = np.expand_dims(cv2.resize(label[:, :, 0], (resize_w, resize_h)), axis = 2)

        if shape.min() > max_size_limit:
            ratio = max_size_limit/float(shape.min())
            resize_w = int(math.floor(shape[1] * ratio)) + 1
            resize_h = int(math.floor(shape[0] * ratio)) + 1
            image = cv2.resize(image, (resize_w, resize_h))
            label = np.expand_dims(cv2.resize(label[:, :, 0], (resize_w, resize_h)), axis = 2)

        if is_gaussian_noise:
            image = add_gaussian_noise(image)

        concatenated_images = np.concatenate((image, label), axis = 2)
        cropped_images = tl.prepro.crop(concatenated_images, wrg=w, hrg=h, is_random=True)
        augmented_images = _random_flip(cropped_images)
        angles = np.array([1, 2, 3, 4])
        angle = np.random.choice(angles)
        augmented_images = _random_rotation(augmented_images, angle)

        image = np.expand_dims(augmented_images[:, :, 0:3], axis=0)
        label = np.expand_dims(np.expand_dims(augmented_images[:, :, 3], axis=3), axis=0)
        
        images_list = np.copy(image) if i == 0 else np.concatenate((images_list, image), axis = 0)
        labels_list = np.copy(label) if i == 0 else np.concatenate((labels_list, label), axis = 0)

    return images_list, labels_list

def add_gaussian_noise(image):
    image = image.astype(np.float32)
    shape = image.shape[:2]

    mean = 0
    var = random.uniform(0,0.1)
    sigma = var ** 0.5
    gamma = 0.25
    alpha = 0.75
    beta = 1 - alpha

    gaussian = np.random.normal(loc=mean, scale = sigma, size = (shape[0], shape[1], 1)).astype(np.float32)
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
    #gaussian_img = image * 0.75 + 0.25 * gaussian + 0.25
    gaussian_img = cv2.addWeighted(image, alpha, beta * gaussian, beta, gamma)

    return gaussian_img

    # noise_sigma = 0.01
    # h = image.shape[0]
    # w = image.shape[1]
    # noise = np.random.randn(h, w) * noise_sigma

    # noisy_image = np.zeros(image.shape, np.float64)
    # if len(image.shape) == 2:
    #     noisy_image = image + noise
    # else:
    #     noisy_image[:,:,0] = image[:,:,0] + noise
    #     noisy_image[:,:,1] = image[:,:,1] + noise
    #     noisy_image[:,:,2] = image[:,:,2] + noise

    # """
    # print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    # print('type = ', type(noisy_image[0][0][0]))
    # """

    # return noisy_image

def _random_flip(images):
    flipped_images = tl.prepro.flip_axis(images, axis=0, is_random=True)

    return flipped_images

def _random_rotation(images, angle):
    if angle != 4:
        rotated_images = np.rot90(images, angle)
    else:
        rotated_images = images

    return rotated_images

def _get_file_path(path, regex):
    file_path = []
    for root, dirnames, filenames in os.walk(path):
        for i in np.arange(len(regex)):
            for filename in fnmatch.filter(filenames, regex[i]):
                file_path.append(os.path.join(root, filename))
    
    return file_path

def remove_file_end_with(path, regex):
    file_paths = _get_file_path(path, [regex])

    for i in np.arange(len(file_paths)):
        os.remove(file_paths[i])

def save_images(images, size, image_path='_temp.png'):
    if len(images.shape) == 3:  # Greyscale [batch, h, w] --> [batch, h, w, 1]
        images = images[:, :, :, np.newaxis]

    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    def imsave(images, size, path):
        return scipy.misc.toimage(merge(images, size), cmin = 0., cmax = 1.).save(path)

    assert len(images) <= size[0] * size[1], "number of images should be equal or less than size[0] * size[1] {}".format(len(images))

    return imsave(images, size, image_path)

def fix_image_tf(image, norm_value):
    return tf.cast(image / norm_value * 255., tf.uint8)

def norm_image_tf(image):
    image = image - tf.reduce_min(image, axis = [1, 2, 3], keepdims=True)
    image = image / tf.reduce_max(image, axis = [1, 2, 3], keepdims=True)
    return tf.cast(image * 255., tf.uint8)

def norm_image(image, axis = (1, 2, 3)):
    image = image - np.amin(image, axis = axis, keepdims=True)
    image = image / np.amax(image, axis = axis, keepdims=True)
    return image
        
def get_disc_accuracy(logits, labels):
    acc = 0.
    for i in np.arange(len(logits)):
        tp = 0
        logits[i] = np.round(np.squeeze(logits[i])).astype(int)
        temp = logits[i]
        tp = tp + len(temp[np.where(temp == labels[i])])
        acc = acc + (tp / float(len(logits[i])))
    return acc / float(len(labels))

