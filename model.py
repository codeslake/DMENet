#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

w_init = tf.random_normal_initializer(stddev=0.02)
b_init = None
g_init = tf.random_normal_initializer(1., 0.02)

def conv_conv_pool(input_,
                   n_filters,
                   training,
                   name,
                   pool=True,
                   activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = Conv2d(net, F, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv_{}'.format(i + 1))
            net = BatchNormLayer(net, act=tf.nn.relu, is_train=training, gamma_init=g_init, name='bn_{}'.format(i + 1))

        if pool is False:
            return net

        pool = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='MaxPool2d')
        return net, pool


def upconv_concat(inputA, input_B, n_filter, name):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D(inputA, n_filter, name)

    return ElementwiseLayer([up_conv, input_B], tf.add, name='concat_{}'.format(name))

def upconv_2D(net, n_filter, name):
    """Up Convolution `tensor` by 2 times
    Args:
        net (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """
    h, w = net.outputs.get_shape().as_list()[1:3]
    return DeConv2d(net, n_filter, (3, 3), (h*2, w*2), (2, 2), act=None, padding='SAME', W_init=w_init, name='upsample_{}'.format(name))

def unet(image_in, training, reuse=False, scope=""):
    """Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor
    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    with tf.variable_scope(scope, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        
        image_in = InputLayer(image_in, name='input')
        conv1, pool1 = conv_conv_pool(image_in, [64, 64], training, name=1)
        conv2, pool2 = conv_conv_pool(pool1, [128, 128], training, name=2)
        conv3, pool3 = conv_conv_pool(pool2, [256, 256, 256, 256], training, name=3)
        conv4, pool4 = conv_conv_pool(pool3, [512, 512, 512, 512], training, name=4)

        conv5 = conv_conv_pool(pool4, [1024, 1024], training, name=5, pool=False)

        up6 = upconv_concat(conv5, conv4, 512, name=6)
        conv6 = conv_conv_pool(up6, [512, 512, 512, 512], training, name=6, pool=False)

        up7 = upconv_concat(conv6, conv3, 256, name=7)
        conv7 = conv_conv_pool(up7, [256, 256, 256, 256], training, name=7, pool=False)

        up8 = upconv_concat(conv7, conv2, 128, name=8)
        conv8 = conv_conv_pool(up8, [128, 128], training, name=8, pool=False)

        up9 = upconv_concat(conv8, conv1, 64, name=9)
        conv9 = conv_conv_pool(up9, [64, 64], training, name=9, pool=False)

        final = Conv2d(conv9, 1, (1, 1), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init, b_init=b_init, name='final')

        return final.outputs
