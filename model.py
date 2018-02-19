#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

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
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

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

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))

def upconv_2D(tensor, n_filter, name):
    """Up Convolution `tensor` by 2 times
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
        name="upsample_{}".format(name))

def unet(X, training, reuse=False, scope=""):
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
        net = X / 127.5 - 1
        conv1, pool1 = conv_conv_pool(net, [8, 8], training, name=1)
        conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, name=2)
        conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, name=3)
        conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, name=4)
        conv5 = conv_conv_pool(
            pool4, [128, 128], training, name=5, pool=False)

        up6 = upconv_concat(conv5, conv4, 64, name=6)
        conv6 = conv_conv_pool(up6, [64, 64], training, name=6, pool=False)

        up7 = upconv_concat(conv6, conv3, 32, name=7)
        conv7 = conv_conv_pool(up7, [32, 32], training, name=7, pool=False)

        up8 = upconv_concat(conv7, conv2, 16, name=8)
        conv8 = conv_conv_pool(up8, [16, 16], training, name=8, pool=False)

        up9 = upconv_concat(conv8, conv1, 8, name=9)
        conv9 = conv_conv_pool(up9, [8, 8], training, name=9, pool=False)

        return tf.layers.conv2d(
            conv9,
            1, (1, 1),
            name='final',
            activation=tf.nn.sigmoid,
            padding='same')