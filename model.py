#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

def resNet(t_image, is_train=False, reuse=False, scope = "resNet"):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    filter_out = [64, 128, 256, 512]
    num_res_block = [3, 4, 6, 3]
    with tf.variable_scope(scope, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (7, 7), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        temp = n

        for i in range(4):
            for j in range(num_res_block[i]):
                nn = Conv2d(n, filter_out[i], (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s/%s' % (i, j))
                nn = Conv2d(nn, filter_out[i], (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s/%s' % (i, j))
                if i != 0 and j != 0:
                    nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s/%s' % (i, j))
                nn = InputLayer(tf.nn.relu(nn.outputs), name = 'relu/%s/%s' % (i, j))
                n = nn

            n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool/%s' % i)

        n = FlattenLayer(n, name='flatten')
        #n = DenseLayer(n, n_units=4096, W_init = w_init, b_init = None, act=None , name='fc6')
        #n = DenseLayer(n, n_units=4096, W_init = w_init, b_init = None, act=None, name='fc7')
        n = DenseLayer(n, n_units=1000, W_init = w_init, b_init = None, act=tf.nn.relu, name='fc1')
        n = DenseLayer(n, n_units=512, W_init = w_init, b_init = None, act=tf.nn.relu, name='fc2')
        n = DenseLayer(n, n_units=1, W_init = w_init, b_init = None, act=tf.nn.sigmoid, name='fc3')

        return n
