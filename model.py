#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

def resNet(t_image, reuse=False, scope = "resNet"):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    #filter_out = [64, 128, 256, 512]
    filter_out = [64, 128, 256, 512]
    num_res_block = [3, 4, 6, 3]
    with tf.variable_scope(scope, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (7, 7), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        #n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')

        for i in range(4):
            for j in range(num_res_block[i]):
                nn = Conv2d(n, filter_out[i], (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s/%s' % (i, j))
                nn = Conv2d(nn, filter_out[i], (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s/%s' % (i, j))
                if i != 0 and j != 0:
                    n = Conv2d(n, filter_out[i], (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='res_n64s1/c2/%s/%s' % (i, j))
                    nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s/%s' % (i, j))
                nn = InputLayer(tf.nn.relu(nn.outputs), name = 'relu/%s/%s' % (i, j))
                n = nn

            #n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool/%s' % i)

        '''
        n = FlattenLayer(n, name='flatten')
        n = DenseLayer(n, n_units=1000, W_init = w_init, b_init = None, act=tf.nn.relu, name='fc1')
        n = DenseLayer(n, n_units=512, W_init = w_init, b_init = None, act=tf.nn.relu, name='fc2')
        n = DenseLayer(n, n_units=1, W_init = w_init, b_init = None, act=tf.nn.tanh, name='fc3')
        '''
        h, w = n.outputs.get_shape().as_list()[1:3]
        n = Conv2d(n, 512, (h, w), act=tf.nn.relu, padding='VALID', W_init=w_init, b_init=b_init, name='fcn1')
        n = Conv2d(n, 512, (1, 1), act=tf.nn.relu, padding='VALID', W_init=w_init, b_init=b_init, name='fcn2')
        output = Conv2d(n, 1, (1, 1), act=tf.nn.tanh, padding='VALID', W_init=w_init, b_init=b_init, name='fcn3')
        n = ReshapeLayer(output, [-1, 1], name='flatten')

        return n, output

def resNet_test(t_image, reuse=False, scope = "resNet_test"):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    #filter_out = [64, 128, 256, 512]
    filter_out = [64, 128, 256, 512]
    num_res_block = [3, 4, 6, 3]
    with tf.variable_scope(scope, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (7, 7), (1, 1), act=tf.nn.relu, padding='VALID', W_init=w_init, name='n64s1/c')
        #n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')

        for i in range(4):
            for j in range(num_res_block[i]):
                nn = Conv2d(n, filter_out[i], (3, 3), (1, 1), act=tf.nn.relu, padding='VALID', W_init=w_init, b_init=b_init, name='n64s1/c1/%s/%s' % (i, j))
                nn = Conv2d(nn, filter_out[i], (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init, b_init=b_init, name='n64s1/c2/%s/%s' % (i, j))
                if i != 0 and j != 0:
                    n = Conv2d(n, filter_out[i], (5, 5), (1, 1), act=tf.nn.relu, padding='VALID', W_init=w_init, b_init=b_init, name='res_n64s1/c2/%s/%s' % (i, j))
                    nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s/%s' % (i, j))
                nn = InputLayer(tf.nn.relu(nn.outputs), name = 'relu/%s/%s' % (i, j))
                n = nn

            #n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool/%s' % i)

        '''
        n = FlattenLayer(n, name='flatten')
        n = DenseLayer(n, n_units=1000, W_init = w_init, b_init = None, act=tf.nn.relu, name='fc1')
        n = DenseLayer(n, n_units=512, W_init = w_init, b_init = None, act=tf.nn.relu, name='fc2')
        n = DenseLayer(n, n_units=1, W_init = w_init, b_init = None, act=tf.nn.tanh, name='fc3')
        '''
        #h, w = n.outputs.get_shape().as_list()[1:3]
        n = Conv2d(n, 1000, (1, 1), act=tf.nn.relu, padding='VALID', W_init=w_init, b_init=b_init, name='fcn1')
        n = Conv2d(n, 512, (1, 1), act=tf.nn.relu, padding='VALID', W_init=w_init, b_init=b_init, name='fcn2')
        output = Conv2d(n, 1, (1, 1), act=tf.nn.tanh, padding='VALID', W_init=w_init, b_init=b_init, name='fcn3')
        n = ReshapeLayer(output, [-1, 1], name='flatten')

        return n, output
