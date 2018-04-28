import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

def UNet_down(image_in, is_train=False, reuse=False, scope = 'unet_down'):
    w_init1 = tf.random_normal_initializer(stddev=0.02)
    w_init2 = tf.random_normal_initializer(stddev=0.01)
    w_init3 = tf.random_normal_initializer(stddev=0.005)
    w_init4 = tf.random_normal_initializer(stddev=0.002)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        n = InputLayer(image_in, name='image_in')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d0/dr1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init1, name='d0/c1')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d0/b1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d0/dr2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init1, name='d0/c2')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d0/b2')
        d0 = n

        #n = MaxPool2d(n, (2, 2), (2, 2), padding='SAME', name='d1/pool1')
        n = Conv2d(n, 64, (2, 2), (2, 2), act=lrelu, padding='SAME', W_init=w_init2, name='d1/pool1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d1/dr1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init2, name='d1/c1')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d1/b1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d1/dr2')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init2, name='d1/c2')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d1/b2')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d1/dr3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init2, name='d1/c3')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d1/b3')
        d1 = n

        #n = MaxPool2d(n, (2, 2), (2, 2), padding='SAME', name='d2/pool1')
        n = Conv2d(n, 128, (2, 2), (2, 2), act=lrelu, padding='SAME', W_init=w_init2, name='d2/pool1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d2/dr1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='d2/c1')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d2/b1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d2/dr2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='d2/c2')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d2/b2')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d2/dr3')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='d2/c3')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d2/b3')
        d2 = n

        #n = MaxPool2d(n, (2, 2), (2, 2), padding='SAME', name='d3/pool1')
        n = Conv2d(n, 256, (2, 2), (2, 2), act=lrelu, padding='SAME', W_init=w_init2, name='d3/pool1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d3/dr1')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init4, name='d3/c1')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d3/b1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d3/dr2')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init4, name='d3/c2')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d3/b2')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d3/dr3')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init4, name='d3/c3')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d3/b3')
        d3 = n

        #n = MaxPool2d(n, (2, 2), (2, 2), padding='SAME', name='d4/pool1')
        n = Conv2d(n, 512, (2, 2), (2, 2), act=lrelu, padding='SAME', W_init=w_init2, name='d4/pool1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d4/dr1')
        n = Conv2d(n, 1024, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init4, name='d4/c1')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d4/b1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d4/dr2')
        n = Conv2d(n, 1024, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init4, name='d4/c2')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d4/b2')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='d4/dr3')
        n = Conv2d(n, 1024, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init4, name='d4/c3')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d4/b3')
        d4 = n

        return [d0.outputs, d1.outputs, d2.outputs, d3.outputs, d4.outputs]

def UNet_up(feats, is_train=False, reuse=False, scope = 'unet_up'):
    w_init1 = tf.random_normal_initializer(stddev=0.02)
    w_init2 = tf.random_normal_initializer(stddev=0.01)
    w_init3 = tf.random_normal_initializer(stddev=0.005)
    w_init4 = tf.random_normal_initializer(stddev=0.002)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        d0 = InputLayer(feats[0], name='d0')
        d1 = InputLayer(feats[1], name='d1')
        d2 = InputLayer(feats[2], name='d2')
        d3 = InputLayer(feats[3], name='d3')
        d4 = InputLayer(feats[4], name='d4')
        
        n = UpSampling2dLayer(d4, (2, 2), is_scale = True, method = 1, align_corners=False, name='u3/u')
        n = ConcatLayer([n, d3], concat_dim = 3, name='u3/concat')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u3/dr1')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u3/c1')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u3/b1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u3/dr2')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u3/c2')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u3/b2')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u3/dr3')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u3/c3')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u3/b3')
        u3 = n

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=False, name='u2/u')
        n = ConcatLayer([n, d2], concat_dim = 3, name='u2/concat')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u2/dr1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u2/c1')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u2/b1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u2/dr2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u2/c2')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u2/b2')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u2/dr3')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u2/c3')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u2/b3')
        u2 = n

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=False, name='u1/u')
        n = ConcatLayer([n, d1], concat_dim = 3, name='u1/concat')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u1/dr1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u1/c1')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u1/dr2')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u1/c2')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u1/b2')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u1/dr3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u1/c3')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u1/b3')
        u1 = n

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=False, name='u0/u')
        n = ConcatLayer([n, d0], concat_dim = 3, name='u0/concat')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u1/dr1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u0/c1')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u0/b1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u1/dr2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u0/c2')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u0/b2')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='u1/dr3')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init3, name='u0/c3')
        #n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u0/b3')
        u0 = n

        n_1c = Conv2d(n, 1, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='uf/1c')
        n_3c = Conv2d(n, 3, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='uf/3c')

        return n.outputs, tf.nn.sigmoid(n_1c.outputs), [u0, u1, u2, u3, d4]

def SRGAN_d(feats, is_train=True, reuse=False, scope = 'Discriminator'):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse):
        d0 = InputLayer(feats[0], name='d0')
        d1 = InputLayer(feats[1], name='d1')
        d2 = InputLayer(feats[2], name='d2')
        d3 = InputLayer(feats[3], name='d3')
        d4 = InputLayer(feats[4], name='d4')
        
        net_h0 = Conv2d(d0, 128, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='h0/c1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='h0/dr1')
        net_h0 = Conv2d(net_h0, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h0/c2')
        #net_h0 = BatchNormLayer(net_h0, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h0/bn1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='h0/dr2')
        net_h0 = Conv2d(net_h0, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h0/c3')
        #net_h0 = BatchNormLayer(net_h0, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h0/b2')

        net_h1 = ElementwiseLayer([net_h0, d1], tf.add, name='add1')
        net_h1 = Conv2d(net_h1, 256, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='h1/dr1')
        net_h1 = Conv2d(net_h1, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c2')
        #net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h1/bn1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='h1/dr2')
        net_h1 = Conv2d(net_h1, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c3')
        #net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h1/bn2')

        net_h2 = ElementwiseLayer([net_h1, d2], tf.add, name='add2')
        net_h2 = Conv2d(net_h2, 512, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='h2/dr1')
        net_h2 = Conv2d(net_h2, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c2')
        #net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h2/bn1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='h2/dr2')
        net_h2 = Conv2d(net_h2, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c3')
        #net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h2/bn2')

        net_h3 = ElementwiseLayer([net_h2, d3], tf.add, name='add3')
        net_h3 = Conv2d(net_h3, 1024, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='h3/dr1')
        net_h3 = Conv2d(net_h3, 1024, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c2')
        #net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h3/bn1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='h3/dr2')
        net_h3 = Conv2d(net_h3, 1024, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c3')
        #net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h3/bn2')

        net_h4 = ElementwiseLayer([net_h3, d4], tf.add, name='add4')
        net_h4 = Conv2d(net_h4, 1024, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='h4/dr1')
        net_h4 = Conv2d(net_h4, 1024, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c2')
        #net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h4/bn1')
        n = DropoutLayer(n, keep=0.9, is_fix=True, is_train=is_train, name='h4/dr2')
        net_h4 = Conv2d(net_h4, 1024, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c3')
        #net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h4/bn2')

        net_hf = FlattenLayer(net_h4, name='hf/flatten')
        n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='hf/dr1')
        net_hf = DenseLayer(net_hf, n_units=1, act=tf.identity, W_init = w_init, name='hf/dense')
        logits = net_hf.outputs

    return logits

def Binary_Net(input_defocus, is_train=False, reuse=False, scope = 'Binary_Net'):
    w_init1 = tf.random_normal_initializer(stddev=0.02)
    w_init2 = tf.random_normal_initializer(stddev=0.01)
    w_init3 = tf.random_normal_initializer(stddev=0.005)
    w_init4 = tf.random_normal_initializer(stddev=0.002)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        n = InputLayer(input_defocus, name='input_defocus')
        
        n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='l1/c1')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='l1/b1')
        n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='l1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='l1/b2')
        n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='l1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='l1/b3')

        n = Conv2d(n, 32, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='l2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='l2/b1')
        n = Conv2d(n, 32, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='l2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='l2/b2')
        n = Conv2d(n, 32, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='l2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='l2/b3')

        n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='l3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='l3/b1')
        n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='l3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='l3/b2')
        n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='l3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='l3/b3')

        n = Conv2d(n, 1, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='l4/c1')
        logits = n.outputs

        return n.outputs, tf.nn.sigmoid(logits)
