import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

def Vgg19_simple_api(rgb, reuse, scope):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope(scope, reuse=reuse) as vs:
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else: # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)

        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv1_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_2')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv1_2')
        d0 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv2_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_2')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv2_2')
        d1 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv3_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv3_2')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv3_3')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_4')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv3_4')
        d2 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv4_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv4_2')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv4_3')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_4')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv4_4')
        d3 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')                           
        """ conv5 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv5_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv5_2')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv5_3')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_4')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='VALID', name='conv5_4')
        d4 = network
        
        return network, [d0.outputs, d1.outputs, d2.outputs, d3.outputs, d4.outputs]

def UNet_up(feats, is_train=False, reuse=False, scope = 'unet_up'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        d0 = InputLayer(feats[0], name='d0')
        d1 = InputLayer(feats[1], name='d1')
        d2 = InputLayer(feats[2], name='d2')
        d3 = InputLayer(feats[3], name='d3')
        if feats[4] != None:
            d4 = InputLayer(feats[4], name='d4')
            n = UpSampling2dLayer(d4, (2, 2), is_scale = True, method = 1, align_corners=True, name='u3/u')
            n = ConcatLayer([n, d3], concat_dim = 3, name='u3/concat')
        else:
            n = d3
            
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad3')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b3')
        u3 = n

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u2/u')
        n = ConcatLayer([n, d2], concat_dim = 3, name='u2/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b3')
        u2 = n

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u1/u')
        n = ConcatLayer([n, d1], concat_dim = 3, name='u1/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad3')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b3')
        u1 = n

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u0/u')
        n = ConcatLayer([n, d0], concat_dim = 3, name='u0/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad1')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad_relu')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad3')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b3')
        u0 = n

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad3')
        n = Conv2d(n, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='uf/1c')

        return n.outputs, tf.nn.sigmoid(n.outputs), [u0.outputs, u1.outputs, u2.outputs, u3.outputs, None]

def Binary_Net(input_defocus, is_train=False, reuse=False, scope = 'Binary_Net'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    b_init = None # tf.constant_initializer(value=0.0)
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        n = InputLayer(input_defocus, name='input_defocus')
        
        n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l1/b1')
        #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='l1/dr1')
        n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l1/b2')
        #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='l1/dr2')
        n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l1/b3')
        #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='l1/dr3')

        n = Conv2d(n, 32, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l2/b1')
        #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='l2/dr1')
        n = Conv2d(n, 32, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l2/b2')
        #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='l2/dr2')
        n = Conv2d(n, 32, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l2/b3')
        #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='l2/dr3')

        n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l3/b1')
        #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='l3/dr1')
        n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l3/b2')
        #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='l3/dr2')
        n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l3/b3')
        #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='l3/dr3')

        #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='l4/dr1')
        n = Conv2d(n, 1, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='l4/c1')
        logits = n.outputs

        return logits, tf.nn.sigmoid(n.outputs)

def feature_discriminator(feats, is_train=True, reuse=False, scope = 'feature_discriminator'):
    w_init = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = None
    
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse):
        d0 = InputLayer(feats[0], name='d0')
        d1 = InputLayer(feats[1], name='d1')
        d2 = InputLayer(feats[2], name='d2')
        d3 = InputLayer(feats[3], name='d3')
        d4 = InputLayer(feats[4], name='d4')
        
        n = Conv2d(d0, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h0/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h0/b1')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h0/dr1')
        n = Conv2d(d0, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h0/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h0/b2')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h0/dr2')
        n = Conv2d(n, 128, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h0/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h0/b3')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h0/dr3')

        n = ElementwiseLayer([n, d1], tf.add, name='add1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h1/b1')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h1/dr1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h1/b2')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h1/dr2')
        n = Conv2d(n, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h1/b3')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h1/dr3')

        n = ElementwiseLayer([n, d2], tf.add, name='add2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h2/b1')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h2/dr1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h2/b2')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h2/dr2')
        n = Conv2d(n, 512, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h2/b3')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h2/dr3')

        n = ElementwiseLayer([n, d3], tf.add, name='add3')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h3/b1')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h3/dr1')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h3/b2')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h3/dr2')
        n = Conv2d(n, 1024, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h3/b3')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h3/dr3')

        n = ElementwiseLayer([n, d4], tf.add, name='add4')
        n = Conv2d(n, 1024, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h4/b1')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h4/dr1')
        n = Conv2d(n, 1024, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h4/b2')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='h4/dr2')

        n = FlattenLayer(n, name='hf/flatten')
        n = DropoutLayer(n, keep=0.2, is_fix=True, is_train=is_train, name='hf/dr1')
        n = DenseLayer(n, n_units=1, act=tf.identity, W_init=w_init_sigmoid, name='hf/dense')
        logits = n.outputs

    return logits, tf.nn.sigmoid(logits)

def defocus_discriminator(t_image, is_train=False, reuse=False, scope='defocus_discriminator'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    b_init = None
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')

        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init_relu, name='h0/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h0/b1')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h0/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h0/b2')
        n = Conv2d(n, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h0/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h0/b3')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h1/b1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h1/b2')
        n = Conv2d(n, 128, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h1/b3')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h2/b1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h2/b2')
        n = Conv2d(n, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h2/b3')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h3/b1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h3/b2')
        n = Conv2d(n, 512, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h3/b3')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h4/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h4/b1')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init_relu, b_init=b_init, name='h4/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h4/b2')

        n = FlattenLayer(n, name='hf/flatten')
        n = DenseLayer(n, n_units=1, act=tf.identity, W_init=w_init_sigmoid, name='hf/dense')

        logits = n.outputs

        return logits, tf.nn.sigmoid(n.outputs)

# def UNet_down(image_in, is_train=False, reuse=False, scope = 'unet_down'):
#     w_init_relu = tf.contrib.layers.variance_scaling_initializer()
#     w_init_sigmoid = tf.contrib.layers.xavier_initializer()
#     #g_init = tf.random_normal_initializer(1., 0.02)
#     g_init = None
#     lrelu = lambda x: tl.act.lrelu(x, 0.2)
#     with tf.variable_scope(scope, reuse=reuse) as vs:
#         n = InputLayer(image_in, name='image_in')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d0/pad1')
#         n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d0/c1')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d0/b1')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d0/pad2')
#         n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d0/c2')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d0/b2')

#         n = Conv2d(n, 32, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, name='d1/pool1')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d1/pad1')
#         n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d1/c1')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d1/b1')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d1/pad2')
#         n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d1/c2')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d1/b2')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d1/pad3')
#         n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d1/c3')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d1/b3')

#         n = Conv2d(n, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, name='d2/pool1')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d2/pad1')
#         n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d2/c1')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d2/b1')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d2/pad2')
#         n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d2/c2')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d2/b2')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d2/pad3')
#         n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d2/c3')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d2/b3')

#         n = Conv2d(n, 128, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, name='d3/pool1')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d3/pad1')
#         n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d3/c1')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d3/b1')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d3/pad2')
#         n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d3/c2')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d3/b2')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d3/pad3')
#         n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d3/c3')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d3/b3')

#         n = Conv2d(n, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, name='d4/pool1')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d4/pad1')
#         n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d4/c1')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d4/b1')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d4/pad2')
#         n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d4/c2')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d4/b2')
#         n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d4/pad3')
#         n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d4/c3')
#         n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d4/b3')

#         return n.outputs
