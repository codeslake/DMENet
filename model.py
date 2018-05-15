import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

def Vgg19_simple_api(rgb, reuse, scope):
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope(scope, reuse=reuse) as vs:
        rgb_scaled = rgb * 255.0
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else: # TF 1.0
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
        
        return network, [d0.outputs, d4.outputs], d2.outputs

def UNet_up(images, feats, is_train=False, reuse=False, scope = 'unet_up'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        d0 = InputLayer(feats[0], name='d0')
        d4 = InputLayer(feats[1], name='d4')

        u4 = d4
        u4 = PadLayer(u4, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4_aux/pad1')
        u4 = Conv2d(u4, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u4_aux/c1')
        u4 = BatchNormLayer(u4, act=lrelu, is_train = is_train, gamma_init = g_init, name='u4_aux/b1')
        u4 = PadLayer(u4, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4_aux/pad2')
        u4 = Conv2d(u4, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u4_aux/c2')
        u4 = BatchNormLayer(u4, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u4_aux/b2')
        u4 = u4.outputs

        n = UpSampling2dLayer(d4, (2, 2), is_scale = True, method = 1, align_corners=True, name='u3/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad3')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b3')

        u3 = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3_aux/pad1')
        u3 = Conv2d(u3, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3_aux/c1')
        u3 = BatchNormLayer(u3, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3_aux/b1')
        u3 = PadLayer(u3, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3_aux/pad2')
        u3 = Conv2d(u3, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u3_aux/c2')
        u3 = BatchNormLayer(u3, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u3_aux/b2')
        u3 = u3.outputs

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u2/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b3')

        u2 = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2_aux/pad1')
        u2 = Conv2d(u2, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2_aux/c1')
        u2 = BatchNormLayer(u2, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2_aux/b1')
        u2 = PadLayer(u2, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2_aux/pad2')
        u2 = Conv2d(u2, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u2_aux/c2')
        u2 = BatchNormLayer(u2, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u2_aux/b2')
        u2 = u2.outputs

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u1/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad3')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b3')

        u1 = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1_aux/pad1')
        u1 = Conv2d(u1, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1_aux/c1')
        u1 = BatchNormLayer(u1, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1_aux/b1')
        u1 = PadLayer(u1, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1_aux/pad2')
        u1 = Conv2d(u1, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u1_aux/c2')
        u1 = BatchNormLayer(u1, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u1_aux/b2')
        u1 = u1.outputs

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u0/u')
        n = ConcatLayer([n, d0], concat_dim = 3, name='u0/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad_init')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c_init')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b_init')
        for i in np.arange(7):
            n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad{}_1'.format(i))
            n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c{}_1'.format(i))
            n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b{}_1'.format(i))
            n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad{}_2'.format(i))
            n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c{}_2'.format(i))
            n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b{}_2'.format(i))

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad1')
        n = Conv2d(n, 1, (3, 3), (1, 1), act=tf.nn.sigmoid, padding='VALID', W_init=w_init_sigmoid, name='uf/c1')

        with tf.variable_scope('GRN') as scope:
            for i in np.arange(5):
                if i == 0 and reuse == False:
                    reuse_grn = False
                else:
                    reuse_grn = True

                residual = GRN(images, n.outputs, reuse = reuse_grn, scope = scope)
                n = ElementwiseLayer([n, residual], tf.subtract, name='grn{}/add'.format(i))

        return n.outputs, [u4, u3, u2, u1]

def GRN(image, blur_map, is_train=False, reuse=False, scope = 'GRN'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        image_n = InputLayer(image, name='image')
        blur_map_n = InputLayer(blur_map, name='blur_map')

        n = ConcatLayer([image_n, blur_map_n], concat_dim = 3, name='d0/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d0/pad1')
        n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d0/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d0/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d0/pad2')
        n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d0/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d0/b2')
        d0 = n

        n = Conv2d(n, 16, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, name='d1/pool1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d1/pad1')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d1/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d1/pad2')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d1/b2')
        d1 = n

        n = Conv2d(n, 32, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, name='d2/pool1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d2/pad1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d2/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d2/pad2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d2/b2')
        d2 = n

        n = Conv2d(n, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, name='d3/pool1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d3/pad1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d3/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d3/pad2')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d3/b2')
        d3 = n

        n = Conv2d(n, 128, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init_relu, name='d4/pool1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d4/pad1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d4/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d4/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='d4/pad2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d4/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d4/b2')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u3/u')
        n = ConcatLayer([n, d3], concat_dim = 3, name='u3_concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad2')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b2')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u2/u')
        n = ConcatLayer([n, d2], concat_dim = 3, name='u2_concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b2')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u1/u')
        n = ConcatLayer([n, d1], concat_dim = 3, name='u1_concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad1')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad2')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b2')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u0/u')
        n = ConcatLayer([n, d0], concat_dim = 3, name='u0_concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad1')
        n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad2')
        n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b2')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad1')
        n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad2')
        n = Conv2d(n, 1, (3, 3), (1, 1), act=tf.nn.sigmoid, padding='VALID', W_init=w_init_sigmoid, name='uf/c2')

        return n


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
        n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l1/b2')
        n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l1/b3')

        n = Conv2d(n, 32, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l2/b1')
        n = Conv2d(n, 32, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l2/b2')
        n = Conv2d(n, 32, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l2/b3')

        n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l3/b1')
        n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l3/b2')
        n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='l3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='l3/b3')

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
        n = InputLayer(feats, name='input_feature')

        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h0/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h0/b1')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h0/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h0/b2')
        n = Conv2d(n, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h0/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h0/b3')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h1/b1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h1/b2')
        n = Conv2d(n, 128, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h1/b3')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h2/b1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h2/b2')
        n = Conv2d(n, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h2/b3')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h3/b1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h3/b2')
        n = Conv2d(n, 512, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h3/b3')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h4/b1')
        n = Conv2d(n, 512, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='h4/b2')

        n = FlattenLayer(n, name='hf/flatten')
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
def UNet_up_test(images, feats, is_train=False, reuse=False, scope = 'unet_up'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        d0 = InputLayer(feats[0], name='d0')
        d4 = InputLayer(feats[1], name='d4')

        u4 = d4
        u4 = PadLayer(u4, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4_aux/pad1')
        u4 = Conv2d(u4, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u4_aux/c1')
        u4 = BatchNormLayer(u4, act=lrelu, is_train = is_train, gamma_init = g_init, name='u4_aux/b1')
        u4 = PadLayer(u4, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4_aux/pad2')
        u4 = Conv2d(u4, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u4_aux/c2')
        u4 = BatchNormLayer(u4, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u4_aux/b2')
        u4 = u4.outputs

        n = UpSampling2dLayer(d4, (2, 2), is_scale = True, method = 1, align_corners=True, name='u3/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad3')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b3')

        u3 = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3_aux/pad1')
        u3 = Conv2d(u3, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3_aux/c1')
        u3 = BatchNormLayer(u3, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3_aux/b1')
        u3 = PadLayer(u3, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3_aux/pad2')
        u3 = Conv2d(u3, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u3_aux/c2')
        u3 = BatchNormLayer(u3, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u3_aux/b2')
        u3 = u3.outputs

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u2/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b3')

        u2 = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2_aux/pad1')
        u2 = Conv2d(u2, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2_aux/c1')
        u2 = BatchNormLayer(u2, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2_aux/b1')
        u2 = PadLayer(u2, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2_aux/pad2')
        u2 = Conv2d(u2, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u2_aux/c2')
        u2 = BatchNormLayer(u2, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u2_aux/b2')
        u2 = u2.outputs

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u1/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad3')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b3')

        u1 = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1_aux/pad1')
        u1 = Conv2d(u1, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1_aux/c1')
        u1 = BatchNormLayer(u1, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1_aux/b1')
        u1 = PadLayer(u1, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1_aux/pad2')
        u1 = Conv2d(u1, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_sigmoid, name='u1_aux/c2')
        u1 = BatchNormLayer(u1, act=tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name='u1_aux/b2')
        u1 = u1.outputs

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u0/u')
        n = ConcatLayer([n, d0], concat_dim = 3, name='u0/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad_init')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c_init')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b_init')
        u0_init = n.outputs
        refine_lists = []
        for i in np.arange(7):
            n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad{}_1'.format(i))
            n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c{}_1'.format(i))
            n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b{}_1'.format(i))
            n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad{}_2'.format(i))
            n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c{}_2'.format(i))
            n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b{}_2'.format(i))
            refine_lists.append(n.outputs)

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad1')
        n = Conv2d(n, 1, (3, 3), (1, 1), act=tf.nn.sigmoid, padding='VALID', W_init=w_init_sigmoid, name='uf/c1')
        refine_lists.append(n.outputs)

        with tf.variable_scope('GRN') as scope:
            grn_residual_lists = []
            grn_sum_lists = []
            for i in np.arange(5):
                if i == 0 and reuse == False:
                    reuse_grn = False
                else:
                    reuse_grn = True

                residual = GRN(images, n.outputs, reuse = reuse_grn, scope = scope)
                grn_residual_lists.append(residual.outputs)
                n = ElementwiseLayer([n, residual], tf.subtract, name='grn{}/add'.format(i))
                grn_sum_lists.append(n.outputs)

        return n.outputs, [u4, u3, u2, u1, u0_init], refine_lists, grn_residual_lists, grn_sum_lists
