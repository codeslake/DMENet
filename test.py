import tensorflow as tf
import numpy as np
import tensorlayer as tl
from tensorlayer.layers import *

sess = tf.Session()

input = tf.placeholder('float32', [None, None, None, 3], name = 'input')
#size = input.get_shape().as_list()
size = tf.shape(input)
print(size)

n = InputLayer(input, name = 'in')
n = UpSampling2dLayer(n, size=[size[1] * 2, size[2] * 2], is_scale=False, method=1, align_corners=False, name='up1/upsample2d')

n = n.outputs

#init_op = tf.global_variables_initializer()
#sess.run(init_op)

a = np.random.rand(1, 3, 3, 3)
result = sess.run(n, feed_dict = {input: a})

print(a.shape)
print(result.shape)

