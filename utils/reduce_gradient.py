import tensorflow as tf
from tensorflow.python.framework import ops

class ReduceGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "ReduceGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _reduce_gradients(op, grad):
            return [grad * l]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y
    
reduce_gradient = ReduceGradientBuilder()