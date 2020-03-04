import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda, Input, concatenate
from tensorflow.keras.models import Model


def spectral_norm(x, iteration=1):
    """
    following taki0112's implement
    :param x:
    :param iteration:
    :return:
    """
    with tf.variable_scope("spectral_norm"):
        x_shape = x.shape.as_list()
        w = tf.reshape(x, [-1, x_shape[-1]])
        u = tf.get_variable("u", [1, x_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
        u_hat = u
        v_hat = None

        for i in range(iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_, dim=None)
            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_, dim=None)
        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, [-1]+x_shape[1:])
        return w_norm

def instance_norm(x):
    """
    following baldFemale's implement
    :param x:
    :return:
    """
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean,var = tf.nn.moments(x,[1,2],keep_dims=True)
        scale = tf.get_variable(name="scale",shape=[x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(stddev=0.02,mean=1.0))
        offset = tf.get_variable(name="offset",shape=[x.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean,tf.sqrt(var+epsilon))+offset
        return out

class Conv2D_Custom(tf.keras.Model):
    def __init__(self, filters = 64, kernel_size = 7, strides = 1, padding = 'VALID', name = 'conv2d', stddev=0.02,
                 do_relu=True, do_norm=True, do_sp_norm=False, leaky_relu_alpha=0.2):
        
        super(Conv2D_Custom, self).__init__(name='')
        
        self.alpha = leaky_relu_alpha
        
        self.do_relu = do_relu
        self.do_norm = do_norm
        self.do_sp_norm = do_sp_norm

        self.conv2d = Conv2D(filters = filters, 
                             kernel_size = kernel_size,
                             strides = strides,
                             padding = padding,
                             activation=None, 
                             kernel_initializer = tf.truncated_normal_initializer(stddev = stddev),
                             bias_initializer = tf.constant_initializer(0.0))

    def call(self, input_tensor, training=False):
        x = self.conv2d(input_tensor)
        
        if self.do_norm:
            x = instance_norm(x)

        if self.do_sp_norm:
            x = spectral_norm(x)

        if self.do_relu:
            if self.alpha != 0:
                x = tf.nn.leaky_relu(x, alpha = self.alpha, name = "lrelu")
            else:
                x = tf.nn.relu(x, name="relu")
        
        return x
