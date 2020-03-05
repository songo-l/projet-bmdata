import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Lambda, Input, concatenate, Activation, Wrapper, BatchNormalization, LeakyReLU, ReLU
#from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.keras import Sequential, layers, initializers, backend as K
from tensorflow.python.ops import array_ops, math_ops


class SpectralNormalization(Wrapper):
    """
    Attributes:
       layer: tensorflow keras layers (with kernel attribute)

    Source: https://medium.com/@FloydHsiu0618/spectral-normalization-implementation-of-tensorflow-2-0-keras-api-d9060d26de77
    """

    def __init__(self, layer, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        """Build `Layer`"""

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    '`SpectralNormalization` must wrap a layer that'
                    ' contains a `kernel` for weights')

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_weight(
                shape=tuple([1, self.w_shape[-1]]),
                initializer=initializers.TruncatedNormal(stddev=0.02),
                name='sn_u',
                trainable=False,
                dtype=dtypes.float32)

        super(SpectralNormalization, self).build()

    @def_function.function
    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training==None:
            training = K.learning_phase()

        if training==True:
            # Recompute weights for each forward pass
            self._compute_weights()

        output = self.layer(inputs)
        return output

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        w_reshaped = array_ops.reshape(self.w, [-1, self.w_shape[-1]])
        eps = 1e-12
        _u = array_ops.identity(self.u)
        _v = math_ops.matmul(_u, array_ops.transpose(w_reshaped))
        _v = _v / math_ops.maximum(math_ops.reduce_sum(_v**2)**0.5, eps)
        _u = math_ops.matmul(_v, w_reshaped)
        _u = _u / math_ops.maximum(math_ops.reduce_sum(_u**2)**0.5, eps)

        self.u.assign(_u)
        sigma = math_ops.matmul(math_ops.matmul(_v, w_reshaped), array_ops.transpose(_u))

        self.layer.kernel.assign(self.w / sigma)

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())


def CustomConv2D(filters=64, kernel_size=7, strides=1, padding='VALID', name='conv2d', stddev=0.02, do_relu=True, do_norm=True, do_sp_norm=False, leaky_relu_alpha=0.2):
    result = Sequential(name=name)
    result.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      kernel_initializer=tf.random_normal_initializer(0., 0.02),
                      bias_initializer=tf.constant_initializer()))

    if do_norm:
        result.add(BatchNormalization())
        #result.add(InstanceNormalization())

    if do_relu:
        if leaky_relu_alpha != 0:
            result.add(LeakyReLU(alpha=leaky_relu_alpha))
        else:
            result.add(ReLU())

    return result


class ResnetBlock(Model):
    def __init__(self, filters=32, name='ResNet_Block'):
        super(ResnetBlock, self).__init__(name=name)

        self.conv2a = CustomConv2D(filters=filters, kernel_size=3, strides=1, padding='VALID', name='c1')
        self.conv2b = CustomConv2D(filters=filters, kernel_size=3, strides=1, padding='VALID', name='c2', do_relu=False)

    def call(self, input_tensor, training=False):
        x = tf.pad(input_tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        x = self.conv2a(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        x = self.conv2b(x)

        return tf.nn.relu(input_tensor + x)


def CustomConv2DTranspose(filters=64, kernel_size=7, strides=1, padding='VALID', name='deconv2d', stddev=0.02,
                          do_relu=True, do_norm=True, do_sp_norm=False, leaky_relu_alpha=0.2):
    result = Sequential(name=name)
    result.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                               kernel_initializer=tf.random_normal_initializer(0., stddev),
                               bias_initializer=tf.constant_initializer()))

    if do_norm:
        result.add(tf.keras.layers.BatchNormalization())
        #result.add(InstanceNormalization())

    if do_relu:
        if leaky_relu_alpha != 0:
            result.add(LeakyReLU(alpha=leaky_relu_alpha))
        else:
            result.add(ReLU())

    return result
