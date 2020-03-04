import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Lambda, Input, concatenate, Activation, Wrapper, BatchNormalization, LeakyReLU, ReLU
#from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.keras import Sequential, layers, initializers, backend as K
from tensorflow.python.ops import array_ops, math_ops


class SpectralNormalization(layers.Wrapper):
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


def CustomConv2D(filters=64, kernel_size=7, strides=1, padding='VALID', name='conv2d', stddev=0.02, do_relu=True,
                 do_norm=True, do_sp_norm=False, leaky_relu_alpha=0.2):
    result = Sequential(name=name)
    result.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      kernel_initializer=tf.random_normal_initializer(0., 0.02),
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


def Generator():
    filters = 32
    k_size = 3

    cnn_stack = [
        Lambda(lambda tensor: tf.pad(tensor, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT"), name = 'padding'),
        CustomConv2D(filters=filters, kernel_size=7, strides=1, padding='VALID', name='cnn_1'),            # 1*256*256*32
        CustomConv2D(filters=filters * 2, kernel_size=k_size, strides=2, padding='SAME', name='cnn_2'),    # 1*128*128*64
        CustomConv2D(filters=filters * 4, kernel_size=k_size, strides=2, padding='SAME', name='cnn_3')     # 1*64*64*128
    ]
    core_stack = [
        ResnetBlock(filters=filters * 8, name='ResNet_Block_1'),
        ResnetBlock(filters=filters * 8, name='ResNet_Block_2'),
        ResnetBlock(filters=filters * 8, name='ResNet_Block_3'),
        ResnetBlock(filters=filters * 8, name='ResNet_Block_4'),
        ResnetBlock(filters=filters * 8, name='ResNet_Block_5'),
        ResnetBlock(filters=filters * 8, name='ResNet_Block_6'),
        ResnetBlock(filters=filters * 8, name='ResNet_Block_7'),
        ResnetBlock(filters=filters * 8, name='ResNet_Block_8'),
        ResnetBlock(filters=filters * 8, name='ResNet_Block_9')
    ]
    ctnn_stack = [
        CustomConv2DTranspose(filters=filters * 2, kernel_size=k_size, strides=2, padding='SAME', name='ctnn_1'),
        CustomConv2DTranspose(filters=filters, kernel_size=k_size, strides=2, padding='SAME', name='ctnn_2'),
        CustomConv2DTranspose(filters=3, kernel_size=k_size, strides=1, padding='SAME', name='ctnn_3', do_relu=False)
    ]

    inputs = x_A, x_B = [Input(shape=[255, 255, 3], name='Input_A'), Input(shape=[255, 255, 3], name='Input_B')]

    #for down_A, down_B in zip(down_stack_A, down_stack_B):
    #    x_A = down_A(x_A)
    #    x_B = down_B(x_B)
    CNN_A_model = Sequential(layers = cnn_stack, name = 'CNN_A')
    CNN_B_model = Sequential(layers = cnn_stack, name = 'CNN_B')

    x_A = CNN_A_model(x_A)
    x_B = CNN_B_model(x_B)

    merge_core = concatenate([x_A, x_B], name='Merge_AB')  # 1*64*64*256

    core_model = Sequential(layers = core_stack, name = 'ResNet')

    merge_core = core_model(merge_core)

    #for core in core_stack:
    #    merge_core = core(merge_core)

    x_A = Lambda(lambda tensor : tf.slice(tensor, [0, 0, 0, 0], [1, 64, 64, 128]), name = 'Split_A')(merge_core)
    x_B = Lambda(lambda tensor : tf.slice(tensor, [0, 0, 0, 128], [1, 64, 64, 128]), name = 'Split_B')(merge_core) # --> Pytonic way : use tf.Tensor.getitem & Lambda layer

    CTNN_A_model = Sequential(layers = ctnn_stack, name = 'CTNN_A')
    CTNN_B_model = Sequential(layers = ctnn_stack, name = 'CTNN_B')

    x_A = CTNN_A_model(x_A)
    x_B = CTNN_B_model(x_B)

    out_A = Activation(activation = 'tanh', name="tanh_A")(x_A)
    out_B = Activation(activation = 'tanh', name="tanh_B")(x_B)

    return tf.keras.Model(inputs=inputs, outputs=[out_A, out_B], name = 'Generator')

def Discriminator():
    filters = 64
    k_size = 4

    def sp_norm(name):
        return lambda tensor : SpectralNormalization(tensor, name = name)

    cnn_stack = [
        CustomConv2D(filters = filters,  kernel_size = k_size, strides = 2, padding = "SAME", name = "Conv_1", do_norm = False),                 # 1*128*128*64
        SpectralNormalization(Conv2D(filters = filters  * 2, kernel_size = k_size, strides = 2, padding= "SAME", name = "conv_2",kernel_initializer = tf.random_normal_initializer(0., 0.02), bias_initializer = tf.constant_initializer()), name = "Sp_norm1"),
        LeakyReLU(alpha = 0.2, name = 'LReLu_1'),
        SpectralNormalization(Conv2D(filters = filters  * 4, kernel_size = k_size, strides = 2, padding= "SAME", name = "conv_3",kernel_initializer = tf.random_normal_initializer(0., 0.02), bias_initializer = tf.constant_initializer()), name = "Sp_norm2"),
        LeakyReLU(alpha = 0.2, name = 'LReLu_2'),
        SpectralNormalization(Conv2D(filters = filters  * 8, kernel_size = k_size, strides = 1, padding= "SAME", name = "conv_4",kernel_initializer = tf.random_normal_initializer(0., 0.02), bias_initializer = tf.constant_initializer()), name = "Sp_norm3"),
        LeakyReLU(alpha = 0.2, name = 'LReLu_3'),
        CustomConv2D(filters = 1, kernel_size = k_size, strides = 1, padding = "SAME", name = "Conv_5", do_norm = False, do_sp_norm = False,do_relu = False)
    ]

    input = x = Input(shape=[255, 255, 3], name='Input')

    for layer in cnn_stack:
        x = layer(x)

    return tf.keras.Model(inputs=input, outputs=x, name = 'Discriminator')
