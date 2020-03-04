import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda, Input, concatenate, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential


def spectral_norm(x, iteration=1):
    """
    following taki0112's implement
    :param x:
    :param iteration:
    :return:
    """
    with tf.compat.v1.variable_scope("spectral_norm"):
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
            w_norm = tf.reshape(w_norm, [-1] + x_shape[1:])
        return w_norm


def CustomConv2D(filters=64, kernel_size=7, strides=1, padding='VALID', name='conv2d', stddev=0.02, do_relu=True,
                 do_norm=True, do_sp_norm=False, leaky_relu_alpha=0.2):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                      bias_initializer=tf.constant_initializer()))

    if do_norm:
        result.add(tf.keras.layers.BatchNormalization())

    if do_relu:
        if leaky_relu_alpha != 0:
            result.add(tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha))
        else:
            result.add(tf.keras.layers.ReLU())

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
    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                        kernel_initializer=tf.random_normal_initializer(0., stddev),
                                        bias_initializer=tf.constant_initializer()))

    if do_norm:
        result.add(tf.keras.layers.BatchNormalization())

    if do_relu:
        if leaky_relu_alpha != 0:
            result.add(tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha))
        else:
            result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    n_G_filter = 32
    k_size = 3

    cnn_stack = [
        Lambda(lambda tensor: tf.pad(tensor, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT"), name = 'padding_A'),
        CustomConv2D(filters=n_G_filter, kernel_size=7, strides=1, padding='VALID', name='down_1A'),            # 1*256*256*32
        CustomConv2D(filters=n_G_filter * 2, kernel_size=k_size, strides=2, padding='SAME', name='down_2A'),    # 1*128*128*64
        CustomConv2D(filters=n_G_filter * 4, kernel_size=k_size, strides=2, padding='SAME', name='down_3A')     # 1*64*64*128
    ]
    core_stack = [
        ResnetBlock(filters=n_G_filter * 8, name='ResNet_Block_1'),
        ResnetBlock(filters=n_G_filter * 8, name='ResNet_Block_2'),
        ResnetBlock(filters=n_G_filter * 8, name='ResNet_Block_3'),
        ResnetBlock(filters=n_G_filter * 8, name='ResNet_Block_4'),
        ResnetBlock(filters=n_G_filter * 8, name='ResNet_Block_5'),
        ResnetBlock(filters=n_G_filter * 8, name='ResNet_Block_6'),
        ResnetBlock(filters=n_G_filter * 8, name='ResNet_Block_7'),
        ResnetBlock(filters=n_G_filter * 8, name='ResNet_Block_8'),
        ResnetBlock(filters=n_G_filter * 8, name='ResNet_Block_9')
    ]
    ctnn_stack = [
        CustomConv2DTranspose(filters=n_G_filter * 2, kernel_size=k_size, strides=2, padding='SAME', name='up_1A'),
        CustomConv2DTranspose(filters=n_G_filter, kernel_size=k_size, strides=2, padding='SAME', name='up_2A'),
        CustomConv2DTranspose(filters=3, kernel_size=k_size, strides=1, padding='SAME', name='up_3A', do_relu=False)
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
    x_B = Lambda(lambda tensor : tf.slice(tensor, [0, 0, 0, 128], [1, 64, 64, 128]), name = 'Split_B')(merge_core) # --> Use tf.Tensor.getitem & Lambda layer

    CTNN_A_model = Sequential(layers = ctnn_stack, name = 'CTNN_A')
    CTNN_B_model = Sequential(layers = ctnn_stack, name = 'CTNN_B')

    x_A = CTNN_A_model(x_A)
    x_B = CTNN_B_model(x_B)

    out_A = Activation(activation = 'tanh', name="A_tanh")(x_A)
    out_B = Activation(activation = 'tanh', name="B_tanh")(x_B)

    return tf.keras.Model(inputs=inputs, outputs=[out_A, out_B], name = 'Generator')
