import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda, Input, concatenate, Activation, LeakyReLU
# from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model
from tensorflow.python.keras import Sequential
from layers import SpectralNormalization, CustomConv2D, ResnetBlock, CustomConv2DTranspose


def Generator():
    filters = 32
    k_size = 3

    cnn_stack = [
        Lambda(lambda tensor: tf.pad(tensor, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT"), name='padding'),
        CustomConv2D(filters=filters, kernel_size=7, strides=1, padding='VALID', name='cnn_1'),  # 1*256*256*32
        CustomConv2D(filters=filters * 2, kernel_size=k_size, strides=2, padding='SAME', name='cnn_2'),  # 1*128*128*64
        CustomConv2D(filters=filters * 4, kernel_size=k_size, strides=2, padding='SAME', name='cnn_3')  # 1*64*64*128
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

    # for down_A, down_B in zip(down_stack_A, down_stack_B):
    #    x_A = down_A(x_A)
    #    x_B = down_B(x_B)
    CNN_A_model = Sequential(layers=cnn_stack, name='CNN_A')
    CNN_B_model = Sequential(layers=cnn_stack, name='CNN_B')

    x_A = CNN_A_model(x_A)
    x_B = CNN_B_model(x_B)

    merge_core = concatenate([x_A, x_B], name='Merge_AB')  # 1*64*64*256

    core_model = Sequential(layers=core_stack, name='ResNet')

    merge_core = core_model(merge_core)

    # for core in core_stack:
    #    merge_core = core(merge_core)

    x_A = Lambda(lambda tensor: tf.slice(tensor, [0, 0, 0, 0], [1, 64, 64, 128]), name='Split_A')(merge_core)
    x_B = Lambda(lambda tensor: tf.slice(tensor, [0, 0, 0, 128], [1, 64, 64, 128]), name='Split_B')(
        merge_core)  # --> Pytonic way : use tf.Tensor.getitem & Lambda layer

    CTNN_A_model = Sequential(layers=ctnn_stack, name='CTNN_A')
    CTNN_B_model = Sequential(layers=ctnn_stack, name='CTNN_B')

    x_A = CTNN_A_model(x_A)
    x_B = CTNN_B_model(x_B)

    out_A = Activation(activation='tanh', name="tanh_A")(x_A)
    out_B = Activation(activation='tanh', name="tanh_B")(x_B)

    return Model(inputs=inputs, outputs=[out_A, out_B], name='Generator')


def Discriminator():
    filters = 64
    k_size = 4

    def sp_norm(name):
        return lambda tensor: SpectralNormalization(tensor, name=name)

    cnn_stack = [
        CustomConv2D(filters=filters, kernel_size=k_size, strides=2, padding="SAME", name="Conv_1", do_norm=False),
        # 1*128*128*64
        SpectralNormalization(Conv2D(filters=filters * 2, kernel_size=k_size, strides=2, padding="SAME", name="conv_2",
                                     kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                     bias_initializer=tf.constant_initializer()), name="Sp_norm1"),
        LeakyReLU(alpha=0.2, name='LReLu_1'),
        SpectralNormalization(Conv2D(filters=filters * 4, kernel_size=k_size, strides=2, padding="SAME", name="conv_3",
                                     kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                     bias_initializer=tf.constant_initializer()), name="Sp_norm2"),
        LeakyReLU(alpha=0.2, name='LReLu_2'),
        SpectralNormalization(Conv2D(filters=filters * 8, kernel_size=k_size, strides=1, padding="SAME", name="conv_4",
                                     kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                     bias_initializer=tf.constant_initializer()), name="Sp_norm3"),
        LeakyReLU(alpha=0.2, name='LReLu_3'),
        CustomConv2D(filters=1, kernel_size=k_size, strides=1, padding="SAME", name="Conv_5", do_norm=False,
                     do_sp_norm=False, do_relu=False)
    ]

    input = x = Input(shape=[255, 255, 3], name='Input')

    for layer in cnn_stack:
        x = layer(x)

    return Model(inputs=input, outputs=x, name='Discriminator')
