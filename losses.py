import tensorflow as tf
from tensorflow.keras.applications import vgg16

#perceptual loss
def perc_loss_cal(input_tensor):
    initial_model = vgg16.VGG16(input_tensor=input_tensor, weights="imagenet", include_top=False)
    initial_model.trainable=False
    return initial_model.get_layer('block4_conv1').output

#loss sur le discriminator
def d_loss(data,data_gen):
    return tf.reduce_mean(tf.square(data_gen))+tf.reduce_mean(tf.squared_difference(data,1))/2.0

# Make up loss
def make_up_loss(generated, target):
    return 0
