import tensorflow as tf

def prec_loss_cal(input_tensor):
    initial_model = vgg16.VGG16(input_tensor=input_tensor, weights="imagenet", include_top=False)
    initial_model.trainable=False
    return initial_model.get_layer('block4_conv1').output
