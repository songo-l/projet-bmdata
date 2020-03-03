import tensorflow as tf

MAKEUP_DIR='all_images/all_images/'

def create_datasets(path):
    images_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, horizontal_flip=True)
    images = images_gen.flow_from_directory(
        path, target_size=(256, 256), color_mode='rgb', classes=None,
        class_mode='categorical', batch_size=32, shuffle=False, seed=None,
        save_to_dir=None, save_prefix='', save_format='png', follow_links=False,
        subset=None, interpolation='nearest'
        )
    return images
