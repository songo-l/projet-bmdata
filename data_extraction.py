import tensorflow as tf

MAKEUP_DIR='all_images/all_images/'

#retourne un ImageDataGenerator qui stocke les images et les labels sous la forme
# np array [batch, image/one_hot, index in the batch]

def create_datasets(path, batch_size=32):
    images_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, horizontal_flip=True)
    images = images_gen.flow_from_directory(
        path, target_size=(256, 256), color_mode='rgb', classes=None,
        class_mode='categorical', batch_size=batch_size, shuffle=False, seed=None,
        save_to_dir=None, save_prefix='', save_format='png', follow_links=False,
        subset=None, interpolation='nearest'
        )
    return images

# pour retrouver le label d'une image à partir de son one hot
# one_hot : np array codant la classe
# indices : dictionnaire obtenu avec x.class_indices
def get_label(one_hot, indices):
    inv_indices = {v: k for k, v in indices.items()}
    return inv_indices[np.nonzero(one_hot)[0][0]]
