import tensorflow_datasets as tfds
import tensorflow as tf


def map_image(image,
              label):
    """
    map the image based on this function
    image: image of dataset
    label: label of the dataset
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.reshape(image, shape=(28, 28, 1,))
    return image


def get_dataset(map_fn,
                valid=True,
                 bs=128):
    """
    get dataset from tensorflow dataset
    """
    split = 'train' if not valid else 'test'
    dataset = tfds.load('mnist', as_supervised=True,
                        split=split)
    dataset = dataset.map(map_fn,
                          tf.data.experimental.AUTOTUNE)
    if not valid:
        dataset = dataset.shuffle(buffer_size=1024).batch(bs).repeat()
    else:
        dataset = dataset.map(map_fn).batch(bs).repeat()
    return dataset
