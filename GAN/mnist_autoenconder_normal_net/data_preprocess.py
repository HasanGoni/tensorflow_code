import tensorflow as tf


def map_image(image,
              label):
    """
    preprocess image before training
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.reshape(image, shape=(784,))
    return image, image
