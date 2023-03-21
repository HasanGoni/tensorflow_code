import tensorflow as tf


def map_image(image, label):
    """
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image, image
