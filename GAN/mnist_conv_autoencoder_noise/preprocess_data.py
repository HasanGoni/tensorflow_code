import tensorflow as tf


def map_image(image, label):
    """
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    noise_factor = 0.5
    factor = noise_factor * tf.random.normal(shape=image.shape)
    noisy_image = image + factor
    noisy_image = tf.clip_by_value(image, 0.0, 0.1)
    return noisy_image, image
