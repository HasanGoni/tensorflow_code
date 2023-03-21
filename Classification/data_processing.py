import tensorflow as tf


def preprocess_image(input_images):
    """
    preprocess image from 255 range to 1 range
    """
    input_images = input_images.astype('float32')
    out_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    return out_ims

