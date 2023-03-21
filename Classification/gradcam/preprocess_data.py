import tensorflow as tf
import numpy as np
from configuration_file import IMAGE_SIZE


def format_image(image, label):
    """
    resize and normalize the data
    """
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, label


def preprocess_image(image):
    """
    adding extra dimension to the image
    so that it can be usable in the model
    prediction
    """
    image = np.expand_dims(image, axis=0)
    return image
