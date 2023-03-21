import numpy as np
from six import BytesIO
from six.moves.urllib.request import urlopen
from PIL import Image
import tensorflow as tf


def load_image_into_numpy_array(path):
    """
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a
    numpy array with shape (height, width, channels),
    where channels=3 for RGB.

    Args:
    path: the file path to the image

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = None

    if path.startswith('http'):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(image_data)

    im_width, im_height = (image.size)
    return np.array(image.getdata()).reshape(
                   (1, im_height, im_width,
                    3)).astype(np.uint8)
