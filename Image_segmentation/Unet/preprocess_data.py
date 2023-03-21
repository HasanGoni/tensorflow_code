import tensorflow as tf


def random_flip(input_image, input_mask):
    """
    randomly flip the data and segmentaion
    mask
    """
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(
            input_image)
        input_mask = tf.image.flip_left_right(
            input_mask)
    return input_image, input_mask


def normalize(input_image, input_mask):
    """
    normalize image to have a range of [0, 1]
    substructs 1 from mask to have a range
    [0, 2]
    """
    input_image = tf.cast(input_image, tf.float32)
    input_image = input_image / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    """
    resize, random_flip and normalize
    the image
    """
    input_image = tf.image.resize(
                   datapoint['image'],
                   (128, 128), method='nearest')
    input_mask = tf.image.resize(
     datapoint['segmentation_mask'],                   (128, 128), method='nearest')
    input_image, input_mask = random_flip(
                               input_image,
                               input_mask)
    input_image, input_mask = normalize(
                               input_image,
                               input_mask)
    return input_image, input_mask


@tf.function
def load_image_test(datapoint):
    """
    resize, normalize
    the image
    """
    input_image = tf.image.resize(
                   datapoint['image'],
                   (128, 128), method='nearest')
    input_mask = tf.image.resize(
     datapoint['segmentation_mask'],                   (128, 128), method='nearest')
    input_image, input_mask = normalize(
                               input_image,
                               input_mask)
    return input_image, input_mask
