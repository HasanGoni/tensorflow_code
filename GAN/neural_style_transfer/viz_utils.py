import matplotlib.pyplot as plt
import tensorflow as tf


def imshow(image, title=None):
    """
    display images with title
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image,
                           axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)


def show_images_with_object(images, title=[]):
    """
    display a row of images with corresponding
    title
    """
    if len(images) != len(title):
        return
    plt.figure(figsize=(20, 12))
    for idx, (image, title) in enumerate(
                                  zip(images,
                                      title)):
        plt.subplot(1, len(images), idx+1)
        plt.xticks([])
        plt.yticks([])
        imshow(image, title)
        # plt.show()
