import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds


def display_one_row(display_images,
                    offset,
                    shape=(28, 28)):
    plt.figure(figsize=(20, 12))
    for idx, im in enumerate(display_images):
        plt.subplot(3, 10, offset + idx + 1)
        plt.xticks([])
        plt.yticks([])
        im = np.reshape(im, shape)
        plt.imshow(im, cmap='gray')


def show_batch(dataset,
               number_of_images=9,
               im_height=28,
               im_width=28):
    """
    vizualize random dataset
    of number_of_imges images from of a dataset
    """
    # getting one batch from the data
    ds = dataset.take(1)
    # Converting one batch dataset to list
    image_list = []
    for image, im in tfds.as_numpy(ds):
        image_list = image
    idxs = np.random.choice(64, number_of_images)
    image_list = np.array(image_list[idxs])
    image_list = np.reshape(image_list,
                            (number_of_images, im_height,
                             im_width, 1))
    display_one_row(image_list, 0,
                    shape=(im_height,
                           im_width))


def display_one_row(disp_images, offset, shape=(28, 28)):
    '''Display sample outputs in one row.'''
    for idx, test_image in enumerate(disp_images):
        plt.subplot(3, 10, offset + idx + 1)
        plt.xticks([])
        plt.yticks([])
        test_image = np.reshape(test_image, shape)
        plt.imshow(test_image, cmap='gray')


def display_results(disp_input_images,
                    disp_encoded,
                    disp_predicted,
                    enc_shape=(8, 4)):
    '''Displays the input, encoded, and decoded output values.'''
    plt.figure(figsize=(15, 5))
    display_one_row(disp_input_images, 0, shape=(28, 28,))
    display_one_row(disp_encoded, 10, shape=enc_shape)
    display_one_row(disp_predicted, 20, shape=(28, 28,))
