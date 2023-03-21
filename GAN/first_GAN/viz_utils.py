from IPython import display
import matplotlib.pyplot as plt
import numpy as np


def plot_multiple_images(images,
                         n_cols=None):
    """
    visualize images
    """
    display.clear_output(wait=False)
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images,
                            axis=-1)
    plt.figure(figsize=(n_cols,
                        n_rows))
    for idx, image in enumerate(images):
        plt.subplot(n_rows,
                    n_cols,
                    idx + 1)
        plt.imshow(image,
                   cmap='binary')
        plt.axis('off')
