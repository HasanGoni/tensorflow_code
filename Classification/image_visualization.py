import matplotlib.pyplot as plt
import os
import numpy as np

# import tensorflow as tf

# Matplotlib config
plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')
mpl_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def display_images(digits, predictions, labels, title, number, classes):
    """
    display random images
    """
    indexes = np.random.choice(len(predictions), size=number)
    n_digits = digits[indexes]
    n_predictions = predictions[indexes]
    n_predictions = n_predictions.reshape((number,))
    n_labels = labels[indexes]

    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])
    for i in range(number):
        ax = fig.add_subplot(1, number,  i+1)
        class_index = n_predictions[i]
        plt.xlabel(classes[class_index])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(n_digits[i])


def plt_metrics(history, metric_name, title, ylim=5):
    """
    training metric plotting
    """
    plt.title(title)
    plt.ylim(ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history[f'val_{metric_name}'], color='green', label=f'val_{metric_name}')
