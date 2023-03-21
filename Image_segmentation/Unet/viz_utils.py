import matplotlib.pyplot as plt
import tensorflow as tf
from configuration_file import BATCH_SIZE
from get_data import info
from configuration_file import class_names

def show_image_from_dataset(dataset):
    """
    display dataset from dataset
    """
    for image, mask in dataset.take(1):
        sample_image, sample_mask = image, mask
    display(
        [sample_image,
         sample_mask],
        titles=['Image',
                'True mask'])


def display(images_list,
            titles=[],
            display_str=None):
    """
    display list of images with titles,
    and metrics
    """
    plt.figure(figsize=(15, 15))

    for i in range(len(images_list)):
        plt.subplot(
            1, len(images_list), i+1)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
        if display_str and i == 1:
            plt.xlabel(
                display_str,
                fontsize=12)
        img_arr = tf.keras.preprocessing.image.array_to_img(images_list[i])
        plt.imshow(img_arr)
    plt.show()


def plot_metrics(model_history, metric_name, title,
                 ylim=1.2):
    """
    plots a given metric from the model history'
    """
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(model_history.history[metric_name], '.-',
             color='blue', label=metric_name)
    plt.plot(model_history.history[f'val_{metric_name}'], '.-',
             color='green', label=f'val_{metric_name}')


def get_test_image_and_annotation_array(test_dataset):
    """
    unpack test dataset and put them in a
    numpy array
    """
    ds = test_dataset.unbatch()
    test_data_length = info.splits['test'].num_examples

    ds = ds.batch(test_data_length)
    rounded_testset = test_data_length - (test_data_length % BATCH_SIZE)

    images_array = []
    annotations_array = []
    for image, annotation in ds.take(1):
        images_array = image.numpy()
        annotations_array = annotation.numpy()

    images_array = images_array[: rounded_testset]
    annotations_array = annotations_array[: rounded_testset]

    return images_array, annotations_array




def display_with_metrics(display_list, iou_list, dice_score_list):
    '''displays a list of images/masks and overlays a list of IOU and Dice Scores'''

    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list)) if iou > 0.0]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place

    display_string_list = ["{}: IOU: {} Dice Score: {}".format(class_names[idx], iou, dice_score) for idx, iou, dice_score in metrics_by_id]
    display_string = "\n\n".join(display_string_list)

    display(display_list, ["Image", "Predicted Mask", "True Mask"], display_str=display_string)
