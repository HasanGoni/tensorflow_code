import numpy as np


def get_images_and_segments_test_arrays(validation_dataset):
    '''
    Gets a subsample of the val set as your test set
    Returns:
    Test set containing ground truth images and label maps
    '''
    y_true_segments = []
    y_true_images = []
    test_count = 64

    ds = validation_dataset.unbatch()
    ds = ds.batch(101)

    for image, annotation in ds.take(1):
        y_true_images = image
        y_true_segments = annotation
    y_true_segments = y_true_segments[:test_count, :, :, :]
    y_true_segments = np.argmax(y_true_segments, axis=3)

    return y_true_images, y_true_segments


def compute_metrics(y_true, y_pred):
    '''
    Computes IOU and Dice Score.
    Args:
    y_true (tensor) - ground truth label map
    y_pred (tensor) - predicted label map
    '''
    class_wise_iou = []
    class_wise_dice_score = []

    smoothening_factor = 0.00001

    for i in range(12):
        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area

        iou = (intersection + smoothening_factor) / (combined_area - intersection + smoothening_factor)
        class_wise_iou.append(iou)
        dice_score =  2 * ((intersection + smoothening_factor) / (combined_area + smoothening_factor))
        class_wise_dice_score.append(dice_score)
    return class_wise_iou, class_wise_dice_score
