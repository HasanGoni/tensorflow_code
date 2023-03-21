import numpy as np
import tensorflow as tf

def class_wise_metrics(y_true, y_pred):

    class_wise_iou = []
    class_wise_dice_score = []

    smoothening_factor = 0.00001
    for i in range(3):

        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area

        iou = (intersection + smoothening_factor) / (combined_area - intersection + smoothening_factor)
        class_wise_iou.append(iou)

        dice_score =  2 * ((intersection + smoothening_factor) / (combined_area + smoothening_factor))
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score


def make_prediction(model,
     image, mask,
     num=1):
    """
    Feeds an image to a model and returns the predicted mask.
    """
    image = np.reshape(
              image, (1, image.shape[0],
                      image.shape[1],
                      image.shape[2]))
    pred = model.predict(image)
    pred = tf.argmax(pred, axis=-1)
    pred = pred[..., tf.newaxis]
    pred = pred[0].numpy()
    return pred

