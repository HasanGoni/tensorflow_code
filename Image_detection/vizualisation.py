import numpy as np
import matplotlib.pyplot as plt
from box_creation import draw_bounding_boxes_on_image_array,\
        draw_bounding_box_on_image, draw_bounding_boxes_on_image


def display_digits_with_boxes(digits, predictions,
                              labels, pred_bboxes,
                              bboxes, iou, title, number,
                              iou_threshold=0.6):
    """
    this function will visualize the images with their bounding boxes.
    digits: image
    predictions: prediction of the image
    labels: actual label of the image
    pred_bboxes: prediction of the box in the image
    bboxes: actual box in the image
    iou: intersection over union if the crieteria is
    filled then box will be plotted
    title: actual title of whole image
    number: how much image needs to be selected
    from the data. It will create a subplot with all images
    """
    indexes = np.random.choice(len(predictions), size=number)
    n_digits = digits[indexes]
    n_labels = labels[indexes]
    n_predictions = predictions[indexes]

    n_iou = []
    if len(iou) > 0:
        n_iou = iou[indexes]

    if len(bboxes) > 0:
        n_boxes = bboxes[indexes, :]

    if len(pred_bboxes) > 0:
        n_pred_bboxes = pred_bboxes[indexes, :]
    # image resiing is done first
    n_digits = n_digits*255.0
    n_digits = n_digits.reshape(number, 75, 75)
    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(1, number):
        ax = fig.add_subplot(1, number, i)
        bboxes_to_plot = []
        if len(pred_bboxes) > i:
            bboxes_to_plot.append(n_pred_bboxes[i])
        if len(bboxes) > i:
            bboxes_to_plot.append(n_boxes[i])
        image_to_draw = draw_bounding_boxes_on_image_array(
            image=n_digits[i], boxes=np.asarray(bboxes_to_plot),
            color=['red', 'green'], display_str_list=['true', 'pred'])
        plt.xlabel(n_predictions[i])
        plt.xticks([])
        plt.yticks([])
        if n_predictions[i] != n_labels[i]:
            ax.xaxis.label.set_color('red')

        plt.imshow(image_to_draw)
        if len(iou) > i:
            color = "black"
            if (n_iou[i][0] < iou_threshold):
                color = "red"
            ax.text(0.2, -0.3, "iou: %s" % (n_iou[i][0]),
                    color=color, transform=ax.transAxes)


def plt_metrics(metric_name, title, history, ylim=2):
    """
    plot desired metric from history
    """
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name],
             color='green', label='val_'+metric_name)
    plt.legend()


def intersection_over_union(pred_box, true_box):
    """
    metrics for bounding box accuracy

    """
    smooting_factor = 1e-10
    x_min_pred, y_min_pred, x_max_pred, y_max_pred = np.split(
                                                     pred_box, 4, axis=1)
    x_min, y_min, x_max, y_max = np.split(true_box, 4, axis=1)
    
    overlap_x =  np.maximum(np.minimum(x_max, x_max_pred), 0)-np.maximum(np.maximum(x_min, x_min_pred), 0)
    overlap_y = np.maximum(np.minimum(y_max, y_max_pred), 0)-np.maximum(np.maximum(y_min, y_min_pred), 0)
    overlap_area = overlap_x * overlap_y
    true_box_area = (x_max-x_min)*(y_max-y_min)
    pred_box_area = (x_max_pred-x_min_pred)*(y_max_pred-y_min_pred)
    union_area = (true_box_area+pred_box_area)-overlap_area

    iou = (overlap_area+smooting_factor)/(union_area+smooting_factor)
    return iou
