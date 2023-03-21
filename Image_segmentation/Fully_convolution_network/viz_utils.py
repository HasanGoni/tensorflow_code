import numpy as np
import seaborn as sns
import PIL
import matplotlib.pyplot as plt
from configuration_file import num_classes
from configuration_file import class_names


def give_color_to_annotation(annotation, num_classes=num_classes):
    """
    Converts a 2-D annotation to a numpy array with
    shape (height, width, 3) where
    the third axis represents the
    color channel. The label values are multiplied by
    255 and placed in this axis to give color to the annotation

    Args:
    annotation (numpy array) - label map array

    Returns:
    the annotation array with an additional color channel/axis
    """
    colors = sns.color_palette(None, num_classes)
    new_anno = np.zeros(
        (annotation.shape[0],
         annotation.shape[1], 3)).astype('float')
    for i in range(num_classes):
        # selecing which part of the annoation
        # have same value.
        seg_clr = (annotation == i)
        new_anno[:, :, 0] += seg_clr * (colors[i][0] * 255.0)
        new_anno[:, :, 1] += seg_clr * (colors[i][1] * 255.0)
        new_anno[:, :, 2] += seg_clr * (colors[i][2] * 255.0)
    return new_anno


def fuse_with_pil(images: list):
    """
    images is a list of images
    which should be pasted together in one image
    """

    width = np.sum([i.shape[1] for i in images])

    height = np.max([i.shape[0] for i in images])
    new_pil_image = PIL.Image.new('RGB', (width, height))

    wd = 0
    for i in images:
        pil_image = PIL.Image.fromarray(
             np.uint8(i))
        new_pil_image.paste(pil_image, (wd, 0))
        wd += i.shape[1]
    return new_pil_image


def show_annotation_and_image(image, annotation):
    """
    getting only one image and annotation
    from the dataset and convert both
    them to image and put color to the
    annotation: give_color_to_annotation function
    and put together both images in one
    image: fuse_with_pil fuction  and
    display image and annotaion side by
    side
    image: dataset image
    annonatation: dataset annotation
    """
    # getting number from the annotation
    # like from one hot encode to actual
    # number on axis 2
    new_ann = np.argmax(annotation, axis=2)

    # putting color to this annotaion
    new_ann = give_color_to_annotation(new_ann)

    image = image + 1
    image = image * 127.5
    image = np.uint8(image)
    images = [image, new_ann]
    images = [image, new_ann]

    # putting both image and annotation
    # in one image
    fused_image = fuse_with_pil(images)
    plt.imshow(fused_image)


def list_show_annotation(dataset, row_num=3):
    """
    take the dataset and plt image and
    annotaion side by side
    dataset:td.data.Dataset()
    row_num: how much row you want to
    see in the plot
    """
    ds = dataset.unbatch()
    ds = ds.shuffle(buffer_size=100)
    plt.figure(figsize=(25, 15))
    plt.title('Images and annotations')
    plt.subplots_adjust(
        bottom=0.1, top=0.9,
        hspace=0.05)
    for idx, (image, annotation) in enumerate(ds.take(9)):
        plt.subplot(row_num, row_num, idx+1)
        plt.yticks([])
        plt.xticks([])
        show_annotation_and_image(image.numpy(), annotation.numpy())


def show_predictions(
      image, labelmaps,
      titles,
      iou_list,
      dice_score_list,
      ):
    """
    showing labels and images together
    """
    true_image = give_color_to_annotation(labelmaps[1])
    pred_image = give_color_to_annotation(labelmaps[0])
    image = image + 1
    image = image * 127.5
    images = np.uint8([image, pred_image, true_image])

    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list)) if iou > 0.0]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place
    display_string_list = ["{}: IOU: {} Dice Score: {}".format(class_names[idx], iou, dice_score) for idx, iou, dice_score in metrics_by_id]
    display_string = "\n\n".join(display_string_list)

    plt.figure(figsize=(15, 4))

    for idx, im in enumerate(images):
        plt.subplot(1, 3, idx+1)
        if idx == 1:
            plt.xlabel(display_string)
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[idx], fontsize=12)
        plt.imshow(im)
