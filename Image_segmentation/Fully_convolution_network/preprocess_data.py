import tensorflow as tf
from pathlib import Path
from configuration_file import class_names
from configuration_file import BATCH_SIZE
Path.ls = lambda x: list(x.iterdir())


def get_dataset_slice_path(
     image_dir, label_dir):
    """
    generates the lists of image and label map paths
    Args:
    image_dir (string) -- path to the input images directory
    label_map_dir (string) -- path to the label map directory
    Returns:
    image_paths (list of strings) -- paths to each image file
    label_dir (list of strings) -- paths to each label
    map_filename_to_image_and_mask
    """
    image_paths = [i.as_posix() for i in Path(image_dir).ls()]
    label_map_paths = [
        i.as_posix() for i in Path(label_dir).ls()]
    return image_paths, label_map_paths


def map_filename_to_image_and_mask(
    t_filename,
    a_filename,
    classes=class_names,
    height=224,
    width=224
     ):
    """
    Preprocesses the dataset by:
    * resizing the input image and label maps
    * normalizing the input image pixels
    * reshaping the label maps from (height, width, 1)
    to (height, width, 12)
    Args:
    t_filename (string) -- path to the raw input image
    a_filename (string) -- path to the raw annotation
    (label map) file
    height (int) -- height in pixels to resize to
    width (int) -- width in pixels to resize to

    Returns:
    image (tensor) -- preprocessed image
    annotation (tensor) -- preprocessed annotation
    """
    img_raw = tf.io.read_file(t_filename)
    anno_raw = tf.io.read_file(a_filename)

    image = tf.image.decode_jpeg(img_raw)
    annotation = tf.image.decode_jpeg(anno_raw)

    # Resizing image and annotation mask
    image = tf.image.resize(image, (height, width, ))
    annotation = tf.image.resize(annotation, (height, width,))
    # Adding additional batch size
    image = tf.reshape(image, (height, width, 3, ))

    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(
        annotation, (height, width, 1, ))

    # Normally last channel of an image is color
    # In previous line another axis is added
    # in annotation. Now in the loop, this
    # axis is filling with true value for
    # subsequent class. so like conversion
    # of one hot encoding we need to search
    # maximum value of last channel
    # and if we want colors we need implement
    # color for each channel
    stack_list = []
    for i in range(len(classes)):
        mask = tf.equal(annotation[:, :, 0], tf.constant(i))
        stack_list.append(mask)

    annotation = tf.stack(stack_list, axis=2)
    # Normalize pixels in the input image
    image = image/127.5
    image -= 1
    return image, annotation


def get_dataset(image_paths, label_map_paths, name='train'):
    """
    prepare dataset for training
    and validation
    """
    _dataset = tf.data.Dataset.from_tensor_slices(
            (image_paths,
             label_map_paths))
    _dataset = _dataset.map(map_filename_to_image_and_mask)
    if name == 'train':
        train_dataset = _dataset.shuffle(
            100, reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(BATCH_SIZE)
        train_dataset = train_dataset.repeat()
        final_dataset = train_dataset.prefetch(-1)
    else:
        validation_dataset = _dataset.batch(BATCH_SIZE)
        final_dataset = validation_dataset.repeat()
    return final_dataset
