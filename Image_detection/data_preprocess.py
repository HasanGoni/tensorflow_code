import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds


def read_image_tfds(image, label):
    """
    creating bounding box in a image
    """
    xmin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    ymin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    image = tf.reshape(image, (28, 28, 1,))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
    # here 75 is used to describe maximum height and width of
    # the image
    image = tf.cast(image, tf.float32)/255.0
    xmin = tf.cast(xmin, tf.float32)
    ymin = tf.cast(ymin, tf.float32)
     
    xmax = (xmin + 28) / 75
    ymax = (ymin + 28) / 75
    xmin = xmin / 75
    ymin = ymin / 75
    return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])


def get_training_dataset():
    """
    dataset downloading
    """
    dataset = tfds.load('mnist', split='train', as_supervised=True)
    dataset = dataset.map(read_image_tfds)
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    dataset = dataset.repeat() #for keras mandatory right now
    dataset = dataset.batch(64)
    return dataset


def get_validation_dataset():
    """
    validation datas downloading
    """
    dataset = tfds.load('mnist', split='test', as_supervised=True)
    dataset = dataset.map(read_image_tfds)
    dataset = dataset.batch(10000)
    return dataset


def dataset_to_numpy(train_dataset, val_dataset, n):
    """
    select number of images:n from both train_dataset
    and val_dataset and convert them to numpy 
    """
    train_bs_ds = train_dataset.unbatch().batch(n)
    for val_dig, (val_label, val_bbox) in val_dataset:
        val_dig = val_dig.numpy()
        val_label = val_label.numpy()
        val_bbox = val_bbox.numpy()
        break
    for train_dig, (train_lbels, train_bboxes) in train_bs_ds:
        train_dig = train_dig.numpy()
        train_lbels = train_lbels.numpy()
        train_bboxes = train_bboxes.numpy()
        break

    #  labels are one hot encoded so 
    val_label = np.argmax(val_label, axis=1)
    train_label = np.argmax(train_lbels, axis=1)
    return (train_dig, train_label, train_bboxes,
            val_dig, val_label, val_bbox)

