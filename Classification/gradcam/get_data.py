import tensorflow_datasets as tfds


def cat_dog_data():
    """
    getting data from tf tensorflow_datasets
    """
    # tfds.disable_progress_bar()

    splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']
    splits, info = tfds.load('cats_vs_dogs',
                             with_info=True,
                             as_supervised=True,
                             split=splits)
    return splits, info


def getting_one_test_image(test_dataset):
    """
    getting sample image and label
    from test_dataset
    """
    for image, label in test_dataset.take(1):
        sample_image = image[0]
        sample_label = label[0]
    return sample_image, sample_label
