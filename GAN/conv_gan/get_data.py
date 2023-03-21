import keras
import tensorflow as tf
import numpy as np


def keras_data():
    """
    getting only training data for gan picture
    generation, no label is available and also
    no test data is available
    after loading data it will be preprocessed
    Normalize :
    """
    (X_train, _), _ = keras.datasets.fashion_mnist.load_data()
    X_train = X_train.astype(np.float32)/255.0

    # But don't understatnd why *2 and -1
    # the shape seems to be same
    # Reshape and rescale
    X_train = X_train.reshape(-1, 28, 28, 1)*2.-1.
    # X_train = tf.data.Dataset.shuffle(
    return X_train


def prepare_data(X_train,
                 bs=128):
    """
    Converting tensorflow datasets
    and batchig them
    """
    ds = tf.data.Dataset.from_tensor_slices(X_train)
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size=bs,
                  drop_remainder=True).prefetch(1)
    return ds
