import keras
import numpy as np


(X_train, _), _ = keras.datasets.mnist.load_data()

X_train = X_train.astype(np.float32) / 255
