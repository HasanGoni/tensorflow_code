import tensorflow as tf
import tensorflow_datasets as tfds


dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
print(dataset.keys())
print(info)
