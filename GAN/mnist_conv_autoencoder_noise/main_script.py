import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from preprocess_data import map_image
from viz_utils import show_batch
import matplotlib.pyplot as plt
from model_creation import autoencoder_conv
from viz_utils import display_results
batch_size = bs = 128
train_dataset = tfds.load(
                          'fashion_mnist', as_supervised=True,
                          split='train')
test_dataset = tfds.load(
                         'fashion_mnist',
                         as_supervised=True,
                         split='test')
train_dataset = train_dataset.map(map_image,
                                  tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(
                    buffer_size=1024).batch(bs).repeat()

test_dataset = test_dataset.map(map_image,
                                tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(bs).repeat()

# showing a batch of the image
# show_batch(train_dataset)
# plt.show()

encoder, autoencoder = autoencoder_conv()
# print(autoencoder.summary())
train_steps = 60_000 // bs
test_steps = 60_000 // bs

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='binary_crossentropy')
history = autoencoder.fit(train_dataset,
                          steps_per_epoch=train_steps,
                          validation_data=test_dataset,
                          validation_steps=test_steps,
                          epochs=40)

# take 1 batch of the dataset
test_dataset = test_dataset.take(1)

# take the input images and put them in a list
output_samples = []
for input_image, image in tfds.as_numpy(test_dataset):
    output_samples = input_image

# pick 10 indices
idxs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# prepare test samples as a batch of 10 images
conv_output_samples = np.array(output_samples[idxs])
conv_output_samples = np.reshape(conv_output_samples, (10, 28, 28, 1))

# get the encoder ouput
encoded = encoder.predict(conv_output_samples)
predicted = autoencoder.predict(conv_output_samples)

# get a prediction for some values in the datasetodicted
# = autoencoder.predict(conv_output_samples)
display_results(conv_output_samples, encoded, predicted, enc_shape=(7, 7))
plt.show()
