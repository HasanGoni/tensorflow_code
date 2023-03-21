import tensorflow as tf
import numpy as np
import random
from prepare_data import get_dataset
from prepare_data import map_image
from viz_utils import show_batches
from model_creation import getting_model
from viz_utils import generate_and_save_images
from IPython import display

np.random.seed(51)
batch_size = bs = 2000
latent_dim = 512

image_size = (64, 64, 3,)
paths = get_dataset(r'/tmp/anime/images')
# shuffle paths
random.shuffle(paths)

# split path list to get validation data
path_len = len(paths)
train_path_len = int(path_len * 0.8)

train_paths = paths[:train_path_len]
val_paths = paths[train_path_len:]

# Creating trianing dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths))
train_dataset = train_dataset.map(map_image,
                                  tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(bs)

# Creating validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices((val_paths))
validation_dataset = validation_dataset.map(map_image,
                                            tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.batch(bs)

# show_batches(train_dataset)


# creating encoder, decoder and vae model
encoder, decoder, vae = getting_model(image_size,
                                      latent_dim)

# Defining optimizer and loss of the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
loss_metric = tf.keras.metrics.Mean()
mse_loss = tf.keras.losses.MeanSquaredError()
# bce_loss = tf.keras.losses.BinaryCrossentropy()

# Training loop
random_vector = tf.random.normal(shape=[16, latent_dim])
epochs = 100

generate_and_save_images(decoder,
                         0,
                         0,
                         random_vector)
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:

            # feed a batch to the VAE model
            reconstructed = vae(x_batch_train)

            # compute reconstruction loss
            flattened_inputs = tf.reshape(x_batch_train, shape=[-1])
            flattened_outputs = tf.reshape(reconstructed, shape=[-1])
            loss = mse_loss(flattened_inputs, flattened_outputs) * (64*64*3)

            # add KLD regularization loss
            loss += sum(vae.losses)

        # get the gradients and update the weights
        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        # compute the loss metric
        loss_metric(loss)

        # display outputs every 100 steps
        if step % 10 == 0:
            display.clear_output(wait=False)
            generate_and_save_images(decoder, epoch, step, random_vector)
        print('Epoch: %s step: %s mean loss = %s' % (epoch, step, loss_metric.result().numpy()))
vae.save("anime.h5")
decoder.save('decoder.h5')
