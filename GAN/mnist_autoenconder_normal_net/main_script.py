import tensorflow as tf
import tensorflow_datasets as tfds
from data_preprocess import map_image
from model_creation import simple_autoencoder
batch_size = bs = 128
train_steps = 60_000 // bs
# Download and prepare dataset
train_ds = tfds.load('mnist',
                     as_supervised=True,
                     split="train")
# Converting the dataset with function
train_ds = train_ds.map(map_image)
train_ds = train_ds.shuffle(buffer_size=1024).batch(bs).repeat()

test_ds = tfds.load('mnist',
                    as_supervised=True,
                    split='test')
test_ds = test_ds.map(map_image)
test_ds = test_ds.batch(bs).repeat()
# Model Creation
encoder, autoencoder = simple_autoencoder()
# Compiling and training the model
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='binary_crossentropy')
history = autoencoder.fit(
            train_ds,
            steps_per_epoch=train_steps,
            epochs=50)
