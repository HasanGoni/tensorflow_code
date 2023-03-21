import tensorflow as tf
from viz_utils import plot_multiple_images
import matplotlib.pyplot as plt


def generator_model(random_dim):
    """
    Creating a generator where
    input shape will be random_dim
    output will be a 28,28
    """
    inputs = tf.keras.layers.Input(
               shape=(random_dim,))
    x = tf.keras.layers.Dense(units=64,
                              activation='selu')(inputs)
    x = tf.keras.layers.Dense(units=128,
                              activation='selu')(x)
    x = tf.keras.layers.Dense(units=(28*28), activation='sigmoid')(x)
    x = tf.keras.layers.Reshape((28, 28))(x)
    model = tf.keras.Model(inputs=inputs,
                           outputs=x)
    return model


def discriminator_model(input):
    """
    Create discriminator
    """
    inputs = tf.keras.layers.Input(
                shape=([input, input]))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(units=128,
                              activation='selu')(x)
    x = tf.keras.layers.Dense(units=64,
                              activation='selu')(x)
    x = tf.keras.layers.Dense(units=1,
                              activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs,
                           outputs=x)
    return model


def gan_model(generator,
              discriminator):
    model = tf.keras.models.Sequential([generator,
                                       discriminator])
    return model


def gan_trining(gan,
                train_dataset,
                epochs,
                image_decode_size=16):
    """
    taining loop of gan
    gan: gan model
    train_dataset: datset for training
    epochs: how much epochs to run
    image_deocde_size: size of the generated image
    """
    generator, discriminator = gan.layers
    for epoch in range(epochs):
        print(f'Epoch {epoch}/ {epochs}')
        for train_image_batch in train_dataset:
            batch_size = train_image_batch.shape[0]

            noise = tf.random.normal(shape=[batch_size,
                                            image_decode_size])
            fake_images = generator(noise)
            mixed_images = tf.concat([fake_images,
                                     train_image_batch],
                                     axis=0)
            discriminator_label = tf.constant(
                                    [[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(mixed_images,
                                         discriminator_label)

            noise = tf.random.normal(shape=[batch_size,
                                            image_decode_size])
            generator_label = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise,
                               generator_label)
        print(end='.')
    plot_multiple_images(fake_images,
                         n_cols=16)
    plt.show()
