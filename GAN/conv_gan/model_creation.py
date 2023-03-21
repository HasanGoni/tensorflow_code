import tensorflow as tf
from viz_utils import plot_images
import matplotlib.pyplot as plt


def generator_creation(code_dim=32):
    """
    """
    inputs = tf.keras.layers.Input(shape=code_dim,)
    x = tf.keras.layers.Dense(
         7*7*128)(inputs)
    # why batch size is first in dimesion
    # don't know
    x = tf.keras.layers.Reshape([7, 7, 128])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(
        filters=64,
        kernel_size=5,
        strides=2,
        padding='SAME',
        activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=5,
        strides=2,
        padding='SAME',
        activation='tanh')(x)
    model = tf.keras.Model(inputs=inputs,
                           outputs=x)
    return model


def discriminator_model():
    """
    """
    inputs = tf.keras.layers.Input(shape=[28, 28, 1])
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=5,
        strides=2,
        padding='SAME',
        activation=tf.keras.layers.LeakyReLU(alpha=0.2))(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               strides=2,
                               padding='SAME',
                               activation=tf.keras.layers.LeakyReLU(0.2))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    models = tf.keras.Model(inputs=inputs,
                            outputs=x)
    return models


def GAN_(generator,
         discriminator):
    """
    """
    model = tf.keras.Sequential([generator,
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
    plot_images(fake_images, 16)
    plt.show()
