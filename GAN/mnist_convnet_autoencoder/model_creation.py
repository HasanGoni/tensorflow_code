import tensorflow as tf


def encoder(inputs):
    """
    """
    conv1 = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='relu')(inputs)
    maxpool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='relu')(maxpool_1)
    maxpool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    return maxpool_2


def bottle_neck(inputs):
    """
    """
    bottle_neck_layer = tf.keras.layers.Conv2D(filters=256,
                                               kernel_size=(3, 3),
                                               padding='same',
                                               activation='relu')(inputs)
    encoder_vizualization = tf.keras.layers.Conv2D(filters=1,
                                                   kernel_size=(3, 3),
                                                   activation='sigmoid',
                                                   padding='same')(bottle_neck_layer)
    return bottle_neck_layer, encoder_vizualization


def decoder(inputs):
    """
    """
    conv1 = tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same')(inputs)
    upsample_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='relu')(upsample_1)
    upsample_2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(filters=1,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='sigmoid')(upsample_2)
    return conv3


def autoencoder_conv():
    """
    """
    inp = tf.keras.layers.Input(shape=(28, 28, 1))
    encoder_output = encoder(inp)
    bottle_neck_layer, encoder_viz = bottle_neck(encoder_output)
    decoder_output = decoder(encoder_viz)
    model = tf.keras.Model(inputs=inp,
                           outputs=decoder_output)
    encoder_model = tf.keras.Model(inputs=inp,
                                   outputs=encoder_viz)
    return encoder_model, model
