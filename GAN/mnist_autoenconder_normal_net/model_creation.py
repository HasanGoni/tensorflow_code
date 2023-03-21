import tensorflow as tf


def simple_autoencoder():
    """
    """
    inp = tf.keras.layers.Input(shape=(784,))
    encoder = tf.keras.layers.Dense(units=32,
                                    activation='relu')(inp)
    decoder = tf.keras.layers.Dense(units=784,
                                    activation='sigmoid')(encoder)
    encoder_model = tf.keras.Model(inputs=inp,
                                   outputs=encoder)
    auto_encoder_model = tf.keras.Model(inputs=inp,
                                        outputs=decoder)
    return encoder_model, auto_encoder_model
