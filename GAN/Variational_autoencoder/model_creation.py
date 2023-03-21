import tensorflow as tf


def enconder_layers(inputs, latent_dim):
    """
    """
    conv1 = tf.keras.layers.Conv2D(filters=32,
                                   strides=2,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu',
                                   name='encoder_conv1')(inputs)
    batch_1 = tf.keras.layers.BatchNormalization()(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=2,
                                   padding='same',
                                   activation='relu',
                                   name='encoder_conv2')(batch_1)
    batch_2 = tf.keras.layers.BatchNormalization()(conv2)
    x = tf.keras.layers.Flatten(name='encoder_flatten')(batch_2)
    x = tf.keras.layers.Dense(units=20,
                              activation='relu',
                              name='encoded_dense')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    mu = tf.keras.layers.Dense(units=latent_dim, name='latent_mu')(x)
    sigma = tf.keras.layers.Dense(units=latent_dim, name='latent_sigma')(x)

    return mu, sigma, batch_2.shape


class Sampling(tf.keras.layers.Layer):
    def call(self,
             inputs):
        """
        Genertes random sample and combines with
        enconder output
        """
        mu, sigma = inputs
        batch, dim = tf.shape(mu)[0], tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * sigma) * epsilon


def encoder_model(latent_dim,
                  input_shape):
    """
    """
    inp = tf.keras.layers.Input(shape=input_shape)
    mu, sigma, conv_shape = enconder_layers(inputs=inp,
                                            latent_dim=latent_dim)
    z = Sampling()((mu, sigma))
    model = tf.keras.Model(inputs=inp,
                           outputs=[mu, sigma, z])
    return model, conv_shape


def decoder_layers(inputs,
                   conv_shape):
    """
    """
    units = conv_shape[1] * conv_shape[2] * conv_shape[3]
    x = tf.keras.layers.Dense(units=units, activation='relu',
                              name='decoder_dense1')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((conv_shape[1],
                                 conv_shape[2],
                                 conv_shape[3]))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        activation='relu',
                                        name='decoder_conv1')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32,
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        activation='relu',
                                        name='decoder_conv2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=1,
                                        kernel_size=3,
                                        strides=1,
                                        padding='same',
                                        activation='sigmoid',
                                        name='decoder_final')(x)
    return x


def decoder_model(latent_dim, conv_shape):
    """
    """
    inpu = tf.keras.layers.Input(shape=(latent_dim,))
    outputs = decoder_layers(inpu, conv_shape)
    model = tf.keras.Model(inputs=inpu,
                           outputs=outputs)
    return model


def kl_reconstruction_loss(mu,
                           sigma):
    """
    """
    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    kl_loss = tf.reduce_mean(kl_loss) * - 0.5
    return kl_loss


def vae_model(encoder,
              decoder,
              input_shape):
    """
    """
    inputs = tf.keras.Input(shape=input_shape)
    mu, sigma, z = encoder(inputs)
    reconstracted = decoder(z)
    model = tf.keras.Model(inputs=inputs,
                           outputs=reconstracted)
    loss = kl_reconstruction_loss(mu,
                                  sigma)
    model.add_loss(loss)
    return model


def get_models(input_shape,
               latent_dim):
    """
    """
    encoder, conv_shape = encoder_model(latent_dim=latent_dim,
                                        input_shape=input_shape)
    print(encoder.summary())
    decoder = decoder_model(latent_dim=latent_dim,
                            conv_shape=conv_shape)
    print(decoder.summary())
    vae = vae_model(encoder,
                    decoder,
                    input_shape)
    return encoder, decoder, vae
