import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    def call(self,
             inputs):
        """
        sampling normal distribution
        """
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(
                                            batch,
                                            dim))
        z = mu + tf.exp(0.5 * sigma) * epsilon
        return z


def encoder_layers(inputs,
                   latent_dim):
    """
    Create different layers for encoder
    """
    x = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               activation='relu',
                               name='encoder_conv1')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               activation='relu',
                               name='encoder_conv2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               activation='relu',
                               name='encoder_conv3')(x)
    batch_3 = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(batch_3)
    x = tf.keras.layers.Dense(units=120, activation='relu',
                              name='encoded_dense')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    mu = tf.keras.layers.Dense(latent_dim, name='latent_mu')(x)
    sigma = tf.keras.layers.Dense(latent_dim, name='latent_sigma')(x)

    return mu, sigma, batch_3.shape


def encoder_model(latent_dim,
                  input_shape):
    """
    Creating model from decoder layers
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    mu, sigma, conv_shape = encoder_layers(inputs,
                                           latent_dim)
    z = Sampling()((mu, sigma))
    model = tf.keras.Model(inputs=inputs, outputs=[mu, sigma, z])
    return model, conv_shape


def decoder_layers(inputs, conv_shape):
    """
    Creating decoder layer
    """
    units = conv_shape[1] * conv_shape[2] * conv_shape[3]
    x = tf.keras.layers.Dense(units=units, activation='relu',
                              name='flat_decoder_1')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((conv_shape[1], conv_shape[2],
                                 conv_shape[3]))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=128,
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        activation='relu',
                                        name='decoder_conv1')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        activation='relu',
                                        name='decoder_conv2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32,
                                        kernel_size=3,
                                        strides=2,
                                        padding='same',
                                        activation='relu',
                                        name='decoder_conv3')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=3,
                                        kernel_size=3,
                                        strides=1,
                                        padding='same',
                                        activation='sigmoid',
                                        name='decoder_final')(x)

    return x


def decoder_model(latent_dim,
                  conv_shape):
    """
    """
    inputs = tf.keras.layers.Input(shape=(latent_dim,))
    outputs = decoder_layers(inputs,
                             conv_shape)
    model = tf.keras.Model(inputs=inputs,
                           outputs=outputs)
    return model


def kl_reconstruction_loss(inputs,
                           outputs,
                           mu,
                           sigma):
    """
    creation Kullback-liebher Divergence loss
    """
    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    kl_loss = tf.reduce_mean(kl_loss) * -0.5
    return kl_loss


def vae_model(encoder,
              decoder,
              input_shape):
    """
    encoder --> encoder model
    decoder --> decoder model
    input_shape --> shape of the input
    return  vae model
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Now using encoder model and
    # Getting outputs from the encoder model
    mu, sigma, z = encoder(inputs)
    # output from encoder model will
    # be used for decoder model
    reconstructed = decoder(z)
    model = tf.keras.Model(inputs=inputs,
                           outputs=reconstructed)
    loss = kl_reconstruction_loss(inputs,
                                  z,
                                  mu, sigma)
    model.add_loss(loss)
    return model


def getting_model(input_shape,
                  latent_dim):
    """
    getting all the models
    from input shape and latent dim

    return encoder model, decoder_model
    and vae model
    """
    encoder, conv_shape = encoder_model(latent_dim,
                                        input_shape)
    decoder = decoder_model(latent_dim,
                            conv_shape)
    vae = vae_model(encoder,
                    decoder,
                    input_shape)
    return encoder, decoder, vae
