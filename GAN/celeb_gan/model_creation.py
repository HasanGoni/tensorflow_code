import tensorflow as tf


def get_initializer():
    """
    Creating function for kernel
    initialiers
    one for convolutional and the
    other for batch normalization
    gamma initializer
    """
    return (tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            # Conv initializer
            tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02))
    # bn_gama initializer
    

def _get_norm_layer(norm):
    """
    Chossing which normalization to use
    """
    if norm == "NA":
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return tf.keras.layers.BatchNormalization
    elif norm == 'instance_normilization':
        return tf.keras.layers.InstanceNormalization
    elif norm == 'layer_normlization':
        return tf.keras.layers.LayerNormalization


def create_generator(input_shape=(1, 1, 128),
                     output_channels=3,
                     dim=64,
                     n_upsamplings=4,
                     norm='batch_norm',
                     name='generator'):
    """
    Creating a generator
    """
    Normalization = _get_norm_layer(norm)
    conv_initailizer, bn_gamma_initializer = get_initializer()

    x = inputs = tf.keras.layers.Input(
                 shape=input_shape)

    # 1: 1x1 --> 4x4
    dimension = min(dim * 8, (dim * 2 ** (n_upsamplings - 1)))

    x = tf.keras.layers.Conv2DTranspose(
        filters=dimension,
        kernel_size=4,
        strides=1,
        padding='valid',
        use_bias=False,
        )(x)
    x = Normalization(
        # gamma_initializer=bn_gamma_initializer
       )(x)
    x = tf.keras.layers.ReLU()(x)
    # 2: 4x4 --> 8X8 --> 16x16
    for i in range(n_upsamplings - 1):
        dimension = min(dim * 2 ** (n_upsamplings - 2 - i), dim*8)
        x = tf.keras.layers.Conv2DTranspose(
            filters=dimension,
            strides=2,
            kernel_size=4,
            # kernel_initializer=conv_initializer,
            padding='same',
            use_bias=False)(x)
        x = Normalization(
            # gamma_initializer =  bn_gamma_initializer
            )(x)
        x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(
            filters=output_channels,
            kernel_size=4,
            strides=2,
            padding='same'
            # kernel_initilizer=conv_initializer
            )(x)
    outputs = tf.keras.layers.Activation('tanh')(x)
    model = tf.keras.Model(inputs=inputs,
                           outputs=outputs,
                           name=name)
    return model



def create_discriminator(input_shape=(64, 64, 3),
                         dim=64,
                         n_downsampling=4,
                         norm='batch_norm',
                         name='discriminator'):
    """
    Creating a discriminator
    """
    Normalization = _get_norm_layer(norm)
    conv_initailizer, bn_gamma_initializer = get_initializer()

    x = inputs = tf.keras.layers.Input(
                 shape=input_shape)

    # 1: 16x16 --> 8x8 --> 4x4

    x = tf.keras.layers.Conv2D(
        filters=dim,
        kernel_size=4,
        strides=2,
        padding='same',
        )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    # 2: 4x4 --> 8X8 --> 16x16
    for i in range(n_downsampling - 1):
        dimension = min(dim * 2 ** (i + 1), dim*8)
        x = tf.keras.layers.Conv2D(
            filters=dimension,
            strides=2,
            kernel_size=4,
            # kernel_initializer=conv_initializer,
            padding='same',
            use_bias=False)(x)
        x = Normalization(
            # gamma_initializer =  bn_gamma_initializer
            )(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    outputs = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=4,
            strides=1,
            padding='valid'
            # kernel_initilizer=conv_initializer
            )(x)
    model = tf.keras.Model(inputs=inputs,
                           outputs=outputs)
    return model
