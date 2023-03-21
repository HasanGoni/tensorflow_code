import tensorflow as tf
from functools import partial
from configuration_file import num_classes
from configuration_file import vgg_weight_path


def block(
        x, n_convs, filters,
        kernel_size,
        activation,
        pool_size, pool_stride, name):

    """
    create custom blog for vgg
    Args:
    x (tensor) -- input image
    n_convs (int) -- number of convolution layers to append
    filters (int) -- number of filters for the convolution layers
    activation (string or object) -- activation to use in the convolution
    pool_size (int) -- size of the pooling layer
    pool_stried (int) -- stride of the pooling layer
    block_name (string) -- name of the block
    """
    for i in range(n_convs):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same', activation=activation,
            name=f'{name}_conv{i}'
        )(x)
    x = tf.keras.layers.MaxPooling2D(
            pool_size=pool_size,
            strides=pool_stride,
            name=f'{name}_pool{i}')(x)
    return x


block_shortcut = partial(
    block, kernel_size=(3, 3),
    pool_size=(2, 2),
    pool_stride=(2, 2),
    activation='relu')


def VGG_16(input_image, vgg_weight_path):
    """
    This will create vgg model
    without weight, after that loads will
    be loaded from the desired path
    Returns:
    tuple of tensors - output of all encoder blocks plus the final convolution
    layers
    """
    x = block_shortcut(
            input_image, n_convs=2,
            filters=64,
            name='block1')
    p1 = x
    x = block_shortcut(
           x, n_convs=2,
           filters=128,
           name='block2')
    p2 = x
    x = block_shortcut(
            x, n_convs=3,
            filters=256,
            name='block3')
    p3 = x
    x = block_shortcut(
           x, n_convs=3,
           filters=512,
           name='block4')
    p4 = x
    x = block_shortcut(
           x, n_convs=3,
           filters=512,
           name='block5')
    p5 = x

    # creating model
    vgg = tf.keras.Model(
            inputs=input_image,
            outputs=p5)
    vgg.load_weights(
           vgg_weight_path)

    # This vgg is our encoder, now
    # we will process outer layer so that
    # decoder can work on it
    # number of filters for the output convolutional layers
    n = 4096

    # our input images are 224x224 pixels so they will be
    # downsampled to 7x7 after the pooling layers above.
    # we can extract more features by chaining two more
    # convolution layers.
    c6 = tf.keras.layers.Conv2D(
           n, (7, 7), padding='same',
           activation='relu',
           name='conv6')(p5)
    c7 = tf.keras.layers.Conv2D(
           n, (1, 1), padding='same',
           activation='relu',
           name='conv7')(c6)
    return (p1, p2, p3, p4, c7)


def fcn_8_decoder(convs, n_classes):
    """
    Defines the FCN 8 decoder.

    Args:
    convs (tuple of tensors) - output of the encoder network
    n_classes (int) - number of classes

    Returns:
    tensor with shape (height, width, n_classes) containing class probabilities
    """
    # unpack the output of the encoder
    f1, f2, f3, f4, f5 = convs
    # upsample the output of the encoder then crop extra pixels
    # that were introduced
    o1 = tf.keras.layers.Conv2DTranspose(
               n_classes, kernel_size=(4, 4),
               strides=(2, 2),
               use_bias=False)(f5)
    o1 = tf.keras.layers.Cropping2D(
               cropping=(1, 1))(o1)
    # load the pool 4 prediction and do a
    # 1x1 convolution to reshape it to the same shape of `o1` above
    o2 = f4
    o2 = tf.keras.layers.Conv2D(
               n_classes,
               (1, 1),
               activation='relu',
               padding='same')(f4)
    # add the results of the upsampling and pool 4 prediction
    o1 = tf.keras.layers.Add()([o1, o2])

    # Now upsampling the resulting tensor 2 times
    o1 = tf.keras.layers.Conv2DTranspose(
              n_classes, (4, 4),
              strides=(2, 2),
              use_bias=False)(o1)
    o1 = tf.keras.layers.Cropping2D(
             cropping=(1, 1))(o1)

    # load the pool 3 prediction and do a 1x1 convolution to reshape it to the
    # same shape of `o1` above
    o2 = f3
    o2 = tf.keras.layers.Conv2D(
             n_classes, (1, 1),
             padding='same',
             activation='relu'
             )(o2)
    # add the results of the upsampling and pool 3 prediction
    o1 = tf.keras.layers.Add()([o1, o2])
    # upsample up to the size of the original image
    o1 = tf.keras.layers.Conv2DTranspose(
            num_classes,
            (8, 8),
            strides=(8, 8),
            use_bias=False)(o1)
    # append a softmax to get the class probabilities
    o1 = tf.keras.layers.Activation(
          'softmax')(o1)

    return o1


def segmentation_model():
    """
    Model will be created from encoder and decoder
    """
    inputs = tf.keras.layers.Input(
           shape=(224, 224, 3,))
    convs = VGG_16(
            input_image=inputs,
            vgg_weight_path=vgg_weight_path)
    outputs = fcn_8_decoder(
             convs,
             num_classes)
    model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs)
    return model
