import tensorflow as tf


def vgg_model(layer_names):
    """
    Download vgg model and will set
    the first layers will be frozen
    there will be another model where input
    will be normal input but output will
    be the output from the named layer_names
    Args:
    layer_names: a list of strings,
    representing the names of the desired content and style layers

    Returns:
    A model that takes the regular vgg19 input and outputs just the content and
    style layers.
    """
    vgg = tf.keras.applications.vgg19.VGG19(
                                            include_top=False,
                                            weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(layer).output for layer in layer_names]
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    return model
