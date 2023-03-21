import tensorflow as tf
from data_preprocess import preprocess_image_for_vgg


def gram_matrix_cal(layer_output):
    """
    Calcualte the scaled gram matrix
    of a layers output.
    Arg --> layer output shape(batch, height,
                               width)
    Return --> Scaler gram matrix devided
    by number of location(= height * width)
    """
    gram = tf.linalg.einsum('bijc, bijd->bcd',
                                layer_output,
                                layer_output)
    im_shape = tf.shape(layer_output)
    num_location =tf.cast((im_shape)[1] * (im_shape)[2], tf.float32)
    return gram / num_location


def get_style_layers_feature(image,
                             vgg,
                             num_style_layer):
    """
    getting gram matrix for all the
    style layers
    Args:
    image --> will be an input image
    vgg --> a vgg model which has 6
    output layers and first layers
    will be style layers.
    num_style_layer --> number of style
    laysers

    return --> gram style feature
    """
    preprocessed_image = preprocess_image_for_vgg(image)
    outputs = vgg(preprocessed_image)
    outputs = outputs[:num_style_layer]
    style_gram_matrix = [gram_matrix_cal(i) for i in outputs]
    return style_gram_matrix


def get_content_layers_feature(image,
                             vgg,
                             num_content_layer):
    """
    getting gram matrix for all the
    style layers
    Args:
    image --> will be an input image
    vgg --> a vgg model which has 6
    output layers and first layers
    will be style layers.
    num_content_layer --> number of style
    laysers

    return --> gram style feature
    """
    preprocessed_image = preprocess_image_for_vgg(image) 
    outputs = vgg(preprocessed_image)
    outputs = outputs[num_content_layer:]
    return outputs
