import tensorflow as tf
from configuration_file import IMAGE_SIZE


def build_model():
    """
    building a model VGG16 model and above it a GlobalAveragePooling2D
    is  appended on top of it. weights are imagenet weights
    last 4 layers are only trainable other are freezed
    """
    base_model = tf.keras.applications.vgg16.VGG16(
                  include_top=False,
                  weights='imagenet',
                  input_shape=IMAGE_SIZE + (3, ))
    output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(2,
                                   activation='softmax')(output)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    return model


def all_layer_model(model, last_conv_layer=18):
    """
    creating a model from a trained model to see the
    activations of those layes.
    first_layer is layer 0 not layer 1 because layer 0
    is input layer
    last_layer is the last convolutional layer not
    actual last layer
    """

    outputs = [layer.output for layer in model.layers[1:last_conv_layer]]
    all_layer_model = tf.keras.models.Model(
                                            model.input,
                                            outputs)
    return all_layer_model
