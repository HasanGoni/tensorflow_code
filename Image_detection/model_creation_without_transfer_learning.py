import tensorflow as tf


def feature_extractor(inputs):
    """
    extracting features from inputs
    and the put them in front of classifier
    """
    x = tf.keras.layers.Conv2D(16, 3, activation='relu',
                               input_shape=(75, 75, 1))(inputs)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    return x


def dense_layers(inputs):
    """
    flattened the data after feature feature_extractor
    """
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    return x


def classification_layer(inputs):
    """
    after feature extraction and dense
    layer last classification_layer will give
    the output
    """
    x = tf.keras.layers.Dense(10,
                              activation='softmax', name='classifier')(inputs)
    return x


def building_box_regression(inputs):
    """
    building box regression model
    """
    x = tf.keras.layers.Dense(4, name='bounding_box')(inputs)
    return x


def final_model(inputs):
    feature_ext = feature_extractor(inputs)
    dense_l = dense_layers(feature_ext)
    classifier_output = classification_layer(dense_l)
    bbox_out = building_box_regression(dense_l)
    model = tf.keras.Model(inputs=inputs,
                           outputs=[classifier_output, bbox_out])
    return model


def define_compile_model(inputs):
    """
    inputs will be given and then model
    cration and compilation will be happen
    """
    model = final_model(inputs)
    model.compile(optimizer='adam',
                  loss={'classifier': 'categorical_crossentropy',
                        'bounding_box': 'mse'},
                  metrics={'classifier': 'accuracy',
                           'bounding_box': 'mse'})
    return model
