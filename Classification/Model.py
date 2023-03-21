import tensorflow as tf
from configuration_file import input_size, transfer_learning_input_size,\
        output_category_number, epochs, batch_size


def feature_ext(inputs):
    """
    Extract features from images noramally transfer learning part
    """
    feature_extractor = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=transfer_learning_input_size)(inputs)
    return feature_extractor


def classifier(inputs):
    """"
    classifier part after feature_extractor classification will happen
    """
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(output_category_number, activation='softmax')(x)
    return x


def final_model(inputs):
    """
    feature_extractor and classifier together with upsampling
    """
    resize_factor = transfer_learning_input_size[0]//input_size[0]
    resize = tf.keras.layers.UpSampling2D(size=(resize_factor, resize_factor))(inputs)
    resnet_extractor = feature_ext(resize)
    classifiction_output = classifier(resnet_extractor)
    return classifiction_output


def define_compile_model():
    inputs = tf.keras.layers.Input(shape=input_size)
    classification_out = final_model(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=classification_out)
    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
