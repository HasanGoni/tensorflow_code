import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import cv2


def get_cam(model, preprocessed_image, label_of_image,
            layer_name):
    """
    get class activation map of a desired
    layer of convolutional network
    """
    cam_model = tf.keras.models.Model(
                    inputs=model.inputs,
                    outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output_values, results = cam_model(preprocessed_image)

        tape.watch(conv_output_values)
        # Use binary cross entropy loss
        # actual_label is 0 if cat, 1 if dog
        # get prediction probability of dog
        # If model does well,
        # pred_prob should be close to 0 if
        # cat, close to 1 if dog
        pred_prob = results[:, 1]
        label_of_image = tf.cast(
                          label_of_image,
                          dtype=tf.float32)

        smoothing = 0.00001

        loss = -1 * (label_of_image * tf.math.log(pred_prob + smoothing)) + ((1 - label_of_image) * tf.math.log(1 - pred_prob + smoothing))

        print(f'binary loss {loss}')
    # get the gradient of the loss with respect to the
    # outputs of the outputs of the last conv layer
    grads_values = tape.gradient(loss, conv_output_values)
    grads_values = K.mean(grads_values, axis=(0, 1, 2))
    conv_output_values = np.squeeze(conv_output_values.numpy())
    # weight the convolutiion outputs with the computed gradients
    for i in range(conv_output_values.shape[-1]):
        conv_output_values[:, :, i] *= grads_values[i]

    heatmap = np.mean(conv_output_values, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    del cam_model, conv_output_values, grads_values, loss
    return heatmap


def heatmap_image_process(heatmap, image_):
    """
    resize heatmap to iamge size
    multiply to 255 and convert integer and colormap it
    then join both heatmap and image

    return preprocessed heatmap and super_imposed image
    """
    heatmap = cv2.resize(heatmap, (image_.shape[0], image_.shape[1]))
    heatmap *= 255
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    super_imposed = cv2.addWeighted(image_.numpy(), 0.8,
                                    heatmap.astype('float32'), 2e-3, 0.0)
    return heatmap, super_imposed


def all_layer_model_sample_activation(preprocessed_image,
                                      all_layer_model,
                                      layer_no,
                                      filter_no):
    """
    random filters activations will be found
    here from all_layer_model
    all_layer_model: a model where output will
    be all layers upto last convolution layaer
    layer_no: which layer not activations needs
    to be found
    filter_no: Each layer's has specific number of
    filters, which filters activations is wished

    sample_activation: at first normalized and then
    multiply it with 255 and then make sure it has
    a range [0 255] and at last converted it to int
    int
    """
    activations = all_layer_model.predict(preprocessed_image)
    sample_activations = activations[layer_no][0, :, :, filter_no]

    sample_activations -= sample_activations.mean()
    sample_activations /= sample_activations.std()
    sample_activations *= 255
    sample_activations = np.clip(
                                sample_activations,
                                0, 255).astype(np.uint8)
    return activations, sample_activations



def visualize_cam_and_activation(image,
                                 label,
                                 predicted_label,
                                 heatmap,
                                 super_imposed,
                                 activation):
    """
    visualize grad_cam, image, and a random
    layer activation in a image
    image :  sample image selected
    heatmap: heatmap created from gradcam
    superimposed: gradcam and image together
    activtion: activation of a random layer of
    a random filter
    """

    f, ax = plt.subplots(2, 2, figsize=(15, 8))
    ax[0, 0].imshow(image)
    ax[0, 0].set_title(f'True label: {label} Prediction: {predicted_label}')
    ax[0, 0].axis('off')
    ax[0, 1].imshow(activation)
    ax[0, 1].set_title('Random feature map')
    ax[0, 1].axis('off')

    ax[1, 0].imshow(heatmap)
    ax[1, 0].set_title('Class activation map')
    ax[1, 0].axis('off')

    ax[1, 1].imshow(super_imposed)
    ax[1, 1].set_title('Activation map super_imposed')
    ax[1, 1].axis('off')
