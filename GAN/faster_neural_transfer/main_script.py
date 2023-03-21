import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from data_preprocess import load_images, show_images_with_objects
from data_preprocess import tensor_to_image

path = '../neural_style_transfer/images'
# set default images
content_path = f'{path}/swan.jpg'
style_path = f'{path}/painting.jpg'
# display the content and style image
content_image, style_image = load_images(content_path, style_path)
show_images_with_objects([content_image, style_image],
                         titles=[f'content image: {content_path}',
                                 f'style image: {style_path}'])
# plt.show()

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
content_image = tf.image.convert_image_dtype(content_image, tf.float32)
style_image = tf.image.convert_image_dtype(style_image, tf.float32)

stylized = hub_module(content_image,
                      style_image)[0]
img = tensor_to_image(stylized)
print(tf.shape(stylized))
plt.imshow(img)
plt.show()
