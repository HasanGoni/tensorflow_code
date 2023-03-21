import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from data_preprocess import load_images
from viz_utils import show_images_with_object
from download_model import vgg_model

image_dir = 'images'

# set default imageimage_dir = 'images's
content_path = f'{image_dir}/swan.jpg'
style_path = f'{image_dir}/painting.jpg'

content_image, style_image = load_images(
                                         content_path,
                                         style_path)
show_images_with_object([content_image,
                        style_image],
                        title=[
                              f'content image: {content_path}',
                              f'style image: {style_path}'])

# plt.show()
# Clear session to make the name of the layer
# Consistent when re-running it again
K.clear_session()
# Download vgg model and see the layer names

# Normally in vgg model there will be two
# convolution layer in each block.
# Style is responsible for first convs
# And Content is responsible for only last conv
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

content_layer = ['block5_conv2']
output_layers = style_layers + content_layer

num_content_layer = len(content_layer)
num_style_layer = len(style_layers)
vgg = vgg_model(output_layers)
# this vgg models output will be
# total 6 output layers. 5 for style
# image and one content image

# we will start preprocess image for vgg model
# what will we do actually centres the pixel
# values of an image
temp_layer_list = [i.output for i in vgg.layers]

