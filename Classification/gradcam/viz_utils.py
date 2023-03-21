import numpy as np
import matplotlib.pyplot as plt


def visualize_intermediate_activation(layer_names,
                                      activations,
                                      images_per_row=16):
    """
    intermediate layer acivation visualization
    layer_names : name of the layer of a trained model
    where all convoluation output is assumed to have as
    an output
    activations: activations of those convolution layer
    after using a test image on this model
    """
    assert len(layer_names) == len(activations),\
           "Make sure layer_names and activations have same length!"
    for layer_name, layer_activations in zip(layer_names, activations):
        no_filters = layer_activations.shape[-1]
        no_columns = no_filters // images_per_row
        
        size = layer_activations.shape[1]
        print(f'size = {size}, no_columns = {no_columns} images_per_row = {images_per_row}')
        #grid = np.zeros((size*no_columns), (size*images_per_row))
        grid = np.zeros((size*no_columns, size*images_per_row))
        for col in range(no_columns):
            for row in range(images_per_row):
                feature_map = layer_activations[0, :, :, col * images_per_row +
                                                row]
                feature_map -= feature_map.mean()
                feature_map *= feature_map.std()
                feature_map *= 255
                feature_map = np.clip(feature_map,
                                      0, 255).astype(np.uint8)
                grid[col*size:(col + 1) * size, row * size: (row + 1) * size] = feature_map
        scale = 1./size
        plt.figure(figsize=(scale * grid.shape[1],
                         scale * grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.axis('off')
        plt.imshow(grid, aspect='auto',   cmap='viridis')
    plt.show()
