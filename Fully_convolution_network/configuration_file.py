BATCH_SIZE = 64
# pixel labels in the video frames
class_names = [
    'sky', 'building',
    'column/pole', 'road',
    'side walk', 'vegetation',
    'traffic light', 'fence',
    'vehicle', 'pedestrian',
    'byciclist', 'void']

num_classes = len(class_names)
vgg_weight_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
train_count = 367

# number of validation images
validation_count = 101

EPOCHS = 170
steps_per_epoch = train_count//BATCH_SIZE
validation_steps = validation_count//BATCH_SIZE
