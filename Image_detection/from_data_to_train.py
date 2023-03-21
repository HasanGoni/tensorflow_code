import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from data_preprocess import get_training_dataset, \
              get_validation_dataset, dataset_to_numpy
from model_creation_without_transfer_learning import feature_extractor, dense_layers, classification_layer, building_box_regression, final_model, define_compile_model
from vizualisation import plt_metrics
from vizualisation import display_digits_with_boxes
from vizualisation import intersection_over_union
from box_creation import draw_bounding_boxes_on_image_array, draw_bounding_boxes_on_image, draw_bounding_box_on_image

train_dataset, val_dataset = get_training_dataset(), get_validation_dataset()
(train_data, train_labels, train_boxes, valid_data,
 valid_labels, valid_boxes) = dataset_to_numpy(train_dataset, val_dataset, 10)
# If train dataset and valid dataset needs to bo seen
# then next two lines needs to be uncommented

# display_digits_with_boxes(train_data, train_labels, train_labels, np.array([]), train_boxes, np.array([]),'train_digits',5)

# display_digits_with_boxes(valid_data, valid_labels, valid_labels, np.array([]), valid_boxes, np.array([]),'valid_digits',5)
# plt.show()
inputs = tf.keras.layers.Input(shape=(75, 75, 1,))
model = define_compile_model(inputs)
# print(model.summary()
epochs = 3
batch_size = bs = 64
history = model.fit(train_dataset, steps_per_epoch=60_032//bs, validation_data=val_dataset, validation_steps=1, epochs=epochs)

loss, classificaion_loss, bounding_box_loss, classificaion_accuracy,\
bounding_box_mse = model.evaluate(val_dataset, steps=1)
# plt_metrics('classifier_loss', 'Classification loss', history)
# plt_metrics('classifier_accuracy', 'Classification accuracy', history)

predictions = model.predict(valid_data, batch_size=64)
predicted_labels = np.argmax(predictions[0], axis=1)
predicted_boxes = predictions[1]

iou = intersection_over_union(predicted_boxes, valid_boxes)
iou_threshold = 0.6
print("Number of predictions where iou > threshold(%s): %s" % (iou_threshold, (iou >= iou_threshold).sum()))
print("Number of predictions where iou < threshold(%s): %s" % (iou_threshold, (iou < iou_threshold).sum()))

display_digits_with_boxes(valid_data, predicted_labels, valid_labels, predicted_boxes, valid_boxes, iou, "True and Predicted values", 5)
plt.show()
# model.save('saved_mode.h5')
