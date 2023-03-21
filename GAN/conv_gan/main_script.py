from get_data import keras_data
from get_data import prepare_data
from viz_utils import plot_images
from model_creation import generator_creation
from model_creation import discriminator_model
from model_creation import GAN_
import matplotlib.pyplot as plt
import tensorflow as tf
from model_creation import gan_trining


X_train = keras_data()
ds = prepare_data(X_train)
generator = generator_creation()
# print(generator.summary())

# Try plotting random image
# which was created by generator
code_size = 32
generator_bs = 16
test_noise = tf.random.normal(shape=[generator_bs,
                                     code_size])
test_noise = generator(test_noise)

discriminator = discriminator_model()
# print(discriminator.summary())
gan = GAN_(generator,
           discriminator)
# print(gan.summary())

# Compiling all models for
# Training

discriminator.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy')
gan.compile(optimizer='rmsprop',
            loss='binary_crossentropy')

# discriminator training stop
discriminator.trainable = False
epochs = 1
gan_trining(gan,
            train_dataset=ds,
            image_decode_size=code_size,
            epochs=2)
plt.show()
