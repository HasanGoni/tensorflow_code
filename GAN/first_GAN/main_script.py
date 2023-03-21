import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess_data import X_train
from model_creation import generator_model
from model_creation import discriminator_model
from viz_utils import plot_multiple_images
from model_creation import gan_model
from model_creation import gan_trining


Batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices(X_train)
train_ds = train_ds.shuffle(1000).batch(Batch_size,
                                        drop_remainder=True).prefetch(1)

# generate batch of noise, batch size = 16
random_dim = 16
test_noise = tf.random.normal(shape=[16, random_dim])


print(test_noise.shape)
generator = generator_model(random_dim)
print(generator.summary())
# feed the batch of test generator to the untrained model
fake_image = generator(test_noise)
print(fake_image.shape)
discriminator = discriminator_model(input=28)

discriminator.compile(
                      optimizer='rmsprop',
                      loss='binary_crossentropy')
discriminator.trainable = False
gan = gan_model(generator,
                discriminator)


gan.compile(optimizer='rmsprop',
            loss='binary_crossentropy')
# We have created all the models and allready
# compiled the model with optimizer
# and loss fuction

# Now we will create our training
gan_trining(gan,
            train_dataset=train_ds,
            epochs=100)
