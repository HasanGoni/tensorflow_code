import tensorflow as tf
import tensorflow_datasets as tfds

LATENT_DIM = 2
BATCH_SIZE = 128
def map_image(image, label):
    '''returns a normalized and reshaped tensor from a given image'''
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    image = tf.reshape(image, shape=(28, 28, 1,))

    return image


def get_dataset(map_fn, is_validation=False):
    '''Loads and prepares the mnist dataset from TFDS.'''
    if is_validation:
        split_name = "test"
    else:
        split_name = "train"

    dataset = tfds.load('mnist', as_supervised=True, split=split_name)
    dataset = dataset.map(map_fn)
      
    if is_validation:
        dataset = dataset.batch(BATCH_SIZE)
    else:
        dataset = dataset.shuffle(1024).batch(BATCH_SIZE)

    return dataset


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        """Generates a random sample and combines with the encoder output
    
        Args:
        inputs -- output tensor from the encoder

        Returns:
        `  inputs` tensors combined with a random sample
        """

        # unpack the output of the encoder
        mu, sigma = inputs

        # get the size and dimensions of the batch
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

         #     generate a random tensor
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        # combine the inputs and noise
        return mu + tf.exp(0.5 * sigma) * epsilon
def encoder_layers(inputs, latent_dim):
    """Defines the encoder's layers.
    Args:
      inputs -- batch from the dataset
      latent_dim -- dimensionality of the latent space

    Returns:
      mu -- learned mean
      sigma -- learned standard deviation
        batch_2.shape -- shape of the features before flattening
    """

    # add the Conv2D layers followed by BatchNormalization
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation='relu', name="encode_conv1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name="encode_conv2")(x)

    # assign to a different variable so you can extract the shape later
    batch_2 = tf.keras.layers.BatchNormalization()(x)

    # flatten the features and feed into the Dense network
    x = tf.keras.layers.Flatten(name="encode_flatten")(batch_2)

    # we arbitrarily used 20 units here but feel free to change and see what results you get
    x = tf.keras.layers.Dense(20, activation='relu', name="encode_dense")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # add output Dense networks for mu and sigma, units equal to the declared latent_dim.
    mu = tf.keras.layers.Dense(latent_dim, name='latent_mu')(x)
    sigma = tf.keras.layers.Dense(latent_dim, name ='latent_sigma')(x)

    return mu, sigma, batch_2.shape
def encoder_model(latent_dim, input_shape):
    """Defines the encoder model with the Sampling layer
    Args:
    latent_dim -- dimensionality of the latent space
    input_shape -- shape of the dataset batch

    Returns:
    model -- the encoder model
    conv_shape -- shape of the features before flattening
    """

    # declare the inputs tensor with the given shape
    inputs = tf.keras.layers.Input(shape=input_shape)

    # get the output of the encoder_layers() function
    mu, sigma, conv_shape = encoder_layers(inputs, latent_dim=LATENT_DIM)

    # feed mu and sigma to the Sampling layer
    z = Sampling()((mu, sigma))

    # build the whole encoder model
    model = tf.keras.Model(inputs, outputs=[mu, sigma, z])

    return model, conv_shape

inputs = tf.keras.layers.Input(shape=(28,28,1,))
encoder, conv_sh = encoder_model(2,(28,28,1,))

print(encoder.summary())
