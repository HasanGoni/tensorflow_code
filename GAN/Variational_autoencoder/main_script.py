import tensorflow as tf
from test import get_dataset
from test import map_image
from model_creation import get_models
import matplotlib.pyplot as plt
from IPython import display
from viz_utils import generate_and_save_images

bs = 128
input_shape = 28, 28, 1
latent_dim = 2
train_dataset = get_dataset(map_image
                            )

encoder, decoder, vae = get_models(input_shape=(28, 28, 1,),
                                   latent_dim=latent_dim)


optimizer = tf.keras.optimizers.Adam()
loss_metric = tf.keras.metrics.Mean()
bce_loss = tf.keras.losses.BinaryCrossentropy()

# Training start
# Generate random vector as test input for decoder model

random_vector = tf.random.normal(shape=[16, 2])
epochs = 100

# generate images for decoder model

predictions = decoder.predict(random_vector)

step = 0
epoch = 0
fig = plt.figure(figsize=(4, 4))
for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
fig.suptitle(f'epoch:{epoch}: step : {step}')
# plt.show()
# plt.close(all)

epochs = 100

for epoch in range(epochs):
    for steps, x_batch_train in enumerate(train_dataset):
        print(x_batch_train.shape)
        with tf.GradientTape() as tape:
            # feed a batch to the VAE model
            # where output will be reconstracted image
            reconstracted = vae(x_batch_train)
            print(reconstracted.shape)

            # flatteing reconstructed
            flattened_outputs = tf.reshape(reconstracted, shape=[-1])
            flattened_inputs = tf.reshape(x_batch_train, shape=[-1])

            # calculating loss
            # loss = bce_loss(flattened_input, flattened_output)* 784
            loss = bce_loss(flattened_inputs, flattened_outputs) * 784
            # Adding kl divergence
            loss += sum(vae.losses)
        # get the gradients and update weights
        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        # compute metrics
        loss_metric(loss)

        if step % 100 == 0:
            display.clear_output(wait=False)
            generate_and_save_images(decoder,
                                     epoch,
                                     step,
                                     random_vector)
            # plt.close(all)
            print('Epoch: %s step: %s mean loss\
                  = %s' % (epoch, step, loss_metric.result().numpy()))
