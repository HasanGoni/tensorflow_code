import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

loss_func = tf.keras.losses.BinaryCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE)
gen_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5)

dis_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5)


def make_grid(images,
              nrow,
              padding=0):
    assert images.ndim == 4 and nrow > 0
    batch, height, width, channel = images.shape
    n = nrow * (batch // nrow + np.sign(batch % nrow))
    ncol = n // nrow

    pad = np.zeros((n-batch, height, width, channel), images.dtype)
    x = np.concatenate([images, pad], axis=0)

    if padding > 0:
        x = np.pad(x, ((0, 0),
                       (0, padding),
                       (0, padding),
                       (0, 0)),
                   'constant',
                    constant_values=(0, 0))
        height += padding
        width += padding
    x = x.reshape(ncol,
                  nrow,
                  height,
                  width,
                  channel)
    x = x.transpose([0, 2, 1, 3, 4])  # (ncol, height, nrow, width, channel)
    x = x.reshape(height * ncol,
                  width * nrow,
                  channel)

    if padding > 0:
        x = x[: (height * ncol - padding),
              : (width * nrow - padding), :]
    return x


def save_image(images,
               filepath,
               nrow,
               padding=0):
    grid_img = make_grid(images,
                         nrow,
                         padding=padding)
    grid_img = ((grid_img + 1.0) * 127.5).astype(np.uint8)
    with Image.fromarray(grid_img) as img:
        img.save(filepath)


def train_on_batch(image1, image2,
                   generator,
                   discriminator,
                   z_dim=128,
                   ):
    """
    Train gan for one batch
    """
    # At first concatenating real images
    real_images = tf.concat([image1,
                            image2], axis=0)

    noise = tf.random.normal(shape=(real_images.shape[0],
                             1, 1, z_dim))

    # First training the discriminator
    with tf.GradientTape() as dis_tape:
        # creating random noise for
        # generator
        fake_img = generator(noise)
        # putting generator image
        # to the discriminator
        fake_out = discriminator(fake_img)
        # putting real images to the
        # real images
        real_out = discriminator(real_images)

        fake_loss = loss_func(tf.zeros_like(fake_out), fake_out)
        real_loss = loss_func(tf.ones_like(real_out), real_out)

        # Now putting the loss together
        dis_loss = fake_loss + real_loss
        dis_loss = tf.reduce_sum(dis_loss)/(real_images.shape[0] * 2)

    grad_dis = dis_tape.gradient(dis_loss,
                                 discriminator.trainable_variables)
    dis_optimizer.apply_gradients(zip(grad_dis,
                                      discriminator.trainable_variables))

    # phase two training the generator
    with tf.GradientTape() as gen_tape:

        # again generate noise
        fake_img = generator(noise)
        fake_out = discriminator(fake_img)
        real_out = discriminator(real_images)
        fake_loss = loss_func(tf.zeros_like(fake_out),
                              fake_out)
        real_loss = loss_func(tf.ones_like(real_out),
                              real_out)
        gen_loss = fake_loss + real_loss
        gen_loss = tf.reduce_sum(gen_loss)/(real_images.shape[0] * 2)
    grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(grad_gen,
                                      generator.trainable_variables))
    return dis_loss, gen_loss, fake_img


def training_loop(dataset,
                  generator,
                  discriminator,
                  z_dim,
                  noise,
                  epochs=1):
    """
    """
    for epoch in range(epochs):
        with tqdm(dataset) as pbar:
            pbar.set_description(f"[Epoch {epoch}]")
            for idx, (x1, x2) in enumerate(pbar):
                dis_loss, gen_loss, fake_img = train_on_batch(x1,
                                                              x2,
                                                              generator,
                                                              discriminator,
                                                              z_dim=128)
                pbar.set_postfix({'generator_loss':
                                  gen_loss.numpy(),
                                  'discriminator_loss':
                                  dis_loss.numpy()})
        fake_img = generator(noise)
        celeb_out = Path('celeb_out')
        celeb_out.mkdir(exist_ok=True,
                        parents=True)
        im_path = celeb_out/f'epoch_{epoch:04}.png'
        save_image(fake_img.numpy()[:64],
                   im_path,
                   8)
