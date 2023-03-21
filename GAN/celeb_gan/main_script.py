#! /usr/bin/env python
from pathlib import Path
import tensorflow as tf
# from getting_data import extract_zip
from model_creation import create_generator
from getting_data import load_celeb
from model_creation import create_discriminator
from custom_training import training_loop


generator = create_generator()
discriminator = create_discriminator()
print(generator.summary())
print(discriminator.summary())
dataset = load_celeb(bs=100)

z_dim = 128
test_z = tf.random.normal(shape=(64, 1, 1, z_dim))
training_loop(dataset=dataset,
              generator=generator,
              discriminator=discriminator,
              z_dim=z_dim,
              noise=test_z,
              epochs=1)
