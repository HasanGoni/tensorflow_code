from model_creation import encoder_model
from model_creation import decoder_model
from model_creation import vae_model


latent_dim = 512
image_size = (64, 64, 3,)

model, conv_shape = encoder_model(latent_dim,
                                  image_size)
print(model.summary())
decoder_model = decoder_model(latent_dim,
                              conv_shape)
vae = vae_model(model,
                decoder_model,
                image_size)
print(vae.summary())
