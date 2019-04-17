from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
# import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
   
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon




    # plt.figure(figsize=(10, 10))
    # start_range = digit_size // 2
    # end_range = (n - 1) * digit_size + start_range + 1
    # pixel_range = np.arange(start_range, end_range, digit_size)
    # sample_range_x = np.round(grid_x, 1)
    # sample_range_y = np.round(grid_y, 1)
    # plt.xticks(pixel_range, sample_range_x)
    # plt.yticks(pixel_range, sample_range_y)
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.imshow(figure, cmap='Greys_r')
    # plt.savefig(filename)
    # plt.show()

train_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)



x_train = train_data_generator.flow_from_directory('train', target_size=(32,32), class_mode='input', batch_size=16)    

x_test = test_data_generator.flow_from_directory('test', target_size=(32,32), class_mode=None, batch_size=16)

y_test = test_data_generator.flow_from_directory('test', target_size=(32,32), class_mode=None, batch_size=16)



original_dim = 3
# network parameters
input_shape = (original_dim, )


intermediate_dim = 512
batch_size = 128
latent_dim = 32
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=(32,32,3), name='encoder_input')
print('input layer',inputs)
x = Dense(intermediate_dim, activation='relu')(inputs)
print('the value', x)
z_mean = Dense(latent_dim, name='z_mean')(x)
print('z_mean value is', z_mean)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
print('z_log_var value is', z_log_var)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
print('this is encoder',encoder)
encoder.summary()
# plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)
print('jjjjjjjjjjjjjjjjjjjjjj',outputs)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
print('this is output', outputs)
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)
    

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
     reconstruction_loss = mse(inputs, outputs)
     print('this other value',reconstruction_loss)
	   
	   
    else:
        reconstruction_loss = binary_crossentropy(inputs,outputs)
        print('this value',reconstruction_loss)

    reconstruction_loss *= original_dim
    
    kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=-1)
    print('this is the other value',kl_loss)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    # plot_model(vae,
    #            to_file='vae_mlp.png',
    #            show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        
       vae.fit_generator(
         x_train,
         steps_per_epoch=10,
         epochs=2)

