import argparse

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import matplotlib as mpl
mpl.use('TkAgg')
from tensorflow.keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

m = 64
n_z = 3
n_epoch = 1000

train_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)


x_train = train_data_generator.flow_from_directory('data/fruit_small_class', target_size=(32,32), class_mode='input', batch_size=16)     
# Q(z|X) -- encoder

inputs = Input(shape=(32,32,3))
print (inputs)
h_q = Dense(512, activation='relu')(inputs)
print (h_q)
mu = Dense(n_z, activation='linear')(h_q)
print(mu)
log_sigma = Dense(n_z, activation='linear')(h_q)
print(log_sigma)

def sample_z(args):
    mu, log_sigma = args
    eps = tf.random.normal(shape=(32, 32, 3), mean=0., stddev=1.)
    return mu + tf.math.exp(log_sigma / 2) * eps

	

# Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])


# P(X|z) -- decoder
decoder_hidden = Dense(1024, activation='relu')
decoder_out = Dense(3, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)


# Overall VAE model, for reconstruction and training
vae = Model(inputs, outputs)


# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputs, mu)

# Generator model, generate new data given latent variable z
d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in)
d_h = Dense(1024, activation='relu')(d_h)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

models = (encoder, decoder)

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = tf.math.reduce_sum(tf.keras.metrics.binary_crossentropy(y_pred, y_true))
    print(recon)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * tf.math.reduce_sum(tf.math.exp(log_sigma) + tf.math.square(mu) - 1. - log_sigma, axis=1)
    print(kl)
   
    return recon + kl

		
vae.summary()
vae.compile(optimizer='adam', loss=vae_loss)
vae.fit_generator(x_train, epochs=n_epoch)

vae.save_weights(f'weights/vae_fruits_weights.h5')
with open(f'weights/vae_fruits_architecture.json', 'w') as f:
    f.write(vae.to_json())
