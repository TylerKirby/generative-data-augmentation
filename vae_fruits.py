import argparse
from tensorflow.examples.tutorials.mnist import input_data

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import matplotlib as mpl
mpl.use('TkAgg')
from keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import os

m = 32
n_z = 3
n_epoch = 10

train_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)


x_train = train_data_generator.flow_from_directory('train', target_size=(32,32), class_mode='input', batch_size=16)     
x_test = test_data_generator.flow_from_directory('test', target_size=(32,32), class_mode=None, batch_size=16)
y_test = test_data_generator.flow_from_directory('test', target_size=(32,32), class_mode=None, batch_size=16)
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
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps

	

# Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])


# P(X|z) -- decoder
decoder_hidden = Dense(512, activation='relu')
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
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

models = (encoder, decoder)
data = (x_test, y_test)

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    print(recon)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    print(kl)
   
    return recon + kl



	
	
	
		
		
vae.summary()
vae.compile(optimizer='adam', loss=vae_loss)
vae.fit_generator(x_train,steps_per_epoch=10, epochs=n_epoch)
