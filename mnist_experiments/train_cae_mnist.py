import argparse
import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def encoder(x):
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, input_dim=128, activation='relu')(x)
    return x

def build_encoder():
    i = layers.Input(shape=(28,28,1))
    o = encoder(i)
    return tf.keras.Model(i, o, name='encoder')

def decoder(x):
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((4, 4, 8))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return x

def build_decoder():
    i = layers.Input(shape=(16,))
    o = decoder(i)
    return tf.keras.Model(i, o, name='decoder')

def autoencoder():
    encoder = build_encoder()
    decoder = build_decoder()
    encoder.summary()
    decoder.summary()
    autoencoder = tf.keras.models.Sequential()
    autoencoder.add(encoder)
    autoencoder.add(decoder)
    return autoencoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=int, required=True)
    parser.add_argument('--epochs', '-e', type=int, default=1)
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--latent_vector_size', type=int, default=16)
    args = parser.parse_args()

    autoencoder = autoencoder()
    autoencoder.summary()
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    df = pd.read_pickle('data/mnist_set.pkl')
    X_train = df.drop('y', axis=1).values.reshape(9100, 28, 28, 1)
    X_train = X_train.astype('float32') / 255

    autoencoder.fit(
        X_train,
        X_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10
            )
        ]
    )


    autoencoder.save_weights(f'weights/autoencoder_mnist_weights_v{args.version}.h5')

    with open(f'weights/autoencoder_mnist_architecture_v{args.version}.json', 'w') as f:
        f.write(autoencoder.to_json())