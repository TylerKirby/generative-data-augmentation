import argparse

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
    i = layers.Input(shape=(32,32,3))
    o = encoder(i)
    return tf.keras.Model(i, o, name='encoder')

def decoder(x):
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((4, 4, 8))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
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
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--latent_vector_size', type=int, default=16)
    args = parser.parse_args()

    autoencoder = autoencoder()
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    train_data_generator = ImageDataGenerator(rescale=1./255)

    training_data = train_data_generator.flow_from_directory(
        'data/fruit_samples',
        target_size=(32,32),
        class_mode='input',
        batch_size=args.batch_size
    )
    autoencoder.fit_generator(
        training_data,
        steps_per_epoch=10,
        epochs=args.epochs
    )

    autoencoder.save_weights(f'weights/autoencoder_fruits_weights_v{args.version}.h5')

    with open(f'weights/autoencoder_fruits_architecture_v{args.version}.json', 'w') as f:
        f.write(autoencoder.to_json())