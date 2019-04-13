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
    return x

def decoder(x):
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return x

def autoencoder():
    i = layers.Input(shape=(32,32,3))
    encoded = encoder(i)
    decoded = decoder(encoded)
    return tf.keras.Model(i, decoded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--version', '-v', type=int, required=True)
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--latent_vector_size', type=int, default=16)
    args = parser.parse_args()

    autoencoder = autoencoder()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    train_data_generator = ImageDataGenerator(rescale=1./255)
    test_data_generator = ImageDataGenerator(rescale=1./255)

    training_data = train_data_generator.flow_from_directory(
        'train',
        target_size=(32,32),
        class_mode='input',
        batch_size=16
    )
    testing_data = test_data_generator.flow_from_directory(
        'test',
        target_size=(32,32),
        class_mode=None,
        batch_size=16
    )

    print(training_data.next()[1].shape)

    autoencoder.fit_generator(
        training_data,
        steps_per_epoch=10,
        epochs=args.epochs
    )

    autoencoder.save_weights(f'autoencoder_weights_v{args.version}.h5')