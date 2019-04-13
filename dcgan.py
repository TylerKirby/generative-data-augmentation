import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# x must be a 100x1 vector sampled from a normal distribution
def generator(x):
    x = layers.Dense(128*8*8, input_dim=100, activation='tanh')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, (5, 5), padding='same', activation='tanh')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(3, (5, 5), padding='same', activation='tanh')(x)
    # Output size: 32x32x3
    x = layers.Activation('tanh')(x)
    return x

def build_generator():
    i = layers.Input(shape=(100,))
    o = generator(i)
    return tf.keras.Model(i, o)

def discriminator(x):
    x = layers.Conv2D(64, (5, 5), padding='same', activation='tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (5, 5), activation='tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='tanh')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return x

def build_discriminator():
    i = layers.Input(shape=(32,32,3))
    o = discriminator(i)
    return tf.keras.Model(i, o)


def build_dcgan(generator, discriminator):
    dcgan = tf.keras.models.Sequential()
    dcgan.add(generator)
    discriminator.trainable = True
    dcgan.add(discriminator)
    return dcgan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=int, required=True)
    parser.add_argument('--epochs', '-e', type=int, default=1)
    parser.add_argument('--batch_size', '-bs', type=int, default=5)
    args = parser.parse_args()

    real_images = ImageDataGenerator().flow_from_directory(
        'images_to_spoof',
        target_size=(32,32),
        class_mode=None,
        batch_size=args.batch_size
    )

    generator = build_generator()
    discriminator = build_discriminator()
    
    generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())

    dcgan = build_dcgan(generator, discriminator)
    dcgan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())

    for i in range(args.epochs):
        print(f'epoch {i}')
        noise = np.random.rand(args.batch_size, 100)
        real_image_batch = real_images.next()
        generated_images = generator.predict(noise, batch_size=args.batch_size)
        X = np.concatenate([generated_images, real_image_batch])
        y = [0]*args.batch_size + [1]*args.batch_size
        discriminator.trainable = True
        discriminator_loss = discriminator.train_on_batch(X, y)
        print(f'discriminator loss: {discriminator_loss}')
        noise = np.random.rand(args.batch_size, 100)
        y_generator = [1]*args.batch_size
        discriminator.trainable = False
        dcgan_loss = dcgan.train_on_batch(noise, y_generator)
        print(f'dcgan loss: {dcgan_loss}')
        print('='*15)

    generator.save_weights(f'dcgan_generator_weights_v{args.version}.h5')
    with open(f'dcgan_generator_architecture_v{args.version}.json', 'w') as f:
        f.write(generator.to_json())
    discriminator.save_weights(f'dcgan_discriminator_weights_v{args.version}.h5')
    with open(f'dcgan_discriminator__architecture_v{args.version}.json', 'w') as f:
        f.write(discriminator.to_json())

    print('finished training')

    

