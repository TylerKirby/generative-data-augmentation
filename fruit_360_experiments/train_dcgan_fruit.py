import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

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
    discriminator.trainable = False
    dcgan.add(discriminator)
    generator.summary()
    discriminator.summary()
    return dcgan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=int, required=True)
    parser.add_argument('--epochs', '-e', type=int, default=500)
    parser.add_argument('--batch_size', '-bs', type=int, default=8)
    args = parser.parse_args()

    real_images = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'images_to_spoof',
        target_size=(32,32),
        class_mode=None,
        batch_size=args.batch_size
    )

    generator = build_generator()
    discriminator = build_discriminator()
    dcgan = build_dcgan(generator, discriminator)
    
    discriminator_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    dcgan.compile(loss='binary_crossentropy', optimizer=generator_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optim)

    num_of_batches = real_images.n // args.batch_size

    writer = tf.summary.create_file_writer('logs')

    for i in range(args.epochs):
        print(f'epoch {i+1}')
        for j in range(num_of_batches):
            print(f'batch {j+1}')
            # Generate noise
            noise = np.random.rand(args.batch_size, 100)
            # Get real examples
            real_image_batch = real_images.next()
            # Generate synthetic examples
            generated_images = generator.predict(noise, batch_size=args.batch_size)
            # Train discriminator
            X = np.concatenate([real_image_batch, generated_images])
            y = [0]*args.batch_size + [1]*args.batch_size
            discriminator_loss = discriminator.train_on_batch(X, y)
            print(f'discriminator loss: {discriminator_loss}')
            # Train generator
            noise = np.random.rand(args.batch_size, 100)
            y_generator = [1]*args.batch_size
            discriminator.trainable = False
            dcgan_loss = dcgan.train_on_batch(noise, y_generator)
            discriminator.trainable = True
            print(f'dcgan loss: {dcgan_loss}')
        with writer.as_default():
           tf.summary.scalar('discriminator_loss', discriminator_loss, i)
           tf.summary.scalar('generator_loss', dcgan_loss, i)
        print('='*15)

    generator.save_weights(f'dcgan_generator_weights_v{args.version}.h5')
    with open(f'dcgan_generator_architecture_v{args.version}.json', 'w') as f:
        f.write(generator.to_json())
    discriminator.save_weights(f'dcgan_discriminator_weights_v{args.version}.h5')
    with open(f'dcgan_discriminator__architecture_v{args.version}.json', 'w') as f:
        f.write(discriminator.to_json())

    print('finished training')

    
