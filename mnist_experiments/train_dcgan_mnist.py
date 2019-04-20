import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

# x must be a 100x1 vector sampled from a normal distribution
def generator(x):
    x = layers.Dense(128*7*7, input_dim=100, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((7, 7, 128))(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(1, (5, 5), padding='same', activation='tanh')(x)
    return x

def build_generator():
    i = layers.Input(shape=(100,))
    o = generator(i)
    return tf.keras.Model(i, o)

def discriminator(x):
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return x

def build_discriminator():
    i = layers.Input(shape=(28,28,1))
    o = discriminator(i)
    return tf.keras.Model(i, o)


def build_dcgan(generator, discriminator):
    generator.summary()
    discriminator.summary()
    dcgan = tf.keras.models.Sequential()
    dcgan.add(generator)
    discriminator.trainable = False
    dcgan.add(discriminator)
    return dcgan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=int, required=True)
    parser.add_argument('--epochs', '-e', type=int, default=500)
    parser.add_argument('--batch_size', '-bs', type=int, default=10)
    args = parser.parse_args()

    generator = build_generator()
    discriminator = build_discriminator()
    dcgan = build_dcgan(generator, discriminator)
    
    discriminator_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    dcgan.compile(loss='binary_crossentropy', optimizer=generator_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optim)

    df = pd.read_pickle('data/mnist_set.pkl')
    df = df.loc[df['y'] == 3]
    X_train = df.drop('y', axis=1).values.reshape(100, 28, 28, 1)
    X_train = (X_train - 127.5) / 127.5
    real_images = ImageDataGenerator().flow(
        X_train,
        y=None,
        batch_size=args.batch_size
    )

    num_of_batches = real_images.n // args.batch_size


    for i in range(args.epochs):
        print(f'epoch {i+1}')
        for j in range(num_of_batches):
            print(f'batch {j+1}')
            # Generate noise
            noise = np.random.rand(args.batch_size, 100)
            # Get real examples
            real_image_batch = real_images.next()
            # Generate synthetic examples
            generated_images = generator.predict(noise)
            # Train discriminator
            discriminator.trainable = True
            X = np.concatenate([real_image_batch, generated_images])
            y = [1]*real_image_batch.shape[0] + [0]*noise.shape[0]
            discriminator_loss = discriminator.train_on_batch(X, y)
            print(f'discriminator loss: {discriminator_loss}')
            # Train generator
            noise = np.random.rand(args.batch_size, 100)
            y_generator = [1]*args.batch_size
            discriminator.trainable = False
            dcgan_loss = dcgan.train_on_batch(noise, y_generator)
            
            print(f'dcgan loss: {dcgan_loss}')
        print('='*15)

    generator.save_weights(f'weights/dcgan_generator_mnist_weights_v{args.version}.h5')
    with open(f'weights/dcgan_generator_mnist_architecture_v{args.version}.json', 'w') as f:
        f.write(generator.to_json())
    discriminator.save_weights(f'weights/dcgan_discriminator_mnist_weights_v{args.version}.h5')
    with open(f'weights/dcgan_discriminator_mnist_architecture_v{args.version}.json', 'w') as f:
        f.write(discriminator.to_json())

    print('finished training')
