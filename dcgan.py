import tensorflow as tf
from tensorflow.keras import layers

# x must be a 100x1 vector sampled from a normal distribution
def generator(x):
    i = layers.Input(shape=(32,32,3))
    x = layers.Dense(1024, activation='tanh')(x)
    x = layers.Dense(128*7*7)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, (5, 5), padding='same', activation='tanh')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(3, (5, 5), padding='same', activation='tanh')(x)
    # Output size: 32x32x3
    x = layers.Activation('tanh')(x)
    return x

def build_generator():
    i = layers.Input(shape=(100,1,1))
    o = generator(o)
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


if __name__ == "__main__":
    i = layers.Input(shape=(32,32,3))
    o = discriminator(i)
    g = tf.keras.Model(i, o)
    g.summary()

