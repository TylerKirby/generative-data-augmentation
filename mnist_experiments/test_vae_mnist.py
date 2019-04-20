import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df = pd.read_pickle('data/mnist_set.pkl')
    df = df.loc[df['y'] == 3]
    x_train = df.drop('y', axis=1).values

    with open('weights/vae_mnist_architecture.json', 'r') as f:
        generator = tf.keras.models.model_from_json(f.read())
    generator.load_weights('weights/vae_mnist_weights.h5')

    generated_images = generator.predict(x_train)
    imgs = generated_images.reshape(100, 28, 28, 1)
    plt.figure(figsize=(10,10))
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(imgs[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig('vae_mnist_output.png')