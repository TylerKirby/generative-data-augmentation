import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

if __name__ == '__main__':
    imgs = ImageDataGenerator(1./255).flow_from_directory('data/fruit_small_class', target_size=(32,32), class_mode='input', batch_size=100)
    imgs = imgs.next()

    with open('weights/vae_fruits_architecture.json', 'r') as f:
        generator = tf.keras.models.model_from_json(f.read())
    generator.load_weights('weights/vae_fruits_weights.h5')

    generated_images = generator.predict(imgs)
    plt.figure(figsize=(10,10))
    for i in range(generated_images.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_images[i, :, :, :])
        plt.axis('off')
    plt.savefig('vae_fruits_output.png')