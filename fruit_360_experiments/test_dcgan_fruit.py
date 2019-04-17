import argparse

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_examples', '-e', type=int, default=100)
    args = parser.parse_args()

    noise = np.random.rand(args.num_of_examples, 100)

    with open('weights/dcgan_generator_fruits_architecture_v1.json', 'r') as f:
        generator = tf.keras.models.model_from_json(f.read())
    generator.load_weights('weights/dcgan_generator_fruits_weights_v1.h5')

    generated_images = generator.predict(noise)
    plt.figure(figsize=(10,10))
    for i in range(generated_images.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_images[i, :, :, :])
        plt.axis('off')
    plt.savefig('gan_output.png')
    