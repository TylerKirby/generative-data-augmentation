import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

if __name__ == '__main__':
    with open('weights/autoencoder_fruits_architecture_v1.json', 'r') as f:
        cae = tf.keras.models.model_from_json(f.read())
    cae.load_weights('weights/autoencoder_fruits_weights_v1.h5')
    encoder = cae.layers[0]

    testing_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'data/fruit_samples',
        target_size=(32,32),
        class_mode='categorical',
    )
    N = testing_data.n
    iterations = N // 32

    X = []
    y = []

    for _ in range(iterations):
        batch = testing_data.next()

        images = batch[0]
        labels = batch[1]

        latent_vectors = encoder(images).numpy()
        X.append(latent_vectors)
        y.append(labels)

    with open('weights/dcgan_generator_fruits_architecture_v2.json', 'r') as f:
        generator = tf.keras.models.model_from_json(f.read())
    generator.load_weights('weights/dcgan_generator_fruits_weights_v2.h5')

    X = np.concatenate(X)
    y = np.concatenate(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    plum_test = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'data/fruit_test',
        target_size=(32,32),
        class_mode=None,
    )
    N = plum_test.n
    iterations = 2

    X = []
    y = []

    for _ in range(iterations):
        batch = plum_test.next()

        images = batch

        latent_vectors = encoder(images).numpy()
        X.append(latent_vectors)

    X = np.concatenate(X)

    X_test = np.concatenate([X_test, X])
    y_test = np.concatenate([y_test, [[0, 0, 0, 0, 0, 1, 0]] * (32*iterations)])

    with open('weights/vae_fruits_architecture.json', 'r') as f:
        generator = tf.keras.models.model_from_json(f.read())
    generator.load_weights('weights/vae_fruits_weights.h5')

    X = []
    y = []

    imgs = ImageDataGenerator(1./255).flow_from_directory('data/fruit_small_class', target_size=(32,32), class_mode=None)
    for _ in range(39):
        batch = imgs.next()
        gen = generator.predict(batch)
        latent = encoder(gen).numpy()
        X.append(latent)

    X = np.concatenate(X)
    X_train = np.concatenate([X_train, X])
    y_train = np.concatenate([y_train, [[0, 0, 0, 0, 0, 1, 0]] * 975])

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    class_names = ['apples', 'cherries', 'grapes', 'peaches', 'pears', 'plums', 'tomatoes']
    metrics_report = classification_report(y_test, y_pred, target_names=class_names)
    print(metrics_report)


    