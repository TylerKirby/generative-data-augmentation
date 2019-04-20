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

    with open('weights/dcgan_generator_fruits_architecture_v1.json', 'r') as f:
        generator = tf.keras.models.model_from_json(f.read())
    generator.load_weights('weights/dcgan_generator_fruits_weights_v1.h5')

    noise = np.random.rand(975, 100)
    generated_images = generator.predict(noise)
    # plum label = [0, 0, 0, 0, 0, 1, 0]
    latent_of_generated = encoder(generated_images).numpy()

    X.append(latent_of_generated)
    y.append([[0, 0, 0, 0, 0, 1, 0]] * 975)

    X = np.concatenate(X)
    y = np.concatenate(y)

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    class_names = ['apples', 'cherries', 'grapes', 'peaches', 'pears', 'plums', 'tomatoes']
    metrics_report = classification_report(y_test, y_pred, target_names=class_names)
    print(metrics_report)
