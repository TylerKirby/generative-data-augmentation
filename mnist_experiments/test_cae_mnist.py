import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

if __name__ == '__main__':
    with open('weights/autoencoder_mnist_architecture_v1.json', 'r') as f:
        cae = tf.keras.models.model_from_json(f.read())
    cae.load_weights('weights/autoencoder_mnist_weights_v1.h5')
    encoder = cae.layers[0]

    df = pd.read_pickle('data/mnist_set.pkl')
    X_train = df.drop('y', axis=1).values.reshape(9100, 28, 28, 1)
    X_train = X_train.astype('float32') / 255
    y = df['y'].values

    X_latent = encoder(X_train).numpy()

    # N = testing_data.n
    # iterations = N // 32

    # X = []
    # y = []

    # for _ in range(iterations):
    #     batch = testing_data.next()

    #     images = batch[0]
    #     labels = batch[1]

    #     latent_vectors = encoder(images).numpy()
    #     X.append(latent_vectors)
    #     y.append(labels)
    
    # X = np.concatenate(X)
    # y = np.concatenate(y)

    X_train, X_test, y_train, y_test = train_test_split(X_latent, y, test_size=0.1)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    metrics_report = classification_report(y_test, y_pred)
    print(metrics_report)

