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
    X_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2
    ).flow(
        X_train
    )
    y = df['y'].values[:9088]

    X = []

    iterations = X_train.n // 32

    for _ in range(iterations):
        batch = X_train.next()
        latent = encoder(batch).numpy()
        X.append(latent)
    
    X = np.concatenate(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    metrics_report = classification_report(y_test, y_pred)
    print(metrics_report)

