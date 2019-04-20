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

    X_train, X_test, y_train, y_test = train_test_split(X_latent, y, test_size=0.1)

    with open('weights/vae_mnist_architecture.json', 'r') as f:
        generator = tf.keras.models.model_from_json(f.read())
    generator.load_weights('weights/vae_mnist_weights.h5')

    three_examples = df.loc[df['y'] == 3]
    three_examples = three_examples.drop('y', axis=1)
    generated_X = []

    for _ in range(9):
        gen_images = generator.predict(three_examples)
        gen_images = imgs = gen_images.reshape(100, 28, 28, 1)
        gen_latent = encoder(gen_images).numpy()
        generated_X.append(gen_latent)
    
    generated_X = np.concatenate(generated_X)
    X_train = np.concatenate([X_train, generated_X])
    y_train = np.concatenate([y_train, [3] * 900])

    df_test = pd.read_pickle('data/mnist_test.pkl')
    more_test_X = df_test.drop('y', axis=1).values.reshape(1000, 28, 28, 1)
    more_test_X = more_test_X.astype('float32') / 255
    more_test_y = df_test['y'].values

    more_test_X = encoder(more_test_X).numpy()
    X_test = np.concatenate([X_test, more_test_X])
    y_test = np.concatenate([y_test, [3] * 1000])

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    metrics_report = classification_report(y_test, y_pred)
    print(metrics_report)

