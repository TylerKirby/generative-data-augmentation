import argparse

import tensorflow as tf
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_class', type=int, default=3)
    args = parser.parse_args()

    (X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(60000, -1)
    df = pd.DataFrame(data=X_train)
    df['y'] = y_train
    df = df.groupby('y').apply(lambda df: df.sample(n=1000))
    df = df.drop(df.query(f'y == {args.small_class}').sample(frac=0.9).index)

    df.to_pickle('data/mnist_set.pkl')