import os
import argparse
import random
import shutil

import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    TRAINING_PATH = os.getcwd()+'/data/fruits-360/Training/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ts_split', type=float, default=0.8, help='Specify size of training/test split')
    parser.add_argument('--class_size', type=int, default=1000, help='Specify number of examples for each class')
    args = parser.parse_args()

    base_classes = {
        'apples': [
            'Apple Braeburn', 'Apple Golden 1', 'Apple Golden 2',
            'Apple Golden 3', 'Apple Granny Smith', 'Apple Red 1', 
            'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 
            'Apple Red Yellow 1', 'Apple Red Yellow 2'
        ], 
        'cherries': [
            'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 
            'Cherry Wax Red', 'Cherry Wax Yellow'
        ], 
        'grapes': [
            'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2',
            'Grape White 3', 'Grape White 4'
        ], 
        'pears': [
            'Pear', 'Pear Abate', 'Pear Kaiser', 'Pear Monster', 'Pear Williams'
        ], 
        'tomatoes': [
            'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4'
        ], 
        'peaches': [
            'Peach', 'Peach 2', 'Peach Flat'
        ],
        'plums': [
            'Plum', 'Plum 2', 'Plum 3'
        ]
    }

    for k, v in base_classes.items():
        image_paths = []
        for subclass in v:
            subclass_image_paths = [TRAINING_PATH+subclass+'/'+f for f in os.listdir(TRAINING_PATH+subclass)]
            image_paths.extend(subclass_image_paths)
        sample = random.sample(image_paths, args.class_size)
        train, test = train_test_split(sample, train_size=args.ts_split)
        os.makedirs(f'{os.getcwd()}/train/{k}')
        os.makedirs(f'{os.getcwd()}/test/{k}')
        for f in train:
            shutil.copy2(f, f'{os.getcwd()}/train/{k}')
        for f in test:
            shutil.copy2(f, f'{os.getcwd()}/test/{k}')
    


