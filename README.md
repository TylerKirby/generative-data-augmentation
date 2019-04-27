# Generative Algorithms for Data Augmentation
## Set Up Instructions
1. After cloning the project, create a folder in this directory called `data`. 
2. Download the Fruit 360 dataset from [Kaggle](https://www.kaggle.com/moltean/fruits), unzip the files, and place them under the `data` directory.
3. Install the required packages via pip (`pip install -r requirements.txt`). It is reccomended that you create a virtual environment for this project.
4. In each experiment directory, run `generate_dataset.py` to create a sampled dataset to work from.
5. To train models, run the appropriate training scripts in the directory.
6. To test models, edit the test script to load the desired weights then run the test script. It will print out a full classification report.