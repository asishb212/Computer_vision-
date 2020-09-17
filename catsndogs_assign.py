import os
import random
import tensorflow as tf 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from os import getcwd
import shutil 
"""
os.mkdir("/home/ashifer/code/tensflow/catsndogsdata")
os.mkdir("/home/ashifer/code/tensflow/catsndogsdata/train")
os.mkdir("/home/ashifer/code/tensflow/catsndogsdata/train/dogs") 
os.mkdir("/home/ashifer/code/tensflow/catsndogsdata/train/cats")
os.mkdir("/home/ashifer/code/tensflow/catsndogsdata/test")
os.mkdir("/home/ashifer/code/tensflow/catsndogsdata/test/dogs")
os.mkdir("/home/ashifer/code/tensflow/catsndogsdata/test/cats")
"""
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
# YOUR CODE STARTS HERE
    dataset = []
    
    for unitData in os.listdir(SOURCE):
        data = SOURCE + unitData
        if (os.path.getsize(data) > 0):
            dataset.append(unitData)
        else:
            print('Skipped ' + unitData)
            print('Invalid file size! i.e Zero length.')
    
    train_data_length = int(len(dataset) * SPLIT_SIZE)
    test_data_length = int(len(dataset) - train_data_length)
    shuffled_set = random.sample(dataset, len(dataset))
    train_set = shuffled_set[0:train_data_length]
    test_set = shuffled_set[-test_data_length:]
    
    for unitData in train_set:
        temp_train_data = SOURCE + unitData
        final_train_data = TRAINING + unitData
        copyfile(temp_train_data, final_train_data)
    
    for unitData in test_set:
        temp_test_data = SOURCE + unitData
        final_test_data = TESTING + unitData
        copyfile(temp_train_data, final_test_data)

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
