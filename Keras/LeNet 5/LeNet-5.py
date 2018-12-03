# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 19:53:30 2018

@author: lukasz.dymanski
"""
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

sess = tf.InteractiveSession()

# Constant variables
image_width = 28
image_height = 28
num_filters = 32
max_pool_size = (2, 2) 
conv_kernel_size = (3, 3)
num_classes = 10
drop_prob = 0.5
epochs = 10
batch_size = 100

# Load input data
def load_data():
    train_set = pd.read_csv('data/train.csv')
    test_set = pd.read_csv('data/test.csv')

    x_train = train_set.iloc[:, 1:].values
    y_train = train_set.iloc[:, 0].values
    x_test = test_set.iloc[:,:].values
    
    return x_train, y_train, x_test

# Convert array into one hot matrix
def convert_to_one_hot(arr):
    one_hot = np.zeros((arr.size, arr.max() + 1))
    one_hot[np.arange(arr.size), arr] = 1
    
    return one_hot

# Normalize data values from input range (1 - 255) into target range (0. - 1) 
def normalize_data(data):
    data = data/data.max()
    return data

# Saving predictions into csv file
def save_results(preds):
    y_test = preds.astype(int)
    csv_content = pd.DataFrame({'ImageId': range(1,len(y_test)+1), 'Label': y_test})
    csv_content.to_csv('result.csv', index = False)


# Prepare input and output of Neural Network
x_train, y_train, x_test = load_data()
x_train = normalize_data(x_train)
x_test = normalize_data(x_test)
y_train = convert_to_one_hot(y_train)

# Change shape of input from list of values into 28 pixel X 28 pixel X 1 grayscale value
x_train = x_train.reshape(x_train.shape[0], image_height, image_width, 1)
x_test = x_test.reshape(x_test.shape[0], image_height, image_width, 1)

# Padding 
x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# Define type of model
model = Sequential()

# 1st layer
# Convolution
model.add(Convolution2D(filters = 6, kernel_size = 5, strides = 1, activation = 'relu',  input_shape = (32,32,1)))
# Max Pooling
model.add(MaxPooling2D(pool_size = 2, strides = 2))

# 2nd layer
# Convolution
model.add(Convolution2D(filters = 16, kernel_size = 5, strides = 1, activation = 'relu',  input_shape = (14,14,6)))
# Max Pooling
model.add(MaxPooling2D(pool_size = 2, strides = 2))

# Flattening
model.add(Flatten())

# 3rd layer 
# Fuly Connected layer 
model.add(Dense(units=120, activation='relu'))

# 4th layer
# Fully Connected
model.add(Dense(units = 84, activation = 'relu'))

#  Output layer
model.add(Dense(units = 10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train model
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)
save_results(y_pred)