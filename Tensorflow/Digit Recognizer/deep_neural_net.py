# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:33:22 2018

@author: lukasz.dymanski
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sess = tf.InteractiveSession()

# Constant variables
image_width = 28
image_height = 28
num_classes = 10
learning_rate = 0.002
epochs = 6
epsilon = 1e-3

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

# Helper functions to create weights, biases, convolution and pooling layers
def weights(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias(shape, name=None):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name=name)

def conv(x, W, name=None):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME', name=name)

def max_pool(x, name=None):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)


# Function display an image
def display(img):
    
    # (784) => (28,28)
    one_image = img.reshape(image_height,image_width)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)

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

    
# Define placeholders 
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape = [None, image_height*image_width], name="x")
    y = tf.placeholder(tf.float32, shape = [None, num_classes], name="y")
    # Change shape of input from list of values into 28 pixel X 28 pixel X 1 grayscale calue
    x_image = tf.reshape(x, [-1, image_height, image_width, 1], name="x_image")


# 1st Convolution layer
with tf.name_scope('Conv1'):
    # 32 features for each 5X5 patch of the image
    W_conv1 = weights([5,5,1,32], name="weights")
    b_conv1 = bias([32], name="bias")
    # Do convolution on images, add bias and push through RELU activation
    h_conv1 = tf.nn.relu(conv(x_image, W_conv1) + b_conv1, name="relu")
    # take results and run through max_pool
    h_pool1 = max_pool(h_conv1, name="pool")

# 2nd Convolution layer
with tf.name_scope('Conv2'):
    # Process the 32 features from Convolution layer 1, in 5 X 5 patch.  
    # Return 64 features weights and biases
    W_conv2 = weights([5,5,32,64], name="weights")
    b_conv2 = bias([64], name="bias")
    # Do convolution of the output of the 1st convolution layer.  Pool results 
    h_conv2 = tf.nn.relu(conv(h_pool1, W_conv2) + b_conv2, name="relu")
    h_pool2 = max_pool(h_conv2, name="pool")
    
with tf.name_scope('FC'):    
    # Fully connected layer
    W_fc1 = weights([7*7*64, 1024], name="weights")
    b_fc1 = bias([1024], name="bias")
    
    # Connect output of 2 pooling layer as input to fully connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    z_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    
    # Batch normalization on FC1 layer 
    batch_mean, batch_var = tf.nn.moments(z_fc1,[0])
    scale = tf.Variable(tf.ones([1024]))
    beta = tf.Variable(tf.ones([1024]))
    z_fc1_batch_norm = tf.nn.batch_normalization(z_fc1,batch_mean,batch_var,beta,scale,epsilon)
    
    h_fc1 = tf.nn.relu(z_fc1_batch_norm, name="relu")

# Perform dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope("FC1"):
    # Output layer
    W_fc2 = weights([1024, 10])
    b_fc2 = bias([10])


# Define output 
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss measurement
with tf.name_scope("cross_entropy"):  
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

# Optimization    
with tf.name_scope("optimizer"):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


with tf.name_scope("accuracy"):
    # What is correct
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
    # How accurate is it?
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define predicted values
predictions = tf.argmax(y_conv,1)

# init global variables
init = tf.global_variables_initializer()
sess.run(init)

# Train the model
import time

#  define number of steps and how often we display progress
mini_batch_size = 100
num_steps = x_train.shape[0]//mini_batch_size
display_every = 100

# Create input object which reads data from MNIST datasets. 
# This dataset will be used as validation set 
# Perform one-hot encoding to define the digit
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Start timer
start_time = time.time()
end_time = time.time()

for epoch in range(epochs):
    print("epoch {0}".format(epoch))
    for i in range(num_steps):
        start = i * mini_batch_size
        end = start + mini_batch_size
        x_batch = x_train[start:end, :]
        y_batch = y_train[start:end, :]
        train_step.run(feed_dict={x: x_batch, y: y_batch, keep_prob: 0.7})
    
        # Periodic status display
        if i%display_every == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:x_batch, y: y_batch, keep_prob: 1.0})
            end_time = time.time()
            print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))


# Display summary 
#     Time to train
end_time = time.time()
print("Total training time: {0:.1f} seconds".format(end_time-start_time))

# Accuracy on validation set data
print("Validation set accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})*100.0))

# Define array of results    
y_test = np.zeros((x_test.shape[0]))
num_steps = x_test.shape[0]//mini_batch_size

# Get predictions for test set
for i in range(num_steps):
    start = i * mini_batch_size
    end = start + mini_batch_size
    y_test[start:end] = predictions.eval(feed_dict={x: x_test[start:end,:], keep_prob: 1})

# Close session    
sess.close()
# Save predictions
save_results(y_test)
    


