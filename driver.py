#!/bin/python3

import ML
from ML.NeuralNetwork import NeuralNetwork
from ML.utils import loadMNIST, flatten, ohe, computeLoss
import random

# Importing Dataset and Splitting it
(train_img, train_labels), (test_img, test_labels) = loadMNIST()

# Data Processing
training_images = flatten(train_img, normalize = True)
training_labels = ohe(train_labels)

# Defining Model Parameters
num_classes = 10
input_shape = (784, 1) 
layers = (input_shape[0], 5, num_classes)
activation = ('leakyrelu', 'tanh', 'softmax')
loss = 'mse'

# Creating the model
nn = NeuralNetwork(layers = layers, activation = activation, loss = loss)

label = training_labels[0]
prediction = nn.forwardProp(training_images[0], label)

# print(f'Predicted value: {oneHotDecoding(prediction)}')
# print(f'Actual value: {oneHotDecoding(label)}')

# print(prediction)
# print(label)
