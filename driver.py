#!/bin/python3

import ML
from ML.NeuralNetwork import NeuralNetwork
from ML.utils import loadMNIST, flatten, ohe, computeLoss, oneHotDecoding
import random

# Importing Dataset and Splitting it
(train_img, train_labels), (test_img, test_labels) = loadMNIST()

# Data Processing
training_images = flatten(train_img, normalize = True)
training_labels = ohe(train_labels, 10)

# Problem specific Parameters
num_classes = 10
input_shape = (784, 1) 

# Defining Model Parameters
layers = (input_shape[0], 5, 3, num_classes)
activations = ('leakyrelu', 'tanh', 'sigmoid')
loss = 'rmse'
optimization = 'sgd'

# Creating the model
nn = NeuralNetwork(layers = layers, activations = activations, loss = loss, optimization = optimization)

nn.train(training_images[:100], training_labels[:100], epochs = 2, batchsize = 17)

label = training_labels[0]
prediction = nn.predict(training_images[0])

