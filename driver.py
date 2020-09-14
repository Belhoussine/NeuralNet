#!/bin/python3

import ML
from ML.NeuralNetwork import NeuralNetwork
from ML.loss import *
from ML.utils import loadMNIST, flatten, oneHotEncoding
import numpy as np
import matplotlib.pyplot as plt

(train_img, train_labels), (test_img, test_labels) = loadMNIST()

#Data Processing
training_images = flatten(train_img, normalize = True)
training_labels = np.array([oneHotEncoding(label) for label in train_labels])

#Defining parameters
num_classes = 10
input_shape = (784, 1)
layers = (input_shape[0], 50, 20, num_classes)
activation = ('leakyrelu', 'tanh', 'softmax')

#Creating the model
nn = NeuralNetwork(layers, activation)

prediction = nn.predict(training_images[0])
label = training_labels[0]

loss = logcosh(prediction,label)
print(loss)