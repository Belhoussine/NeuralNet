#!/bin/python3

# import NeuralNet
from NeuralNet.NeuralNetwork import NeuralNetwork
from NeuralNet.utils import loadMNIST, flatten, ohe, computeLoss, oneHotDecoding

# Importing Dataset and Splitting it
# (train_img, train_labels), (test_img, test_labels) = loadMNIST()

# Data Processing
# training_images = flatten(train_img, normalize = True)
# training_labels = ohe(train_labels, 10)

# Problem specific Parameters
# num_classes = 10
# input_shape = (784, 1) 
input_shape = (3, 1)
num_classes = 1

# Defining Model Parameters
layers = (input_shape[0], 2, num_classes)
activations = ('leakyrelu', 'tanh', 'sigmoid')
loss = 'rmse'
optimizer = 'sgd'

# Creating the model
nn = NeuralNetwork(layers = layers, activations = activations, loss = loss, optimizer = optimizer)

# nn.train(training_images, training_labels, epochs = 2, batchsize = 17)

# label = training_labels[0]
# prediction = nn.predict(training_images[0])

