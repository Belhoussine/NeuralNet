#!/bin/python3

from ML import NeuralNetwork, loadMNIST, flatten, normalize, RMSE, MSE,oneHotEncoding
import random
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
print(f'Prediction: {np.argmax(prediction)}')
print(f'Actual Number: {np.argmax(training_labels[0])}')

print(prediction)
print(training_labels[0])
rmse = RMSE(prediction, training_labels[0])
mse = MSE(prediction, training_labels[0])
print(mse, rmse)