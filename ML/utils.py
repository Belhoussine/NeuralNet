import numpy as np
import random
from . import NeuralNetwork as nn
from .loss import *
from .activation import *

# Number of correct predictions over total predictions
def accuracy(predictions, labels):
    num_correct = sum([1 if np.argmax(a) == b else 0 for a,b in zip(predictions, labels)])
    print(f'{num_correct}/{len(predictions)} -> Accuracy: {round((num_correct / len(predictions))*100, 2)}%')
    return num_correct / len(predictions)


# Converting 2D matrix to 1D
def flatten(matrices, normalize = False):
    factor = 255 if normalize else 1
    return np.expand_dims([np.ndarray.flatten(matrix) / factor for matrix in matrices], axis = 2)


# Mapping values between 0 and 1 (Linear Mapping)
def normalize(layer):
        mini = min(layer)
        maxi = max(layer)
        return  np.array([((elem - mini) / (maxi - mini)) for elem in layer])
        

# Converting numerical value N to binary array where Nth cell is 1 and rest is 0
def ohe(labels, nClasses):
    return np.array([oneHotEncoding(label, 10) for label in labels])

def oneHotEncoding(label, nClasses):
    return np.expand_dims([1 if i == label else 0 for i in range(nClasses)], axis = 1)


# Reversing One Hot Encoding Algorithm
def oneHotDecoding(encoded):
    return np.argmax(encoded)


# Applying chosen activation function on given layer
def activate(layer, activation):
    keys = [key.lower() for key in nn.NeuralNetwork.activationFunctions()]
    values = [eval(key.lower()) for key in keys]
    activationMapping = dict(zip(keys, values))
    return activationMapping[activation.lower()](layer)


# Computing Loss Depending on chosen Loss function
def computeLoss(prediction, label, lossFunction):
    keys = [key.lower() for key in nn.NeuralNetwork.lossFunctions()]
    values = [eval(key.lower()) for key in keys]
    lossMapping = dict(zip(keys, values))
    return lossMapping[lossFunction.lower()](prediction, label)


# Loading MNIST Data Set
def loadMNIST():
    mnist = np.load('./MNIST/mnist.npz')
    train_img, train_labels = mnist['x_train'], mnist['y_train'] 
    test_img, test_labels = mnist['x_test'], mnist['y_test']
    return (train_img, train_labels), (test_img, test_labels)


# Shuffles training data
def shuffle(training_set, training_labels):
    training_data = list(zip(training_set, training_labels))
    random.shuffle(training_data)
    return training_data