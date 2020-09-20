import numpy as np
import random
import urllib3
import io
from . import NeuralNetwork as nn
from .loss import *
from .activation import *
from .optimization import *
import sys
from time import sleep


# Defining colors for display messages
class colors:
    HEADER = '\033[95m'
    INFO = '\033[94m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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


# Applying given optimization algorithm for backpropagation
def optimize(loss, optimizer = 'adam'):
    if(optimizer.lower() in ['sgd', 'batchGD', 'minibatchgd', 'gradientdescent']):
        gradientdescent()

    if(optimizer.lower() == 'adam'):
        adam()

    if(optimizer.lower() == 'rmsprop'):
        rmsprop()

# Loading MNIST Data Set
# https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
def loadMNIST():
    # Request bytes from server
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    http  = urllib3.PoolManager()
    write('[ ] Loading MNIST Dataset from server...', color = colors.INFO, clear = False)
    r = http.request('GET', url)
    data = r.data

    # Turn it to file-like object
    write('[ ] Unzipping files...', color = colors.INFO)
    file_like_object = io.BytesIO(data)
    del data

    # Unzip and Split data
    mnist = np.load(file_like_object)
    train_img, train_labels = mnist['x_train'], mnist['y_train'] 
    test_img, test_labels = mnist['x_test'], mnist['y_test']
    write('[x] Dataset loaded successfully.', color = colors.SUCCESS)
    return (train_img, train_labels), (test_img, test_labels)


# Shuffles training data
def shuffle(training_set, training_labels):
    training_data = list(zip(training_set, training_labels))
    random.shuffle(training_data)
    return training_data


# Overrides text on same line
def write(string, color = colors.ENDC, wait = 0.5, clear = True):
    sleep(wait)
    if clear: 
        clearLine()
    print(f'{color}{string}{colors.ENDC}')

# Clears Line 
def clearLine():
    sys.stdout.write("\033[F\033[K")
    sys.stdout.flush()
