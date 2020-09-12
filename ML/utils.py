import numpy as np
from .activation import *

# Number of correct predictions over total predictions


def accuracy(predictions, labels):
    num_correct = sum([1 if np.argmax(a) == b else 0 for a,
                       b in zip(predictions, labels)])
    print(f'{num_correct}/{len(predictions)} -> Accuracy: {round((num_correct / len(predictions))*100, 2)}%')
    return num_correct / len(predictions)

# Converting 2D matrix to 1D


def flatten(matrix):
    size = len(matrix)
    flat = []
    for i in range(size):
        flat.append(np.ndarray.flatten(matrix[i])/255.0)
    return np.expand_dims(np.array(flat), axis=2)

# Converting numerical value N to binary array where Nth cell is 1 and rest is 0


def oneHotEncoding(labels):
    oneHotEncoded = []
    for label in labels:
        encoded = [1 if i == label else 0 for i in range(10)]
        oneHotEncoded.append(encoded)
    # return np.expand_dims(np.array(oneHotEncoded), axis=2)
    return np.array(oneHotEncoded)

# Root Mean Squared Error (loss function)


def RMSE(predictions, labels):
    labels = oneHotEncoding(labels)
    loss = [rmse(prediction, label)
            for prediction, label in zip(predictions, labels)]
    return np.array(loss)

# Helper Method for RMSE


def rmse(predicted, actual):
    return np.sqrt(sum(((predicted - actual) ** 2) / 10))

# Applying chosen activation function on given layer


def activate(layer, activation):
    if(activation.lower() == 'sigmoid'):
        return sigmoid(layer)
    if(activation.lower() == 'softmax'):
        return softmax(layer)
    if(activation.lower() == 'leakyrelu'):
        return leakyReLU(layer)
    if(activation.lower() == 'relu'):
        return relu(layer)
    if(activation.lower() == 'tanh'):
        return tanh(layer)
    if(activation.lower() == 'elu'):
        return elu(layer)


# Mapping values between 0 and 1 (Linear Mapping)
def lerp(layer):
    mini = min(layer)
    maxi = max(layer)
    return np.array([((elem - mini) / (maxi - mini)) for elem in layer])
