import numpy as np

# Activation function (scaling numbers between 0 and 1 non-linearly)
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# Map non normalized output to a probability distribution (Last layer)
def softmax(vector):
    return np.exp(vector) / sum(np.exp(vector))


# ReLU (Rectified Linear Unit)
def relu(vector):
    return np.array([max(0, elem) for elem in vector])


# Leaky ReLU (Positive Linear Function Leaking on big Negative values)
def leakyReLU(vector, alpha=0.01):
    return np.array([max(alpha * elem, elem) for elem in vector])


# Maps real valued input between -1 and 1 (S shape)
def tanh(vector):
    return np.tanh(vector)

