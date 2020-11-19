import numpy as np

# Logistic activation function (scaling numbers between 0 and 1 non-linearly)
def test_sigmoid(layer):
    return 1/(1 + np.exp(-layer))


# Map non normalized output to a probability distribution (Last layer)
def test_softmax(layer):
    return np.exp(layer) / sum(np.exp(layer))


# ReLU (Rectified Linear Unit)
def test_relu(layer):
    return np.array([max(0 * elem, elem) for elem in layer])


# Leaky ReLU (Positive Linear Function Leaking on big Negative values)
def test_leakyrelu(layer, alpha=0.01):
    return np.array([max(alpha * elem, elem) for elem in layer])


# Maps real valued input between -1 and 1 (S shape)
def test_tanh(layer):
    return np.tanh(layer)

# Exponential Linear Unit
def test_elu(layer, alpha = 0.01):
    return ([elem if elem >= 0 else alpha*(np.exp(elem) -1) for elem in layer])
