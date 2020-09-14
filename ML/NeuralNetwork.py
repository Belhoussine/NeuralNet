import numpy as np
from .activation import softmax, sigmoid, leakyReLU
from .utils import oneHotEncoding, activate

class NeuralNetwork:
    # Initializing number of layers and their shapes and randomizing weights
    def __init__(self, layers, activation):
        self.activation = activation
        self.layers = layers
        self.weight_shapes = [(a, b) for a, b in zip(self.layers[1:], self.layers[:-1])]
        self.weights = [np.random.standard_normal(shape)/shape[1]**0.5 for shape in self.weight_shapes]
        self.biases = [np.zeros((shape, 1)) for shape in layers[1:]]


    # Training the Model
    # def train(self, training_set, training_labels, iterations):
    #     for iteration in range(iterations):
    #         self.forwardProp(training_set)
    #         self.backProp()


    # Prediction -Forward Propagation- algorithm
    def predict(self, a):
        for (w, b, f) in zip(self.weights, self.biases, self.activation):
            # print(a.shape, w.shape, b.shape, f)
            a = activate(np.matmul(w, a) + b, f)
            # print(a, '\n')
        return a


    # Representation of the Neural Network
    def __repr__(self):
        print(f"Layers: {self.layers} \n")

        print(f"Weights:")
        for i, weight in enumerate(self.weights, start=1):
            print(f" Layer {i}: {weight.shape}\n {weight}\n")

        print(f"Biases:")
        for i, bias in enumerate(self.biases, start=1):
            print(f" Layer {i}: {bias.shape}\n {bias}\n")
        
        return '\n'
    