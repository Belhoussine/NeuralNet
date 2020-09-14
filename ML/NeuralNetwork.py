import numpy as np
from .utils import activate

class NeuralNetwork:
    # Initializing number of layers and their shapes and randomizing weights
    def __init__(self, layers, activation):
        self.__activationFunctions = ['Sigmoid', 'Softmax', 'ReLU', 'LeakyReLU', 'elu', 'tanh']
        self.__lossFunctions = ['RMSE', 'MSE', 'MAE', 'huber', 'logcosh']
        self.activation = activation
        self.layers = layers
        self.weight_shapes = [(a, b) for a, b in zip(self.layers[1:], self.layers[:-1])]
        self.weights = [np.random.standard_normal(shape)/shape[1]**0.5 for shape in self.weight_shapes]
        self.biases = [np.zeros((shape, 1)) for shape in layers[1:]]


    # Training the Model
    def train(self, training_set, training_labels, iterations):
        for i in range(iterations):
            self.forwardProp(training_set)
            self.backProp()


    def predict(self, input_layer):
        return self.forwardProp(input_layer)  

    # Forward Propagation algorithm
    def forwardProp(self, input_layer):
        for (w, b, f) in zip(self.weights, self.biases, self.activation):
            # print(input_layer.shape, w.shape, b.shape, f)
            input_layer = activate(np.matmul(w, input_layer) + b, f)
            # print(input_layer, '\n')
        return input_layer


    # Backward Propagation algorithm
    def backProp(self):
        pass

    # Displaying Activation functions
    def activationFunctions(self):
        return self.__activationFunctions


    # Displaying Loss functions
    def lossFunctions(self):
        return self.__lossFunctions

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
    