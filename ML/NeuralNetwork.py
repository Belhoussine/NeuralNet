import numpy as np
from .utils import activate, computeLoss

class NeuralNetwork:

    # Initializing number of layers and their shapes and randomizing weights
    def __init__(self, layers, activation, loss = 'mse'):
        # Getting Parameters from User
        self.layers = layers
        self.activation = activation
        self.loss = loss

        # Creating Weights and Biases
        self.weight_shapes = [(a, b) for a, b in zip(self.layers[1:], self.layers[:-1])]
        self.weights = [np.random.standard_normal(shape)/shape[1]**0.5 for shape in self.weight_shapes]
        self.biases = [np.zeros((shape, 1)) for shape in layers[1:]]


    # Training the Model
    def train(self, training_set, training_labels, iterations, batchsize = 1):
        loss = 0
        for i in range(iterations):
            # Looping Over the whole dataset
            for input_layer, label in zip(training_set, training_labels):
                loss += self.forwardProp(input_layer, label)
            
            self.backProp(loss)


    # Prediction Algorithm
    def predict(self, a):
        for (w, b, f) in zip(self.weights, self.biases, self.activation):
            print(a.shape, w.shape, b.shape, f)
            a = activate(np.matmul(w, a) + b, f)
            print(a, '\n')
        return a

    # Forward Propagation algorithm
    def forwardProp(self, input_layer, label):
        activated = input_layer
        # Propagating Values and Activating Neurons
        for (w, b, f) in zip(self.weights, self.biases, self.activation):
            print(activated.shape, w.shape, b.shape, f)
            activated = activate(np.matmul(w, activated) + b, f)
            print(activated, '\n')

        prediction = activated
        loss = computeLoss(prediction, label, self.loss)
        print(f'Loss function: {self.loss} ==>', loss)
        return loss


    # Backward Propagation algorithm
    def backProp(self, loss):
        pass

    # Displaying Activation functions
    @staticmethod
    def activationFunctions():
        return  ['Sigmoid', 'Softmax', 'ReLU', 'LeakyReLU', 'elu', 'tanh']

    # Displaying Loss functions
    @staticmethod  
    def lossFunctions():
        return ['MSE', 'SSE', 'RMSE', 'MAE', 'huber', 'logcosh']

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
    