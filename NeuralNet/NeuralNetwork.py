import numpy as np
from .utils import activate, computeLoss, optimize, shuffle, write, colors, progressBar


class NeuralNetwork:

    # Initializing number of layers and their shapes and randomizing weights
    def __init__(self, layers, activations, loss, optimizer):
        # Getting Parameters from User
        self.layers = layers
        self.activations = activations
        self.loss = loss.lower()
        self.optimizer = optimizer.lower()

        write('[ ] Initializing weights and biases...', color = colors.INFO, clear = False)
        # Creating Weights and Biases
        self.weight_shapes = [(a, b) for a, b in zip(self.layers[1:], self.layers[:-1])]
        self.weights = [np.random.standard_normal(shape)/shape[1]**0.5 for shape in self.weight_shapes]
        self.biases = [np.zeros((shape, 1)) for shape in layers[1:]]

        write('[x] Weights and Biases initialized successfully.\n', color = colors.SUCCESS, wait = 0.7)
        write('\n', wait = 0.5)

    # Training the Model
    def train(self, training_set, training_labels, epochs = 10, batchsize = 20):
        loss = 0
        batchsize = 1 if self.loss == 'sgd' else batchsize
        batchsize = len(training_set) if self.loss == 'batchgd' else batchsize
        # Training for number of epochs
        for epoch in range(epochs):
            training_data = shuffle(training_set, training_labels)
            write(f'EPOCH {epoch}:\n', wait = 0, color = colors.INFO, clear = False)

            # Looping over the whole dataset in each epoch
            for index, (input_data, label) in enumerate(training_data):
                progressBar(current = index+1, total = len(training_set), limit = 30)

                prediction, label = self.forwardProp(input_data, label)
                loss += computeLoss(prediction, label, self.loss)
                # Back propagate every batch size:
                # Batch size is 1 for SGD
                # Batch size is len(training_set) for batchGD
                if((index + 1) % batchsize == 0 or index+1 == len(training_data)):
                    self.backProp(loss / batchsize)
                    loss = 0


    # Prediction Algorithm
    def predict(self, a):
        for (w, b, f) in zip(self.weights, self.biases, self.activations):
            # print(a.shape, w.shape, b.shape, f)
            a = activate(np.matmul(w, a) + b, f)
            # print(a, '\   n')        
        return a


    # Forward Propagation algorithm
    def forwardProp(self, input_layer, label):
        activated = input_layer
        # Propagating Values and Activating Neurons
        for (w, b, f) in zip(self.weights, self.biases, self.activations):
            # print(activated.shape, w.shape, b.shape, f)
            activated = activate(np.matmul(w, activated) + b, f)
            # print(activated, '\n')

        prediction = activated
        return (prediction, label)

    # Backward Propagation algorithm
    def backProp(self, loss):
        for i in reversed(range(len((self.weights, self.biases)))):
            optimize((self.weights, self.biases), loss, self.optimizer)

    # Displaying Activation functions
    @staticmethod
    def activationFunctions():
        return  ['Sigmoid', 'Softmax', 'ReLU', 'LeakyReLU', 'elu', 'tanh']

    # Displaying Loss functions
    @staticmethod  
    def lossFunctions():
        return ['MSE', 'SSE', 'RMSE', 'MAE', 'huber', 'logcosh']

    # Displaying Loss functions
    @staticmethod  
    def optimizers():
        return ['SGD', 'gradientdescent', 'minibatchGD', 'adam', 'RMSProp']

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
    