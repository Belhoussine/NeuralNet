from ML import NeuralNetwork, accuracy

layers = (4, 8, 6, 4, 4, 2)
activation = ('relu', 'sigmoid', 'tanh', 'leakyrelu', 'softmax')
nn = NeuralNetwork(layers, activation)

nn.forward([1,2,3,4])