from ML import NeuralNetwork, accuracy

layers = (4, 6, 2)
activation = ('leakyrelu', 'sigmoid')
nn = NeuralNetwork(layers, activation)

nn.forward([1,2,3,4])