import numpy as np

class NN:

    def __init__(self, topology):
        
        self.logsig = (lambda x: 1 / (1 + np.e** (-x)),lambda x: x * (1 - x))
        self.relu = lambda x: np.maximum(0, x)
        self.functionCost = (lambda Yp, Yr: np.mean((Yp - Yr)** 2), lambda Yp, Yr: (Yp - Yr))

        self.buildNeuralNetwork(topology, self.logsig)

    def buildNeuralNetwork(self, topology, activation_function):
        self.neural_network = []
        for index, layer in enumerate(topology[:-1]):
            self.neural_network.append(Neural_Layer(topology[index], topology[index+1], activation_function))
        
    def trainingNeuralNetwork(self, X, Y, learning_factor = 0.1, training = True):
        self.out = [(None, X)]
        self.forwarPass(Y)
        
        if training:
            self.backwarPass(Y, learning_factor)
        
        return self.out[-1][1]

    def forwarPass(self, Y):
        for l, layer in enumerate(self.neural_network):
            weighted_sum = self.out[-1][1] @ self.neural_network[l].weight + self.neural_network[l].bias
            activation_function = self.neural_network[l].activation_function[0](weighted_sum)
            self.out.append((weighted_sum, activation_function))
        # print(funcionCoste[0](out[-1][1], Y))

    def backwarPass(self, Y, learning_factor):
        deltas = []
            
        for layer in reversed(range(0, len(self.neural_network))):
            weighted_sum = self.out[layer+1][0]
            activation_function = self.out[layer+1][1]
            
            if layer == len(self.neural_network)-1:
                deltas.insert(0, self.functionCost[1](activation_function, Y) * self.neural_network[layer].activation_function[1](activation_function))
            else:
                deltas.insert(0, deltas[0] @ _weight.T * self.neural_network[layer].activation_function[1](activation_function))
            
            _weight = self.neural_network[layer].weight

            self.neural_network[layer].bias = self.neural_network[layer].bias - np.mean(deltas[0], axis=0, keepdims=True) * learning_factor
            self.neural_network[layer].weight = self.neural_network[layer].weight - self.out[layer][1].T  @ deltas[0] * learning_factor
        pass

    def testingNeuralNetwork(self):
        pass


class Neural_Layer():

    def __init__(self, nConnections, nNeurons, activation_function):
    
        self.activation_function = activation_function
        self.bias = np.random.rand(1, nNeurons) * 2 - 1 
        self.weight = np.random.rand(nConnections, nNeurons) * 2 - 1 