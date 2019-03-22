import numpy as np

class NeuralNetwork:
    def __init__(self, sizes):
        self.num_layers = len(sizes) - 1
        self.weight = []
        self.bias = []
        self.weight_grad = []
        self.bias_grad = []
        # randomly initalize your weights and biases

    def forward(self, x):
        # perform forward pass
        return x # output of forward pass

    def backward(self, y): # y is the target output
        # calculate MSE loss
        # backpropagate
        # calculate weights gradients
        
    def updateParams(self, eta):
        # update weights based on your learning rate eta
