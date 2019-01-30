import numpy as np
import torch
from neural_network import NeuralNetwork

class AND():
    def __init__(self):
        layer_sizes = [2, 1]
        self.network = NeuralNetwork(layer_sizes)

    def __call__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        return self.forward()
    
    def getLayer(self, theta, layer):
        self.network.thetas[layer] = theta
        
    def forward(self):
        input_array = [self.x, self.y]
        result = self.network.forward(input_array)
        
        return bool(result)

class OR():
    def __init__(self):       
        layer_sizes = [2, 1]
        self.network = NeuralNetwork(layer_sizes)

    def __call__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        return self.forward()
    
    def getLayer(self, theta, layer):
        self.network.thetas[layer] = theta
        
    def forward(self):
        input_array = [self.x, self.y]
        result = self.network.forward(input_array)
        
        return bool(result)

class NOT():
    def __init__(self):
        layer_sizes = [2, 1]
        self.network = NeuralNetwork(layer_sizes)

    def __call__(self, x):
        self.x = int(x)
        return self.forward()
    
    def getLayer(self, theta, layer):
        self.network.thetas[layer] = theta
        
    def forward(self):
        input_array = [self.x]
        result = self.network.forward(input_array)
        
        return bool(result)

class XOR():
    def __init__(self):
        layer_sizes = [2, 2, 1]
        self.network = NeuralNetwork(layer_sizes)

    def __call__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        return self.forward()
    
    def getLayer(self, theta, layer):
        self.network.thetas[layer] = theta
        
    def forward(self):
        input_array = [self.x, self.y]
        result = self.network.forward(input_array)
        
        return bool(result)
