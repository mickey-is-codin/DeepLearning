import numpy as np
import torch

class NeuralNetwork():
    def __init__(self, layer_sizes):
        
        # Creating variables for the sizes of each layer
        self.layer_sizes = layer_sizes
        self.in_size = self.layer_sizes[0]
        self.out_size = self.layer_sizes[-1]
        self.hidden_sizes = self.layer_sizes[1:-1]
        self.num_layers = len(self.layer_sizes)
        
        # Build the weights for our network.
        # The layout for the weights will be a list of matrices for now.
        # The shape of a given weight item in the list will be [from layer size, to layer size]
        self.thetas = []
        for layer_ix in range(1, len(layer_sizes)):
            self.thetas.append((1/(np.sqrt(layer_sizes[layer_ix-1]+1))) * torch.rand(layer_sizes[layer_ix-1]+1, layer_sizes[layer_ix]))
            
    def getLayer(self, layer):
        return self.thetas[layer]
        
    def forward(self, input_list):

        z = []
        a = []
        
        first_layer_no_bias = torch.FloatTensor(input_list).view(1, len(input_list))
        
        z.append(first_layer_no_bias)
        a.append(z[0])
                        
        for i in range(0,self.num_layers-1):
            bias = torch.ones(1, 1)
            layer_with_bias = torch.cat([bias, a[i]], 1)
          
            #print("Layer %d 'in' size: %dx%d" % (i, layer_with_bias.shape[0], layer_with_bias.shape[1]))        
            #print("Layer %d theta size: %dx%d" % (i, self.thetas[i].shape[0], self.thetas[i].shape[1]))
            
            z.append(torch.mm(layer_with_bias, self.thetas[i]))
            a.append(1/(1+np.exp(-z[-1])))
            
            #print("Layer %d result size: %dx%d" % (i, a[-1].shape[0], a[-1].shape[1]))

        result = a[-1][0][0]
        
        # If binary output desired:
        if result >= 0.5:
            result = 1.0
        elif result < 0.5:
            result = 0.0
        
        return result
