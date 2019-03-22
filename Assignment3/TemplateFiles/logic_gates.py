import numpy as np
from neural_network import NeuralNetwork

class AND():
    def __init__(self):	
        # initialize neural network for this gate        
    
    def train(self):
        inputs = [np.float32([[0.0, 0.0]]), np.float32([[0.0, 1.0]]), np.float32([[1.0, 0.0]]), np.float32([[1.0, 1.0]])]
        labels = [np.float32([[0.0]]), np.float32([[0.0]]), np.float32([[0.0]]), np.float32([[1.0]])]
        
        # call forward of your neural network
        # call backward of your neural network
        # update parameters
        # make sure to do the above for multiple iterations so that you get the best results

# do similar for other logic gates

