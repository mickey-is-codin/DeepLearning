from neural_network import NeuralNetwork

import torch

class AND():
    def __init__(self):
        # initialize neural network for this gate
        sizes = [2, 1]
        self.network = NeuralNetwork(sizes, print_results=False)

    def __call__(self, inputs):
      return(self.network.forward(inputs))

    def train(self):
        inputs = torch.tensor([
          [0.0, 0.0, 1.0, 1.0],
          [0.0, 1.0, 0.0, 1.0]
        ])

        labels = torch.tensor([
          [0.0, 0.0, 0.0, 1.0],
        ])

        print("\n==Training AND Gate==")

        x_forward_result = self.network.forward(inputs)
        #print(x_forward_result)
        self.network.backward(labels)
        print("Final weights: ")
        print(self.network.weight)

        # call forward of your neural network
        # call backward of your neural network
        # update parameters
        # make sure to do the above for multiple iterations so that you get the best results

class OR():
    def __init__(self):
        # initialize neural network for this gate
        sizes = [2, 1]
        self.network = NeuralNetwork(sizes, print_results=False)

    def __call__(self, inputs):
      return(self.network.forward(inputs))

    def train(self):
        inputs = torch.tensor([
          [0.0, 0.0, 1.0, 1.0],
          [0.0, 1.0, 0.0, 1.0]
        ])

        labels = torch.tensor([
          [0.0, 1.0, 1.0, 1.0],
        ])

        print("\nTraining OR Gate")
        x_forward_result = self.network.forward(inputs)
        self.network.backward(labels)

        # call forward of your neural network
        # call backward of your neural network
        # update parameters
        # make sure to do the above for multiple iterations so that you get the best results

class NOT():
    def __init__(self):
        # initialize neural network for this gate
        sizes = [1, 1]
        self.network = NeuralNetwork(sizes, print_results=False)

    def __call__(self, inputs):
      return(self.network.forward(inputs))

    def train(self):
        inputs = torch.tensor([
          [0.0, 1.0],
        ])

        labels = torch.tensor([
          [1.0, 0.0],
        ])

        print("\nTraining NOT Gate")
        x_forward_result = self.network.forward(inputs)
        self.network.backward(labels)

        # call forward of your neural network
        # call backward of your neural network
        # update parameters
        # make sure to do the above for multiple iterations so that you get the best results

class XOR():
    def __init__(self):
        # initialize neural network for this gate
        sizes = [2, 2, 1]
        self.network = NeuralNetwork(sizes, print_results=False)

    def __call__(self, inputs):
      return(self.network.forward(inputs))

    def train(self):
        inputs = torch.tensor([
          [0.0, 0.0, 1.0, 1.0],
          [0.0, 1.0, 0.0, 1.0]
        ])

        labels = torch.tensor([
          [0.0, 1.0, 1.0, 0.0],
        ])

        print("\nTraining XOR Gate")
        x_forward_result = self.network.forward(inputs)
        self.network.backward(labels)

        # call forward of your neural network
        # call backward of your neural network
        # update parameters
        # make sure to do the above for multiple iterations so that you get the best results
