import torch
from neural_network import NeuralNetwork

def main():

  x = torch.tensor([
    [0.45, 0.65],
    [0.25, 0.75]
  ])

  y = torch.tensor([
    [0.01, 0.99],
    [0.01, 0.99]
  ])

  sizes = [len(x[:,0]), 3, 3, len(y[:,0])]

  network_model = NeuralNetwork(sizes, print_results=False)

  x_forward_result = network_model.forward(x)
  network_model.backward(y)

if __name__ == "__main__":
  main()
