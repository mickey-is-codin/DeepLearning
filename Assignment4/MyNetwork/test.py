#from custom_network import CustomNetwork
from multiple_layer import MultipleLayer

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def main():

    # Max number of times we will run through every input and output combination
    max_epochs = 500

    # The number of inputs should be number of batches * batch size
    # This is the number of inputs we will feed to the network at a time.
    # The number of batches should be the number of times the weights and biases
    # are updated for our network. So for every batch we should run one
    # forward/backward/update iteration.
    batch_size = 4
    num_batches = 2
    num_inputs = batch_size * num_batches

    # Load MNIST data
    load_dataset()

    # Define the number of nodes in each layer of our network
    input_dim = 2
    hidden_dims = [3, 4, 5]
    output_dim = 2

    # Instantiate a network. This should build all of the layer sizes for our
    # network as well as initialize random values for all of the weights
    # and biases of the network.
    network_model = MultipleLayer(input_dim, hidden_dims, output_dim)

    # Create our "dataset". Should have an x and y that are the same
    # amount of rows and a set number of batches for each of those
    # rows.
    #x = torch.randn((batch_size * num_batches, input_dim))
    #y = torch.ones((batch_size * num_batches, output_dim))

    mean_losses_graph = []

    # Batch iteration
    batch_num = 0
    epoch_num = 0

    # One epoch is the number of times we train our network for every single
    # input and outtput combination.
    while epoch_num < max_epochs:

        #print("\n==Epoch number: " + str(epoch_num))

        # One batch is one iteration of updated weights and biases for our network.
        while batch_num < num_batches:

            # Set the inputs and outputs for this training iteration.
            # Remember, train/update parameters once per batch
            current_batch_x = x[(batch_num*batch_size):((batch_num*batch_size)+batch_size),:]
            current_batch_y = y[(batch_num*batch_size):((batch_num*batch_size)+batch_size),:]

            # Now train the network once per batch size
            #print("==Batch number: " + str(batch_num))
            mean_losses = network_model.train(current_batch_x, current_batch_y)

            for loss in mean_losses:
                print(loss)
                mean_losses_graph.append(loss)

            batch_num += 1

        # After finishing going through all batches, zero the number of batches
        # and increment the epoch we're on.
        batch_num  = 0
        epoch_num += 1

    plot_loss(mean_losses_graph)

def get_dataset():
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    example_data.shape

def plot_loss(y):
    x = range(0,len(y))
    plt.figure()
    plt.plot(x,y)
    plt.title("Mean Loss")
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Loss Value")
    plt.show()

if __name__ == "__main__":
    main()
