from mnist_network import DigitNet

import torch, torchvision
import visdom
import numpy as np

def main():

    # Create a transformation that we will perform on the dataset that we load in.
    # Since torchvision has raw PIL images we need to convert them to tensors.
    T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Load the dataset itself
    mnist_dataset = torchvision.datasets.MNIST(
        'data/',
        transform=T,
        download=True
    )

    # Create a data_loader wrapper so that we can actually iterate through the
    # dataset that we've loaded in.
    mnist_loader = torch.utils.data.DataLoader(
        mnist_dataset,
        batch_size=32
    )

    # Instantiate our network model
    network_model = DigitNet()

    # Train our network using the MNIST data loader
    train(network_model, mnist_loader, loss_spec="cross")

def train(network_model, mnist_loader, loss_spec="cross"):

    # Initialize the loss function we will be using
    if loss_spec == "mse":
        loss_fn = torch.nn.MSELoss(reduction='sum')
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Isolate the parameters of our network model for inspection
    model_params = network_model.parameters()

    # Instantiate the optimizer we will be using and the parameters it will optimize
    optimizer = torch.optim.SGD(params=model_params, lr=1e-2)

    # Number of times the model will see the whole MNIST dataset
    max_epochs = 3

    # Iterate through epochs and then batches to complete the dataset within each epoch.
    for current_epoch in range(1,max_epochs+1):

        # Instantiate visdom object. We will be using it to make a line plot
        # so we will initialize the line method at point (0,0).
        vis = visdom.Visdom()
        vis_window = vis.line(np.array([0]), np.array([0]))
        loss_x = 0

        for batch_ix, (batch_images, batch_labels) in enumerate(mnist_loader):

            # Convert the data and labels to auto_grad variables so that we can
            # track the partial derivatives associated with them at any time.
            batch_images = torch.autograd.Variable(batch_images)
            batch_labels = torch.autograd.Variable(batch_labels)

            # MSELoss needs very specific shapes for some reason
            if loss_spec == "mse":
                batch_labels = build_batch_labels(network_model, batch_labels)

            # Get the current model's prediction for a batch of input images.
            batch_output = network_model(batch_images)

            # Zero the gradients so we don't have overlap between batches
            network_model.zero_grad()

            # Calculate the loss for the current batch of inputs
            loss = loss_fn(batch_output, batch_labels)

            # Backward pass to compute the gradients of our network model
            loss.backward()

            # Set the optimizer to update all of the model parameters based
            # on our gradients computed from the backward pass.
            optimizer.step()

            # Pass an x and y point to our visdom graph
            vis.line(
                np.array([loss.item()]),np.array([loss_x]),
                win=vis_window,
                update='append',
                opts=dict(title=str(loss_spec))
            )

            # Vis layout commands

            # Increment the loss_x value
            loss_x = loss_x + 1

def build_batch_labels(network_model, batch_labels):

    template = torch.zeros(network_model.batch_size, network_model.num_classes)

    for row_ix, row in enumerate(template):
        template[row_ix, int(batch_labels[row_ix])] = 1

    return template

if __name__ == "__main__":
    main()
