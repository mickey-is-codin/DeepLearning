from cifar10_network import ObjectNet

import torch, torchvision
import visdom

import numpy as np
import matplotlib.pyplot as plt

def main():

    # Create a transformation that we will perform on the dataset that we load in.
    # Since torchvision has raw PIL images we need to convert them to tensors.
    T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Load the dataset itself
    cifar_train = torchvision.datasets.CIFAR10(
        'data/',
        transform=T,
        download=True,
        train=True
    )
    cifar_test = torchvision.datasets.CIFAR10(
        'data/',
        transform=T,
        download=True,
        train=False
    )

    batch_size = 64

    # Define the classes that exist in the CIFAR dataset
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    # Create a data_loader wrapper so that we can actually iterate through the
    # dataset that we've loaded in.
    train_loader = torch.utils.data.DataLoader(
        cifar_train,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        cifar_test,
        batch_size=batch_size,
        shuffle=True
    )

    # Instantiate our network model
    network_model = ObjectNet(batch_size)

    # Train our network using the CIFAR data loader
    train(
        network_model,
        train_loader,
        test_loader,
        loss_spec="cross",
        plot_engine="visdom",
        optim="sgd"
    )

    test(network_model, test_loader, classes)

def train(network_model, cifar_loader, test_loader, loss_spec="cross", plot_engine="visdom", optim="sgd"):

    # Initialize the loss function we will be using
    if loss_spec == "mse":
        loss_fn = torch.nn.MSELoss(reduction='sum')
    elif loss_spec == "cross":
        loss_fn = torch.nn.CrossEntropyLoss()

    # Isolate the parameters of our network model for inspection
    model_params = network_model.parameters()

    # Instantiate the optimizer we will be using and the parameters it will optimize
    learning_rate = 1e-2
    if optim == "sgd":
        optimizer = torch.optim.SGD(
            params=model_params,
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,
            dampening=0
        )
    elif optim == "adam":
        optimizer = torch.optim.Adam(
            params=model_params,
            lr=learning_rate
        )

    # Number of times the model will see the whole CIFAR dataset
    max_epochs = 3

    if plot_engine == "matplotlib":
        plot_loss_x = []
        plot_loss_y = []
        mat_loss_x = 0

    # Iterate through epochs and then batches to complete the dataset within each epoch.
    for current_epoch in range(1,max_epochs+1):

        if plot_engine == "visdom":
            # Instantiate visdom object. We will be using it to make a line plot
            # so we will initialize the line method at point (0,0).
            vis = visdom.Visdom()
            vis_window = vis.line(np.array([0]), np.array([0]))
            vis_loss_x = 0

        for batch_ix, (batch_images, batch_labels) in enumerate(cifar_loader):

            # Convert the data and labels to auto_grad variables so that we can
            # track the partial derivatives associated with them at any time.
            batch_images = torch.autograd.Variable(batch_images)
            batch_labels = torch.autograd.Variable(batch_labels)

            # MSELoss needs very specific shapes for some reason
            if loss_spec == "mse":
                batch_labels = build_batch_labels(network_model, batch_labels)
            #batch_labels = build_batch_labels(network_model, batch_labels)

            # Get the current model's prediction for a batch of input images.
            batch_output = network_model(batch_images)

            # Zero the gradients so we don't have overlap between batches
            network_model.zero_grad()

            # Calculate the loss for the current batch of inputs
            loss = loss_fn(batch_output, batch_labels)

            # Debugging prints
            print("current epoch: %d\tcurrent batch: %d\tloss: %.4f" % (current_epoch, batch_ix, loss.item()))

            # Backward pass to compute the gradients of our network model
            loss.backward()

            # Set the optimizer to update all of the model parameters based
            # on our gradients computed from the backward pass.
            optimizer.step()

            if plot_engine == "visdom":
                # Pass an x and y point to our visdom graph
                vis.line(
                    np.array([loss.item()]),np.array([vis_loss_x]),
                    win=vis_window,
                    update='append',
                    opts=dict(title=str(loss_spec))
                )
                vis_loss_x = vis_loss_x + 1
            elif plot_engine == "matplotlib":
                plot_loss_x.append(mat_loss_x)
                plot_loss_y.append(loss.item())
                print("Samples seen: %d" % (mat_loss_x))
                mat_loss_x = mat_loss_x + 1

        # Test the network after the training for this epoch
        guess_random_image(network_model, test_loader)

    if plot_engine == "matplotlib":
        mat_plot_loss(plot_loss_x, plot_loss_y, loss_spec, max_epochs)

def test(network_model, test_loader, classes):

    class_correct = [0 for x in range(10)]
    class_total = [0 for x in range(10)]

    for data in test_loader:
        images, labels = data
        outputs = network_model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def build_batch_labels(network_model, batch_labels):

    template = torch.zeros(len(batch_labels), network_model.num_classes)

    for row_ix, row in enumerate(template):
        template[row_ix, int(batch_labels[row_ix])] = 1

    return template

def guess_random_image(network_model, test_loader):
    test_images, test_labels = next(iter(test_loader))

    np_img = test_images[0].numpy()

    plt.figure()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.axis("off")
    plt.show()

    test_images = torch.autograd.Variable(test_images)
    test_labels = torch.autograd.Variable(test_labels)

    output = network_model(test_images)
    print(output)

def mat_plot_loss(x, y, loss_spec, max_epochs):
    fig = plt.figure()
    plt.plot(x, y)
    plt.title("Network Training over "+str(max_epochs)+" Epochs(loss_spec = "+loss_spec+")")
    plt.xlabel("Number of samples seen by network")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
