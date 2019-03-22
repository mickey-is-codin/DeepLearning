import torch, torchvision

class ObjectNet(torch.nn.Module):

    def __init__(self, batch_size):
        super(ObjectNet, self).__init__()

        self.input_dim = 32
        self.num_classes = 100
        self.batch_size = batch_size

        # ==Neural network architecture (layer definition)==
        # First convolutional layer takes the 32 by 32 image and performs a conv2d
        # Conv2d(in_channels, out_channels, kernel_size)
        self.conv_1 = torch.nn.Conv2d(3, 6, (5,5))

        # Second convolutional layer convolves the first
        self.conv_2 = torch.nn.Conv2d(6, 16, (5,5))

        # Set up max pooling function
        self.pool = torch.nn.MaxPool2d(2, 2)

        # First linear layer takes 32x32 image and maps it to 100 neurons
        self.linear_1 = torch.nn.Linear(16 * 5 * 5, 100)

        # Second linear layer takes those 100 neurons and maps them to 50
        self.linear_2 = torch.nn.Linear(100, 50)

        # Third layer takes those 50 neurons and maps them to the number of classes.
        self.linear_final = torch.nn.Linear(50, self.num_classes)

        # Initialize our nonlinearity function
        self.relu = torch.nn.ReLU()

    def forward(self, data_batch):

        # First convolution layer passed through nonlinear activation
        x = self.relu(self.conv_1(data_batch))
        x = self.pool(x)

        # Second convolution layer passed through nonlinear activation
        x = self.relu(self.conv_2(x))
        x = self.pool(x)

        # Take the input image and flatten it into a 784 element vector
        x = x.view(-1, self.num_flat_features(x))

        # Pass our resized input through the first linear layer and
        # the nonlinear activation function
        x = self.relu(self.linear_1(x))

        # Now pass x through our second linear layer and again the nonlinearity
        x = self.relu(self.linear_2(x))

        # Final layer we have no activation function
        x = self.linear_final(x)

        # Return the final complete result of the forward pass
        # Shape of x should now be a vector with one element per output class
        return x

    def num_flat_features(self, x):

        # Determine the length of a single vector array mapping an image

        # All dimensions except the batch
        size = x.size()[1:]

        num_features = 1
        for s in size:
            num_features *= s

        return num_features

