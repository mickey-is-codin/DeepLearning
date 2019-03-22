import torch, torchvision

class DigitNet(torch.nn.Module):

    def __init__(self):
        super(DigitNet, self).__init__()

        self.input_dim = 28
        self.num_classes = 10
        self.batch_size = 32

        # ==Neural network architecture (layer definition)==
        # First linear layer takes 28x28 image and maps it to 100 neurons
        self.linear_1 = torch.nn.Linear(self.input_dim*self.input_dim, 100)

        # Second linear layer takes those 100 neurons and maps them to 50
        self.linear_2 = torch.nn.Linear(100, 50)

        # Third layer takes those 50 neurons and maps them to the number of classes.
        self.linear_final = torch.nn.Linear(50, self.num_classes)

        # Initialize our nonlinearity function
        self.relu = torch.nn.ReLU()

    def forward(self, data_batch):

        # Take the input image and flatten it into a 784 element vector
        x = data_batch.view(-1, self.input_dim * self.input_dim)

        # Pass x through our resized input through the first linear layer and
        # the nonlinear activation function
        x = self.relu(self.linear_1(x))

        # Now pass x through our second linear layer and again the nonlinearity
        x = self.relu(self.linear_2(x))

        # Final layer we have no activation function
        x = self.linear_final(x)

        # Return the final complete result of the forward pass
        # Shape of x should now be a vector with one element per output class
        return x
