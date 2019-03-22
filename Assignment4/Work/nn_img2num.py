import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#import any other packages you might need from pytorch

class NNImg2Num(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(NNImg2Num, self).__init__() #initialize super class

        ### Define your model here in pytorch. We reccomend two hidden layers.
        ### You should do this homework only with linear layers as explained in class.
        ### Remember the output should be one hot encoding,
        ### this means the dimension of your output should be the same
        ### as the number of classes

        self.linear_1 = nn.Linear(input_dim, hidden_dims[0])
        self.linear_2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear_3 = nn.Linear(hidden_dims[1], output_dim)


        ### Since we have not explained the pytorch dataloader in class we are giving you an example here for the mnist dataset.
        ### Please read the pytorch documentation on dataloaders.

        # self.train_loader = torch.utils.data.DataLoader(
        #     datasets.MNIST('../data', train=True, download=True,
        #                    transform=transforms.Compose([
        #                        transforms.ToTensor()
        #                    ])),
        #     batch_size=self.batch_size, shuffle=True)
        # # Load test data
        # self.test_loader = torch.utils.data.DataLoader(
        #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
        #                        transforms.ToTensor()
        #                    ])),
        # batch_size=self.test_batch_size, shuffle=False)

    def forward(self, x):
        ### Define your forward pass with pytorch here.
        ### For any resizing you need to do look at pytorch's view function.
        non_linearity = nn.Sigmoid()
        x = non_linearity(self.linear_1(x))
        x = non_linearity(self.linear_2(x))
        x = non_linearity(self.linear_3(x))

        return(x)


    def train(self, forward_input, labels):

        epochs_graph = []
        loss_graph = []

        epochs = 0
        max_epochs = 100000
        learning_rate = 1e-4
        stopping_criterion = torch.nn.MSELoss(reduction='sum')
        #optimizer = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)

        while epochs < max_epochs or not stopping_criterion:
            predicted_output = self(forward_input)

            loss = stopping_criterion(predicted_output, labels)
            print(epochs, loss.item())

            self.zero_grad()

            #optimizer.zero_grad()
            loss.backward()
            #optimizer.step()

            with torch.no_grad():
                for param in self.parameters():
                    param.data -= learning_rate * param.grad

            epochs_graph.append(epochs)
            loss_graph.append(loss)

            epochs += 1

        self.plot_loss(epochs_graph, loss_graph)

    def plot_loss(self, x, y):
        plt.figure(figsize=(10,10))

        plt.plot(x, y)
        plt.title("Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")

        plt.show()

