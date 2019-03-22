import torch
import numpy
#import any other packages you might need from pytorch

class MyImg2Num:

        ### Define your model here as in HW3. We reccomend two hidden layers. You should do this homework only with linear layers as explained in class.
        ### Remember the output should be one hot encoding , this means the dimension of your output should be the same as the number of classes





        ### Since we have not explained the pytorch dataloader in class we are giving you an example here for the mnist dataset.
        ### Please read the pytorch documentation on dataloaders.

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=self.batch_size, shuffle=True)
        # Load test data
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
        batch_size=self.test_batch_size, shuffle=False)

    def forward(self, x):
        ###Define your forward pass with your NeuralNetwork class here.



    def train(self):
        while epochs < max_epochs or not stopping_criteria:
            # iterate over whole dataset
                # NeuralNetwork forward pass
                # NeuralNetwork backward pass
                # NeuralNetwork update params
