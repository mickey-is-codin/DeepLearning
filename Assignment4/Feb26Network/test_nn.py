from nn_img2num import PersonalNetwork

import torch
import torchvision

def main():

    max_epochs = 3

    batch_size = 16

    personal_network = PersonalNetwork()

    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=personal_network.batch_size, shuffle=True)

    learning_rate = 1e-4
    optimizer = torch.optim.SGD(personal_network.parameters(), lr=learning_rate)

    for epoch in range(0,max_epochs):
        personal_network.train(train_loader, optimizer, epoch)

    personal_network.plot_loss(personal_network.loss_y)

if __name__ == "__main__":
    main()
