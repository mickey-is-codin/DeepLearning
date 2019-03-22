import torch
import matplotlib.pyplot as plt

class PersonalNetwork(torch.nn.Module):
    def __init__(self):
        super(PersonalNetwork, self).__init__()

        self.input_dim = 784
        self.hidden_dim = [784, 10]
        self.output_dim = 10

        self.batch_size = 16

        self.linear_1 = torch.nn.Linear(self.input_dim, self.hidden_dim[0])
        self.linear_2 = torch.nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
        self.linear_3 = torch.nn.Linear(self.hidden_dim[1], self.output_dim)

        self.loss_y = []

    def forward(self, x):
        non_linearity = torch.nn.ReLU()

        x = x.view(x.shape[0], x.shape[2] * x.shape[3])

        x = non_linearity(self.linear_1(x))
        x = non_linearity(self.linear_2(x))
        x = self.linear_3(x)

        return x
        #return torch.nn.functional.log_softmax(x, dim=1)

    def train(self, train_loader, optimizer, epoch_num):

        num_batches = len(train_loader)
        print("Num batches in test loader: " + str(num_batches))
        print("Size of each batch: " + str(self.batch_size))

        loss_fn = torch.nn.MSELoss(reduction='sum')

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            #print("Current epoch: %d\tCurrent batch: %d" % (epoch_num, batch))

            optimizer.zero_grad()
            output = self.forward(batch_x)

            batch_y = self.build_batch_y(batch_y, self.batch_size, self.output_dim)

            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

            self.loss_y.append(loss)

            if batch_idx % 100 == 0:
                print("epoch: %d\tbatch: %d\tloss: %d" % (epoch_num, batch_idx, loss))

            if batch_idx == num_batches-1:
                output = self.forward(batch_x)
                print(output)
                print(batch_y)

    @staticmethod
    def build_batch_y(batch_y, batch_size, output_dim):
        y_template = torch.zeros(batch_size, output_dim)

        for target_ix, target in enumerate(batch_y):
            y_template[target_ix, int(target)] = 1

        return(y_template)

    def plot_loss(self, loss):
        x = [val for val in range(0,len(loss))]

        plt.figure()
        plt.plot(x,loss)
        plt.title("Loss over time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.show()
