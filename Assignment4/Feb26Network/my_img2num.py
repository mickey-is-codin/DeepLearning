import torch
import numpy as np
import matplotlib.pyplot as plt

class MyImg2Num(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.layer_sizes = []
        self.layer_sizes.insert(0, input_dim)
        self.layer_sizes.extend(hidden_dims)
        self.layer_sizes.insert(len(self.layer_sizes), output_dim)

        print("Layer sizes: ", self.layer_sizes)

        self.num_layers = len(self.layer_sizes)

        self.w = []
        self.b = []

        for layer_ix in range(0,len(self.layer_sizes) - 1):

            current_weight = torch.randn(self.layer_sizes[layer_ix], self.layer_sizes[layer_ix+1])
            self.w.append(current_weight)

            current_bias = torch.randn((1, self.layer_sizes[layer_ix+1]))
            self.b.append(current_bias)

    @staticmethod
    def sigmoid_activation(z):
        return(1 / (1 + torch.exp(-z)))

    def forward(self, input_batch):
        batch_size = len(input_batch[:,0])

        # Goal for z_list/a_list:
        #       -a/z_list[batch_number][layer_number]
        self.batch_z_list = []
        self.batch_a_list = []

        for which_input, single_input in enumerate(input_batch):

            single_input = single_input.view(1, len(single_input))

            self.z = []
            self.a = [single_input]

            for w_ix, weight in enumerate(self.w):
                current_z = torch.mm(self.a[w_ix], weight) + self.b[w_ix]
                self.z.append(current_z)

                current_a = self.sigmoid_activation(self.z[-1])

                self.a.append(current_a)

            self.batch_z_list.append(self.z)
            self.batch_a_list.append(self.a)

        output_dim = self.batch_a_list[0][-1].shape[1]

        all_outputs = [a[-1] for a in self.batch_a_list]
        all_outputs_tensor = torch.zeros(batch_size, output_dim)
        for r, row in enumerate(all_outputs_tensor):
            all_outputs_tensor[r] = all_outputs[r]

        return(all_outputs_tensor)

    @staticmethod
    def sigmoid_delta(x):
        return(x * (1 - x))

    def backward(self, predicted_output_batch, labels_batch):
        batch_size = len(labels_batch[:,0])

        self.batch_deltas_list = []
        self.batch_dE_dw_list = []

        for which_label, single_label in enumerate(labels_batch):
            # Initialize the deltas with the last layer delta
            last_delta = (predicted_output_batch[which_label] - single_label) * self.sigmoid_delta(self.batch_a_list[which_label][-1])
            self.deltas = [last_delta]

            self.dE_dw = []

            # Iterate backwards through the layer of the network appending to deltas
            # This builds it backwards
            for layer_ix in range(0,len(self.batch_a_list[which_label]) - 1):
                current_a = self.batch_a_list[which_label][len(self.batch_a_list[which_label]) - layer_ix - 2]
                current_w = self.w[len(self.w) - layer_ix - 1]

                # Deltas = change in cost wrt weighted inputs
                self.deltas.append(torch.mm(self.deltas[-1], current_w.t()) * self.sigmoid_delta(current_a))
                # dE_dw = change in error wrt weights
                self.dE_dw.append(current_a.t() * (self.deltas[-2]))

            # Reverse the deltas
            self.deltas = self.deltas[::-1]
            self.dE_dw = self.dE_dw[::-1]

            self.deltas = self.deltas[1:len(self.deltas)]

            self.batch_deltas_list.append(self.deltas)
            self.batch_dE_dw_list.append(self.dE_dw)

    def update_params(self, learning_rate, batch_size):

        for input_number in range(batch_size):
            #print("Updateing weights for input/output number: " +str(input_number))
            for w_ix in range(0,len(self.w)):
                #print(self.w[w_ix].shape)
                #print(self.dE_dw[w_ix].shape)
                self.w[w_ix] -= learning_rate * (self.dE_dw[w_ix])
                self.b[w_ix] -= learning_rate * (self.deltas[w_ix])

    def train(self, input_batch, output_batch):

        batch_size = len(input_batch[:,0])

        # Training session algorithm:
            # -Forward pass
            # -Backward pass8
            # -Update parameters

        learning_rate = 0.1

        #print("Size of input to forward method ([batch_size, input_size]): " + str(input_batch.shape))
        forward_pass_results = self.forward(input_batch)
        #print("Shape of all of the outputs from that forward pass: " + str(forward_pass_results.shape))

        loss = output_batch - forward_pass_results
        mean_losses = [row.mean() for row in loss]

        self.backward(forward_pass_results, output_batch)
        self.update_params(learning_rate, batch_size)

        return(mean_losses)

