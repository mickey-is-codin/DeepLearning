import numpy as np
import matplotlib.pyplot as plt
import torch

class NeuralNetwork:
    def __init__(self, sizes, print_results=False):
        self.sizes = sizes
        self.num_layers = len(sizes) - 1
        self.weight = []
        self.bias = []
        self.weight_grad = []
        self.bias_grad = []

        self.print_results = print_results

        # randomly initalize your weights and biases
        for layer_ix in range(1, len(sizes)):
            prev_layer_size = sizes[layer_ix-1]
            next_layer_size = sizes[layer_ix]
            std_dev = 1 / (np.sqrt(sizes[layer_ix-1]+1))
            if layer_ix < (len(sizes)-1):
                random_matrix = torch.rand(next_layer_size + 1, prev_layer_size + 1, )
            else:
                random_matrix = torch.rand(next_layer_size, prev_layer_size + 1, )
            self.weight.append(std_dev * random_matrix)
            print("Initialized weights: ")
            print(self.weight)

            if (self.print_results):
                print("\nRandom weight matrix mapping layer " + str(layer_ix)  + \
                " of size " + str(prev_layer_size) + " to layer " + str(layer_ix+1) + \
                " of size " + str(next_layer_size) + ": ")

                print(self.weight[-1])

    def forward(self, x):

        # Class attribute for our input
        self.x = x

        #print(self.x)

        # List that will store each column of the input's forward pass.
        self.forward_result_list = []

        # Iterate through each column of the input x tensor. This will be our
        # batch operation where we get essentially one forward pass per each
        # column of the input.
        for c, column in enumerate(x[0,:]):

            first_layer_no_bias = x[:,c].view(len(x[:,c]),1)

            self.z = []
            self.a = []
            self.a_hat = []

            self.z.append(first_layer_no_bias)
            self.a.append(first_layer_no_bias)

            bias = torch.ones(1, 1)

            self.a_hat.append(torch.cat([bias, self.a[0]], 0))

            for i in range(0, self.num_layers):
                self.z.append(torch.mm(self.weight[i], self.a_hat[i]))
                self.a_hat.append(self._activation_function(self.z, activation="sigmoid"))

            #print(self.a_hat[-1])
            self.forward_result_tensor = self.a_hat[-1]

            #print("\nFull Forward a Matrix: ")
            #print(self.a_hat)

            if (self.print_results):
                print("\nNumber of layers including input: %d" % (self.num_layers+1))
                for size in self.sizes:
                    print(size)
                print("Number of a vectors: %d" % (len(self.a)))
                for a_layer in self.a:
                    print(a_layer.shape)
                print("Number of z vectors: %d" % (len(self.z)))
                for z_layer in self.z:
                    print(z_layer.shape)
                print("Number of weight matrices: %d" % (len(self.weight)))
                for current_weight in self.weight:
                    print(current_weight.shape)

            self.forward_result_list.append(self.a_hat[-1])

        return self.forward_result_list

    @staticmethod
    def _activation_function(z, activation="sigmoid"):

        if activation == "sigmoid":
            # Sigmoid Nonlinearity activation function
            return(1 / (1 + np.exp((-1) * z[-1])))

    def backward(self, y): # y is the target output

        self.y = y
        original_x = self.x
        original_forward_results = self.forward_result_list
        original_results_tensor = torch.stack(self.forward_result_list).view(1,len(self.forward_result_list))
        original_errors = self.y - original_results_tensor
        errors = original_errors
        squared_errors = torch.mul(errors, errors)
        mse = (1.0 / 2.0) * torch.mean(squared_errors)
        self.sum_mse = torch.sum(mse)

        self.corrected_results = original_results_tensor

        back_pass_count = [0 for x in y[0,:]]
        self.MSE_error_graph_points = [[] for x in y[0,:]]

        while self.sum_mse > 0.01:

            for c, column in enumerate(y[0,:]):
                x_tensor = original_x[:,c].view(len(original_x[:,c]),1)
                y_tensor = y[:,c].view(len(y[:,c]),1)
                self.forward(x_tensor)

                #print(x_tensor)
                #print(y_tensor)

                #if c == 3:
                #    print(self.corrected_results[0,c])

                back_pass_count[c] = back_pass_count[c] + 1

                errors[0,c] = abs(y_tensor - self.corrected_results[0,c])
                #if c == 0:
                    # print("\nY: ")
                    # print(y_tensor)
                    # print("Output: ")
                    # print(self.corrected_results[0,c])
                    # print("Error: ")
                    # print(errors[0,c])
                    # print("A_hat: ")
                    # print(self.a_hat)
                squared_errors = torch.mul(errors, errors)
                mse = (1.0 / 2.0) * squared_errors
                self.sum_mse = torch.sum(mse)
                #print("Sum Errors: %.9f" % self.sum_mse)

                self.MSE_error_graph_points[c].append(float(self.sum_mse))

                # backpropagate
                dE_dh = self.corrected_results[0,c] - y_tensor
                last_sigmoid = torch.mul(self.corrected_results[0,c], (1.0 - self.corrected_results[0,c]))
                last_layer_delta = torch.mul(dE_dh, last_sigmoid)

                deltas = [last_layer_delta]
                self.weight_grad = [0]

                backwards_weights = self.weight[::-1]
                backwards_a = self.a_hat[::-1]
                #if c == 0:
                #   print(self.a_hat)
                for layer in range(1,len(backwards_a)):

                    a_layer = backwards_a[layer]
                    weight_layer = backwards_weights[layer-1]
                    weight_layer_T = torch.t(weight_layer)

                    sigmoid_prime_layer = torch.mul(a_layer, (1.0 - a_layer))

                    weight_by_delta = torch.mm(weight_layer_T, deltas[0])

                    delta_layer = torch.mul(weight_by_delta, sigmoid_prime_layer)
                    deltas.insert(0, delta_layer)

                    weight_effect = torch.mm(a_layer, torch.t(deltas[-(layer)]))

                    self.weight_grad = torch.t(weight_effect)

                    if (self.print_results):
                        print("\n==NEW LAYER "+str(layer-1)+"==")
                        print("Current a_hat shape: [%dx%d]" % (a_layer.shape[0], a_layer.shape[1]))
                        print("Current weight shape: [%dx%d]" % (weight_layer.shape[0], weight_layer.shape[1]))
                        print("Current transposed weight shape: [%dx%d]" % (weight_layer_T.shape[0], weight_layer_T.shape[1]))
                        print("Current sigmoid prime shape: [%dx%d]" % (sigmoid_prime_layer.shape[0], sigmoid_prime_layer.shape[1]))
                        print("Shape of (transposed) mm (next delta): [%dx%d]" % (weight_by_delta.shape[0], weight_by_delta.shape[1]))
                        print("Current layer delta shape: [%dx%d]" % (delta_layer.shape[0], delta_layer.shape[1]))
                        print("\nCalculated shape of dE_dw for this layer: [%dx%d]" % (weight_effect.shape[0], weight_effect.shape[1]))

                learning_rate = 0.1
                self.update_params(learning_rate)
                current_forward_result = self.forward(x_tensor)

                self.corrected_results[0,c] = current_forward_result[0]

        self.graph_error(back_pass_count)

        return

    def update_params(self, eta):
        for w, current_weight in enumerate(self.weight):
            self.weight[w] = self.weight[w] - eta * self.weight_grad[w]

    def graph_error(self, back_pass_count):

        plt.figure(figsize=(10,10))

        for c, column in enumerate(self.y[0,:]):
            plt.semilogy(range(0,back_pass_count[c]), self.MSE_error_graph_points[c], label="batch "+str(c+1)+" error")

        plt.title("Error Convergence")
        plt.xlabel("Number of backwards passes")
        plt.ylabel("Sum of squared errors")
        plt.legend()

        plt.show()
