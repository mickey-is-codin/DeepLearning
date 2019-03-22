from nn_img2num import NNImg2Num

import torch

def main():
    batch_size = 4
    input_dim = 5
    hidden_dims = [15, 15]
    output_dim = 5

    network_model = NNImg2Num(input_dim, hidden_dims, output_dim)

    random_in = torch.randn(batch_size, input_dim)
    random_out = torch.randn(batch_size, output_dim)

    random_out_pred = network_model(random_in)

    network_model.train(random_in, random_out)

    print("Input: ")
    print(random_in)

    print("Initial Predicted Output: ")
    print(random_out_pred)

    print("Desired Output: ")
    print(random_out)

    print("Final Output: ")
    print(network_model(random_in))


if __name__ == "__main__":
    main()
