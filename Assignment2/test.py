from neural_network import NeuralNetwork
from logic_gates import AND, OR, NOT, XOR
import torch

def main():
	
    # Regular Network Testing
    print("====Regular Network Testing====")
    layer_sizes = [2, 2, 1, 5, 2, 3]
    network_model = NeuralNetwork(layer_sizes)	

    thetas = [network_model.getLayer(x) for x in range(len(layer_sizes)-2)]
    print("Randomly generated thetas (weights): ")
    print(thetas)

    input_array = [42, 42]
    network_result = network_model.forward(input_array)
    print("\nTest of network with input: ")
    print(input_array)
    print(network_result)

    # Logic Gates Testing
    print("\n====Logic Gate Testing====")
    And = AND()
    Or = OR()
    Not = NOT()
    Xor = XOR()

    # Custom Weights
    # Rows = prev layer size
    # Cols = next layer size
    and_thetas = [torch.tensor([
	[-2.0], 
	[ 1.5], 
	[ 1.5]
    ]).float()]
    or_thetas = [torch.tensor([
	[-0.25], 
	[ 1.5], 
	[ 1.5]
    ]).float()]
    not_thetas = [torch.tensor([
	[ 0.0],
	[-1.0]
    ]).float()]
    xor_thetas = [torch.tensor([
	[-0.25, 2.0],
	[ 1.5, -1.5],
	[ 1.5, -1.5]
    ]).float(), torch.tensor([
	[-2.0],
	[ 1.5],
	[ 1.5]
    ]).float()]

    And.getLayer(and_thetas[0], 0)
    Or.getLayer(or_thetas[0], 0)
    Not.getLayer(not_thetas[0], 0)
    Xor.getLayer(xor_thetas[0], 0)
    Xor.getLayer(xor_thetas[1], 1)

    print("AND result: ", And(False, False))
    print("OR result: ", Or(False, True))
    print("NOT result: ", Not(False)) 
    print("XOR result: ", Xor(True, False))	

if __name__ == '__main__':
    main()
