from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR

import torch

and_gate = AND()
or_gate = OR()
not_gate = NOT()
xor_gate = XOR()

and_gate.train()
print("Finished training AND weights")

or_gate.train()
print("Finished training OR weights")

not_gate.train()
print("Finished training NOT weights")

xor_gate.train()
print("Finished training XOR weights")

print("\n")

print("==Calling AND gate==")
and_gate_result = and_gate(torch.tensor([
          [0.0, 0.0, 1.0, 1.0],
          [0.0, 1.0, 0.0, 1.0]
        ]))
print("AND Result: ")
print(and_gate_result)

or_gate_result = or_gate(torch.tensor([
          [0.0, 0.0, 1.0, 1.0],
          [0.0, 1.0, 0.0, 1.0]
        ]))
print("OR Result: ", or_gate_result)

not_gate_result = not_gate(torch.tensor([
          [0.0, 1.0]
        ]))
print("NOT Result: ", not_gate_result)

xor_gate_result = xor_gate(torch.tensor([
          [0.0],
          [0.0]
        ]))
print("XOR Result: ", xor_gate_result)
