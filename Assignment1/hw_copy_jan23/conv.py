import time
import math
import random

import torch
import torchvision
from torch.nn import functional as F
import PIL

import numpy as np

class Conv2D(object):

    def __init__(self, in_channel, o_channel, kernel_size, stride, mode, task):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.task = task

    def forward(self, input_image):

        # Set up kernels
        kernels = []

        K1 = torch.tensor([
            [-1, -1, -1], 
            [ 0, 0, 0], 
            [1, 1, 1]
        ])
        K2 = torch.tensor([
            [-1, 0, 1], 
            [-1, 0, 1], 
            [-1, 0, 1]
        ])
        K3 = torch.tensor([
            [1, 1, 1], 
            [1, 1, 1], 
            [1, 1, 1]
        ])
        K4 = torch.tensor([
            [-1, -1, -1, -1, -1], 
            [-1, -1, -1, -1, -1], 
            [0, 0, 0, 0, 0], 
            [1, 1, 1, 1, 1], 
            [1, 1, 1, 1, 1]
        ])
        K5 = torch.tensor([
            [-1, -1, 0, 1, 1], 
            [-1, -1, 0, 1, 1], 
            [-1, -1, 0, 1, 1], 
            [-1, -1, 0, 1, 1], 
            [-1, -1, 0, 1, 1]
        ])
        K6 = torch.tensor([
            [-1, -1, -1],
            [ 0,  0,  0], 
            [ 1,  1,  1]
        ])
        K7 = torch.tensor([
             [-1, -1, -1],
             [-1,  8, -1],  
             [-1, -1, -1]
        ])
        K8 = torch.tensor([
            [-1, 0, -1],
            [-2, 0,  2],
            [-1, 0,  1]
        ])
        if self.task == 1:
            kernels.append(K1)
        elif self.task == 2:
            kernels.append(K2)
        elif self.task == 3:
            kernels.append(K3)
        elif self.task == 4:
            kernels.append(K4)
        elif self.task == 5:
            kernels.append(K5)
        elif self.task == 6:
            kernels.append(K6)
        elif self.task == 7:
            kernels.append(K7)
            kernels.append(K8)
       
        num_kernels = self.o_channel

        if self.mode == 'rand':
            for i in range(0,num_kernels):
                kernels.append(torch.randint(-10, 10, (self.kernel_size,self.kernel_size)))

        # Convert image object to tensor
        image_array = np.array(input_image)
        image_tensor = torch.from_numpy(image_array)
        dimensions_list = list(image_tensor.shape)
       
        padding = int((self.kernel_size) / 2)

        if self.in_channel > 1:
            padded_tensor = torch.zeros(dimensions_list[0] + padding, dimensions_list[1] + padding, dimensions_list[2])
            padded_tensor[1:dimensions_list[0]+1, 1:dimensions_list[1]+1, :] = image_tensor
        else:
            padded_tensor = torch.zeros(dimensions_list[0] + padding, dimensions_list[1] + padding)
            padded_tensor[1:dimensions_list[0]+1, 1:dimensions_list[1]+1] = image_tensor

        num_rows = dimensions_list[0]
        num_cols = dimensions_list[1]

        output_rows = int((num_rows - self.kernel_size + 2*padding) / self.stride + 1)
        output_cols = int((num_cols - self.kernel_size + 2*padding) / self.stride + 1)

        print("Input image resolution: %dx%d.\n" % (num_rows, num_cols), end="", flush=True)

        # Setting up the output array
        output_tensors = [torch.zeros(output_rows, output_cols) for x in kernels]
        num_operations = [0 for x in kernels]

        for i in range(0,num_kernels):
            print("\nRandomly generated kernel: ")
            print(kernels[i])
            print("\nCurrent kernel number: %d" % (i+1))
            for channel in range(0,self.in_channel):
                print("Current input channel: %d" % (channel+1))
                half_kernel = math.floor(self.kernel_size / 2)
                # Iterate through each row of the image (outer loop -- y)
                row_out = 0
                for row in range(half_kernel, num_rows-half_kernel, self.stride):
                    # Iterate through each column of the image (inner loop -- x)
                    col_out = 0
                    for col in range(half_kernel, num_cols-half_kernel, self.stride): 
                        num_operations[i] = num_operations[i] + self.kernel_size + self.kernel_size-1

                        if self.in_channel > 1:
                            region_of_interest = padded_tensor[row-half_kernel:row+half_kernel+1, col-half_kernel:col+half_kernel+1, channel]    
                        else:
                            region_of_interest = padded_tensor[row-half_kernel:row+half_kernel+1, col-half_kernel:col+half_kernel+1]    
                        
                        region_of_interest = region_of_interest.double()
                        kernel = kernels[i].double()

                        output_tensors[i][row_out, col_out] = output_tensors[i][row_out, col_out] + torch.tensordot(region_of_interest, kernel)
                        
                        col_out = col_out + 1
                    row_out = row_out + 1

        output_tensors = [torch.clamp(output_tensor, min=0, max=255) for output_tensor in output_tensors]

        return [num_operations, output_tensors]
