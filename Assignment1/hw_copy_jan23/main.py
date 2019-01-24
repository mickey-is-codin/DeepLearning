from conv import Conv2D

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL

def main():
    
    # Start timing execution of the entire program
    overall_start = time.time()

    # Load and show input image
    # Include the path to the image that should be convolved here
    provided_image = PIL.Image.open('images/checkerboard.png')

    # Convert image to grayscale
    provided_image_bw = provided_image.convert('L')

    # Load and show personal image
    # Include the path to the image that should be convolved here
    personal_image = PIL.Image.open('images/HTTT.jpg')

    # Convert image to grayscale
    personal_image_bw = personal_image.convert('L')

    #====Task One====#
    print("\nBeginning Task One")
    task_one_start = time.time()
    # Initialize object of class Conv2D for task 1
    conv2d = Conv2D (
            in_channel=1, 
            o_channel=1, 
            kernel_size=3,
            stride=1, 
            mode='known',
            task=1
    )

    # Call the forward method on the checkerboard provided_image
    [num_operations, conv_tensors] = conv2d.forward(provided_image_bw)
    tensors_to_images(conv_tensors, conv2d)

    #====Task Two===#
    print("\nBeginning Task Two")
    task_two_start = time.time()
    # Initialize object of class Conv2D for task 2
    conv2d = Conv2D (
            in_channel=1, 
            o_channel=1, 
            kernel_size=3,
            stride=1, 
            mode='known',
            task=2
    )

    # Call the forward method on the checkerboard provided_image
    [num_operations, conv_tensors] = conv2d.forward(provided_image_bw) 
    tensors_to_images(conv_tensors, conv2d)

    #====Task Three===#
    print("\nBeginning Task Three")
    task_three_start = time.time()
    # Initialize object of class Conv2D for task 3
    conv2d = Conv2D (
            in_channel=1, 
            o_channel=1, 
            kernel_size=3,
            stride=2,  
            mode='known',
            task=3
    )

    # Call the forward method on the checkerboard provided_image
    [num_operations, conv_tensors] = conv2d.forward(provided_image_bw)
    tensors_to_images(conv_tensors, conv2d)
    
    #====Task Four===#
    print("\nBeginning Task Four")
    task_four_start = time.time()
    # Initialize object of class Conv2D for task 4
    conv2d = Conv2D (
            in_channel=1, 
            o_channel=1, 
            kernel_size=5,
            stride=2, 
            mode='known',
            task=4
    )

    # Call the forward method on the checkerboard provided_image
    [num_operations, conv_tensors] = conv2d.forward(provided_image_bw)
    tensors_to_images(conv_tensors, conv2d)
    
    #====Task Five===#
    print("\nBeginning Task Five")
    task_five_start = time.time()
    # Initialize object of class Conv2D for task 5
    conv2d = Conv2D (
            in_channel=1, 
            o_channel=1, 
            kernel_size=5,
            stride=2, 
            mode='known',
            task=5
    )

    # Call the forward method on the checkerboard provided_image
    [num_operations, conv_tensors] = conv2d.forward(provided_image_bw)
    tensors_to_images(conv_tensors, conv2d)

    #====Task Six====# 
    print("\nBeginning Task Six")
    task_six_start = time.time()
    # Initialize object of class Conv2D for task 7
    conv2d = Conv2D (
            in_channel=3, 
            o_channel=1, 
            kernel_size=3,
            stride=1, 
            mode='known',
            task=6
    )

    [num_operations, conv_tensors] = conv2d.forward(personal_image)
    tensors_to_images(conv_tensors, conv2d)

    #====Task Seven====# 
    print("\nBeginning Task Seven")
    task_seven_start = time.time()
    # Initialize object of class Conv2D for task 7
    conv2d = Conv2D (
            in_channel=3, 
            o_channel=2, 
            kernel_size=3,
            stride=1, 
            mode='known',
            task=7
    )

    [num_operations, conv_tensors] = conv2d.forward(personal_image)
    tensors_to_images(conv_tensors, conv2d)

    '''#====Task Eight====#
    print("\nBeginning Task Eight")
    forward_times = []
    task_eight_start = time.time()
    # Initialize object of class Conv2D for task 8
    for i in range(0,10):
        conv2d = Conv2D (
            in_channel=3,
            o_channel=2**i,
            kernel_size=3,
            stride=1,
            mode='rand',
            task=8
        )

        forward_start = time.time()
        [num_operations, conv_tensors] = conv2d.forward(personal_image)
        forward_times.append(time.time() - forward_start)
        total_task_8_time = time.time() - task_eight_start
        print("Convolution forward method finished in %.2f seconds." % (forward_times[-1]))
        print("Convolution total operations: %d" % (sum(num_operations)))
        print("Total time for task 8: %.2f seconds" % (total_task_8_time))
        tensors_to_images(conv_tensors, conv2d)
        if total_task_8_time > 1800:
            break

    plot_task_8(forward_times)

    #====Task Nine====#
    print("\nBeginning Task Nine")
    operation_times = []
    task_nine_start = time.time()
    kernel_sizes = [3, 5, 7, 9, 11]
    operations = [0 for x in kernel_sizes]
    kernel_sizes_plotting = []
    operations_plotting = []
    for n,size in enumerate(kernel_sizes):
        conv2d = Conv2D (
            in_channel=3,
            o_channel=3,
            kernel_size=size,
            stride=1,
            mode='rand',
            task=9
        )

        operation_start = time.time()
        [num_operations, conv_tensors] = conv2d.forward(personal_image)
        operation_times.append(time.time() - operation_start)
        operations[n] = sum(num_operations)
        operations_plotting.append(sum(num_operations))
        kernel_sizes_plotting.append(kernel_sizes[n])
        total_task_9_time = time.time() - task_nine_start
        print("Convolution forward method finished in %.2f seconds." % (operation_times[-1]))
        print("Convolution total operations: %d" % (sum(num_operations)))
        print("Total time for task 9: %.2f seconds" % (total_task_9_time))

        tensors_to_images(conv_tensors, conv2d)
        if total_task_9_time > 1800:
            break
  
    plot_task_9(kernel_sizes_plotting, operations_plotting)'''
 
    print("Homework took %.2f seconds to complete." % (time.time() - overall_start))

def tensors_to_images(tensors, conv2d):

    # Take the returned tensor convolution result and turn it into an image.
    # A user-defined function would be super useful for this but the instructions
    # don't mention any functions other than main being allowed for this file
    for i, tensor in enumerate(tensors):
        # Normalize the convolution output tensor to a 0-1 scale
        tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

        # Convert the tensor to a numpy array and then normalize to 0-255 scale
        numpy_result = tensor.numpy()
        numpy_result = numpy_result * 255

        # Convert the numpy array to a PIL image and then convert it to black and white
        conv_result_image = PIL.Image.fromarray(numpy_result)
        conv_result_image = conv_result_image.convert('RGB')
        
        # Save the resultant image as a png image file
        conv_result_image.save('results/Task_'+str(conv2d.task)+'_Image_Kernel'+str(i+1)+'_'+str(int(time.time()))+'.png', 'PNG')
        print("Convolution image saved as results/"+'Task_'+str(conv2d.task)+'_Image_Kernel'+str(i+1)+'_'+str(int(time.time()))+'.png')

def plot_task_8(forward_times):
    
    x = [2**i for i, time in enumerate(forward_times)]

    plt.figure(figsize=(10,10))
    plt.scatter(x=x,y=forward_times)
    plt.title('Forward Method Run Time')
    plt.xlabel('Number of output channels')
    plt.ylabel('Amount of time to run (seconds)')
    plt.grid(True)
    plt.show()

def plot_task_9(kernel_sizes, num_operations):
   
    plt.figure(figsize=(10,10))
    plt.scatter(x=kernel_sizes,y=num_operations)
    plt.title('Kernel Size Effect on Operation Number')
    plt.xlabel('Kernel Size')
    plt.ylabel('Number of Operations')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
