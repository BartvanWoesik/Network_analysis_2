import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import image
from tensorflow import keras
import math

img = image.imread("test2.jpg")

img = img/255

# Question 20
def convolve(input: np.array, kernel_size, depth):
    input_height, input_width,input_depth = input.shape
    output_shape = ( depth, input_height - kernel_size+1, input_width - kernel_size +1)
    kernel_shape = (depth, kernel_size, kernel_size, input_depth)
    kernels = np.random.randn(*kernel_shape)

    output = np.zeros(output_shape)
    for i in range(depth):
            kernel = kernels[i]
            
            for j in range(input_height - kernel_size +1):
                for k in range( input_width- kernel_size+1):
                 
                    
                    
       
                    input_part = input[j:(j + kernel_size),
                                        k:(k+kernel_size),:]
                  
                    output[i,j,k]  = np.sum(np.multiply(kernel, input_part))
                
                    
           
                    
    return output


# Question 21
def ReLu(input):
    return np.maximum(0, input)

# Question 22
def Pool(input, pool_size, pool_depth):
    input_depth,input_height, input_width = input.shape
    output_height =  math.floor(input_height / pool_size)
    output_width = math.floor(input_width / pool_size)
    output_depth = math.floor(input_depth / pool_depth)
    output_shape = (output_depth, output_height, output_width)
    output = np.zeros(output_shape)

    for i in range(output_depth):
        for j in range(output_height):
            for k in range(output_width):
                
                output[i,j,k] = np.max(input[i*pool_depth : ((i+1)*pool_depth-1),
                                            j*pool_size: ((j+1)*pool_size-1), 
                                            k*pool_size: ((k+1)*pool_size-1)], )
               
               
    return output

# Question 23
def norm(input):
    input_depth, input_height, input_width = input.shape
    output_shape = input.shape
    output = np.zeros(output_shape)

    for x in range(input_depth):
        sd = np.std(input[x])
        mean = np.mean(input[x])
        output[x] = (input[x]- mean) / sd
    return output
        

# Question 24
def full(input, nodes):
    input_depth, input_height, input_width = input.shape
    flatten_size = input_depth* input_height*input_width
    flatten = input.reshape(1, flatten_size)
    weight_size = ( flatten_size , nodes)
    weights = np.random.randn(*weight_size)

    output = np.dot(flatten, weights)
    return output.reshape(nodes)


# Question 25
def softmax(input):
    input_size = input.shape
    output = np.zeros(input_size)

    output = np.exp(input ) 

    output = output / output.sum()
    return output

    

img2 = convolve(img,5,15)
plt.imshow(img2[0,:,:])
plt.show()

img2 = ReLu(img2)
plt.imshow(img2[0,:,:])
plt.show()

img2 = Pool(img2, 3,3)
plt.imshow(img2[0,:,:])
plt.show()

img2 = norm(img2)
plt.imshow(img2[0,:,:])
plt.show()

img2 = full(img2, 10)
img2 = softmax(img2)
print("The outcome is:")
print(img2)
print(img2.sum())
