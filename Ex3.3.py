import numpy as np

from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import image

from tensorflow import keras
import math

img = image.imread("test2.jpg")

img = img/255


def convolve(input: np.array, kernel_size, depth):
    input_height, input_width,input_depth = input.shape
    output_shape = ( depth, input_height - kernel_size+1, input_width - kernel_size +1)
    kernel_shape = (depth, kernel_size, kernel_size, input_depth)
    kernels = np.random.randn(*kernel_shape)

    output = np.zeros(output_shape)
    for i in range(depth):
        print(kernels[i].shape)
        p= signal.convolve(input, kernels[i], 'valid')
        p = (np.squeeze(p, axis = 2))
        output[i] = p
    return output



def ReLu(input):
    return np.maximum(0, input)


def Pool(input, pool_size, pool_depth):
    input_depth,input_height, input_width = input.shape
    output_height =  math.floor(input_height / pool_size)
    output_width = math.floor(input_width / pool_size)
    output_depth = math.floor(input_depth / pool_depth)
    output_shape = (output_depth, output_height, output_width)
    output = np.zeros(output_shape)
    print(output_shape)
    for i in range(output_depth):
        for j in range(output_height):
            for k in range(output_width):
                
                output[i,j,k] = np.max(input[i*pool_depth : ((i+1)*pool_depth-1),
                                            j*pool_size: ((j+1)*pool_size-1), 
                                            k*pool_size: ((k+1)*pool_size-1)], )
               
               
    return output


def norm(input):
    input_depth, input_height, input_width = input.shape
    output_shape = input.shape
    output = np.zeros(output_shape)

    for x in range(input_depth):
        sd = np.std(input[x])
        mean = np.mean(input[x])
        output[x] = (input[x]- mean) / sd
    return output
        


img2 = convolve(img,5,8)

img2 = ReLu(img2)
img2 = Pool(img2, 5,2)
img2 = norm(img2)

plt.imshow(img2[0,:,:])
plt.show()
