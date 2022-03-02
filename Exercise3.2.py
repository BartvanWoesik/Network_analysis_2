
from multiprocessing import pool
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import image

from tensorflow import keras

import math

(x_train, y_train), (x_test, y_test) =keras.datasets.cifar10.load_data()
x_train, x_test = x_train /255, x_test/255

img = image.imread("test2.jpg")

img = img/255




kernel1 = np.array( [[-1,-1,-1],
                    [-1,8,-1],
                    [-1,-1,-1]],
                   )
kernel2 = np.array( [[-1,0,1,1],
                    [-2,0,2,1],
                    [-1,0,1,1],
                    [1,1,1,1]],
                   )


kernel3 = np.array([kernel1, kernel2])
print(kernel3[1])
print(kernel3.shape[0])



# QUESTION 20
def convolve2(img: np.array, kernel: np.array) -> np.array:

    kernel_num      = kernel.shape[0]
    
    img_height      = img.shape[0]
    img_width       = img.shape[1]

    # find out the colors of the image
    try:
        img_depth       = img.shape[2]
    except:
        img_depth = 1

    
    final_shape = []

  
    for p in range(0, kernel_num):
        kern = kernel[p]
        print(kern)
        print("go")
        kernel_height   = kern.shape[0] 
        kernel_width    = kern.shape[1]

        mis_match_width = math.floor(kernel_width/2)
        mis_match_height = math.floor(kernel_height/2)

        fit_width = img_width - 2*mis_match_width
        fit_height= img_height - 2*mis_match_height

        new_img = np.zeros(shape=( fit_height,fit_width, img_depth, kernel_num))
        for x in range(0, img_depth):
            for j in range( fit_width- mis_match_width):
            # Iterate over the columns
                for i in range( fit_height - mis_match_height):
                    # img[i, j] = individual pixel value
                    # Get the current matrix
                    mat = img[i:i+kernel_width, j:j+kernel_height, x]
                    
                    # Apply the convolution - element-wise multiplication and summation of the result
                    # Store the result to i-th row and j-th column of our convolved_img array
                    new_img[i+mis_match_height, j+ mis_match_width,x] = np.sum(np.multiply(mat, kern))
        final_shape.append(new_img)
    
    final_shape = np.array(final_shape)

    return final_shape


new_img = convolve2(img, kernel3)

print(new_img[1])


# QUESTION 21
def rel(img: np.array) -> np.array:
    num_features = img.shape[0]
    final_shape = []
    for i in range(num_features):
        new_img = img[1]
        new_img[new_img <0] = 0
        final_shape.append(new_img)
    return new_img

# QUESTION 22
def max_pooling(img: np.array, pool_height, pool_width) -> np.array:
    img_width = img.shape[1]
    img_height = img.shape[0]

    # find out the colors of the image
    try:
        img_depth       = img.shape[2]
    except:
        img_depth = 1

    fit_width = int(math.floor(img_width / pool_width))
    fit_height = int(math.floor(img_height / pool_height))


    # Find the dimensions of the boundries where the pooling size does not fit
    left_widtch = img_width % pool_width
    left_height = img_height % pool_height


  
    new_img = np.zeros(shape=( fit_height,fit_width, img_depth))

    # Go through all whole pooling fit areas of the image
    for x in range(img_depth):
        for j in range(fit_width):
            for i in range(fit_height):
                new_img[i, j , x] = np.max(img[i*pool_height:( (i+1)*pool_height-1), j*pool_height: ((j+1)*pool_width-1), x])
    

    return new_img




new_img = rel(new_img)
new_img= max_pooling(new_img, 2,2)
plt.imshow(new_img)
plt.show()
print(new_img.shape)
