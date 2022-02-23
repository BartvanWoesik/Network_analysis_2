
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from tensorflow import keras

import math

(x_train, y_train), (x_test, y_test) =keras.datasets.cifar10.load_data()
x_train, x_test = x_train /255, x_test/255

img = x_train[4]

img = np.array(img)





plt.imshow(img)
plt.show()



kernel1 = np.array( [[-1,-1,-1],
                    [0,0,0,],
                    [1,1,1,]],
                   )



def calculate_target_size(img_size: int, kernel_size: int) -> int:
    num_pixels = 0
    
    # From 0 up to img size (if img size = 224, then up to 223)
    for i in range(img_size):
        # Add the kernel size (let's say 3) to the current i
        added = i + kernel_size
        # It must be lower than the image size
        if added <= img_size:
            # Increment if so
            num_pixels += 1
            
    return num_pixels





def convolve(img: np.array, kernel: np.array) -> np.array:
    # Assuming a rectangular image
    tgt_size = calculate_target_size(
        img_size=img.shape[0],
        kernel_size=kernel.shape[0]
    )
    # To simplify things
    k = kernel.shape[0]
    
    # 2D array of zeros
    convolved_img = np.zeros(shape=(tgt_size, tgt_size))
    
    # Iterate over the rows
    for i in range(tgt_size):
        # Iterate over the columns
        for j in range(tgt_size):
            # img[i, j] = individual pixel value
            # Get the current matrix
            mat = img[i:i+k, j:j+k]
            
            # Apply the convolution - element-wise multiplication and summation of the result
            # Store the result to i-th row and j-th column of our convolved_img array
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))
            
    return convolved_img

img2 = np.zeros(shape=(30,30,3))
for i in range(img.shape[2]):

    img2[:,:,i] = convolve(img[:,:,i], kernel1)
plt.imshow(img2)
plt.show()


