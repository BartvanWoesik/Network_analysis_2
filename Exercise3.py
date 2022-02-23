
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from tensorflow import keras
import ssl

(x_train, y_train), (x_test, y_test) =keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255, x_test/255

img = x_train[4]
plt.imshow(img)
plt.show()

print(img.size)

def plot_image(img: np.array):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray');

plot_image(img)


kernel1 = np.array( [[1,0,1],
                    [1,0,1],
                    [1,0,1]])



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

