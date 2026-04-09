import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Leer Imagen
digit_0_array_og = cv2.imread("mnist_0.jpg")
digit_1_array_og = cv2.imread("mnist_1.jpg")

digit_0_array_gray = cv2.imread("mnist_0.jpg",cv2.IMREAD_GRAYSCALE)
digit_1_array_gray = cv2.imread("mnist_1.jpg",cv2.IMREAD_GRAYSCALE)

#Ver Imagen
fig, axs=plt.subplot(1,2, figsize=(10,5))

axs[0].imshow(digit_0_array_og, cmap="gray", interpolation="none")
axs[0].set_title("Digit 0 image")
axs[0].axis('off')

axs[1].imshow(digit_1_array_og, cmap="gray", interpolation="none")
axs[1].set_title("Digit 1 image")
axs[1].axis('off')

plt.show()