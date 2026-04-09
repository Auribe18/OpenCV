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
fig, axs = plt.subplots(1,2, figsize=(10,5))

axs[0].imshow(digit_0_array_og, cmap="gray", interpolation="none")
axs[0].set_title("Digit 0 image")
axs[0].axis('off')

axs[1].imshow(digit_1_array_og, cmap="gray", interpolation="none")
axs[1].set_title("Digit 1 image")
axs[1].axis('off')

plt.show()

print("Forma del arreglo de imagen 0: ",digit_0_array_og.shape)
print(f"Valor mínimo de  pixel:{np.min(digit_0_array_og)}; Valor máximo de pixel:{np.max(digit_0_array_gray)}")

img_tensor_0 = torch.tensor(digit_0_array_og, dtype=torch.float32) /255.0
img_tensor_1 = torch.tensor(digit_1_array_og, dtype=torch.float32) /255.0

print("Forma normalizada del tensor de la imagen 0: ", img_tensor_0)
print(f"Valor mínimo normalizado de pixel: {torch.min(img_tensor_0)} ; Valor máximo normalizado de pixel: {torch.max(img_tensor_0)}")

plt.imshow(img_tensor_0,cmap="gray")
plt.title("Imagen 0 normalizada")
plt.axis("off")
plt.show()