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

batch_tensor = torch.stack([img_tensor_0, img_tensor_1])
print("Forma del batch de tensores en formato tensorflow: ", batch_tensor.shape)

batch_input = batch_tensor.permute(0,3,1,2)
print("Forma del batch de tensores en formato Pytorch: ", batch_input.shape)

#Crear tensor unidimensional que esté relleno de 1
a = torch.ones(5)
print(a)

#Crear tensor unidimensional relleno de 0
b = torch.zeros(5)
print(b)

#Crear tensor unidimensional con valores customizados
c = torch.tensor([1.0,2.0,3.0,4.0,5.0])
print(c)

#Crear tensor con más dimensiones
d = torch.ones(3,2)
print(d)
e = torch.zeros(3,2)
print(e)
f = torch.tensor([[1,2],[3,4]])
print(f)
#Crear Torch 3D
g = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
print(g)

print(d.shape)
print(e.shape)
print(f.shape)
print(g.shape)

#Acceder a elementos dentro de un tensor
#Acceder a elemento en índice 2

print(c[2])

#Acceder a elementos dentro de un tensor de más de 1 dimension

print (f[1,0]) #Fila 1, columna 0
print(g[1,0,0])

# Todos los elementos
print(f[:])

# Todos los elementos del index 1 al 2, excluyendo el 3
print(c[1:3])

#Todos los elementos hasta el indice 4, excluyéndolo
print(c[:4])

# Primera Columna
print(f[0, :])

# Segunda Columna
print(f[:,1])

#Tipo de dato del tensor
int_tensor = torch.tensor([[1,2,3],[4,5,6]])
print(int_tensor.dtype)

# Si se cambia cualquier elemento a float32, entonces el tipo de dato del tensor completo cambia
int_tensor = torch.tensor([[1,2,3],[4.,5,6]])
print(int_tensor.dtype)
print(int_tensor)

# El cambio de tipo de dato se puede anular de la siguiente forma
float_tensor = torch.tensor([[1, 2, 3],[4., 5, 6]])
int_tensor = float_tensor.type(torch.int64)
print(int_tensor.dtype)
print(int_tensor)

# De tensor a arreglo
f_numpy = f.numpy()
print(f_numpy)

# De arreglo a tensor
h = np.array([[8,7,6,5],[4,3,2,1]])
h_tensor = torch.from_numpy(h)
print(h_tensor)

# Operaciones aritméticas en tensores
#Se crean los tensores
tensor1 = torch.tensor([[1,2,3],[4,5,6]])
tensor2 = torch.tensor([[-1,2,-3],[4,-5,6]])

# Suma
print(tensor1+tensor2)
# También se puede usar esta forma
print(torch.add(tensor1,tensor2))

# Resta
print(tensor1-tensor2)
# También
print(torch.sub(tensor1,tensor2))

# Multiplicación
# Tensor con Scalar
print(tensor1 * 2)

# Tensor con otro tensor
# Múltiplicación por elementos
print(tensor1 * tensor2)

# Múltiplicación de matríces
tensor3 = torch.tensor([[1,2],[3,4],[5,6]])
print(torch.mm(tensor1,tensor3))

# División
# Tensor con Scalar
print(tensor1/2)

# Tensor con otro tensor
# División por elementos
print(tensor1/tensor2)

#Broadcasting
# Se crean dos tensores de 1 dimension
a = torch.tensor([1, 2, 3])
b = torch.tensor([4])

# Se suma un scalar a un vector
result = a + b

print("Resultado del Broadcast:\n",result)

#El broadcast permite que PyTorch realice operaciones por elemento 

# Crear dos tensores con forma (1, 3) y (3, 1)
a = torch.tensor([[1, 2, 3]])
b = torch.tensor([[4], [5], [6]])

# Sumar tensores con distintas formas
result = a + b
print("Forma: ", result.shape)
print("\n")
print("Resultado del broadcast:\n", result)

# Crear tensores para CPU
# Esto utiliza la RAM del sistema
tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cpu')

# Crear tensores para GPU
# Esto ocupa la ram de la GPU
#tensor_gpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')

# CPU RAM
tensor_cpu = tensor_cpu * 5

# GPU RAM
# Enfoque en consumo de ram de gpu
#tensor_gpu = tensor_gpu * 5