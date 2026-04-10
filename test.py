import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.tensor([[2], [2]])
d = a + b
e = d * c

print(d)
print(e)