import torch
import numpy as np

t = np.array([1, 2, 3, 4, 5, 6])

t = torch.Tensor(t)
print(t.shape)