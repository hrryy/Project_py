import torch
import numpy as np
from PIL import Image

print("==== Image Read ====")
img = Image.open("dataset/test/Mild Impairment/1 (2).jpg") #pythong list
print("==== Image Convert to numpy array ====")
img = np.array(img)                             #np.array
print(img)
print("==== Image Convert to Tensor ====")
tensor_img = torch.tensor(img)
print(tensor_img)
print(tensor_img.shape)

# add /reduce dimenstion 
tensor_img_add_dim = tensor_img.unsqueeze(0)
print(tensor_img_add_dim.shape)

#tensor_img_add_Dim = tensor_img_add_dim.squeeze()


#repeat
tensor_img_add_dim = tensor_img_add_dim.repeat(1,3,1,1) # [1,3,128,128]
print(tensor_img_add_dim.shape)

 