import torch


y = torch.Tensor(1, 88)  # placeholder
x = torch.Tensor(1, 88)  # placeholder

x = torch.cat((x, y), 1) # concatenate x with object detection and past action values

print(x)