import torch


y = torch.Tensor(1, 88)  # placeholder
x = torch.Tensor(1, 88)  # placeholder

x = torch.cat((x, y), 1) # concatenate x with object detection and past action values


a=torch.Tensor(32,1,88)
b=a.view(32, 88)

print(b.size())