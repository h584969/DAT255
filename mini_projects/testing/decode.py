import torch

file: dict[str, torch.Tensor] = torch.load("models/xor.pth")



for x in file:
    print (x)
    print(file[x])