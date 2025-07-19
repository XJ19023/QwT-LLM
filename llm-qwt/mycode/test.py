import torch

tensor = torch.empty(2, 4, device='cpu')
print(tensor)

del tensor
