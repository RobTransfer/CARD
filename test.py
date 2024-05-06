import torch
a = torch.rand([1024,1,10,1])
print(a,a[1])
a = torch.rand(a.shape)
print(a.shape)