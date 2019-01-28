from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
tensor_one = torch.tensor([[1,2,3],[4,5,6]])
print(tensor_one)
tensor_two = torch.tensor([[7,8,9],[10,11,12]])
tensor_tre = torch.tensor([[13,14,15],[16,17,18]])
tensor_list = [tensor_one, tensor_two, tensor_tre]
stacked_tensor = torch.stack(tensor_list)
print(stacked_tensor.shape)
stacked_tensor.
#facebook detectron
#in model zoo