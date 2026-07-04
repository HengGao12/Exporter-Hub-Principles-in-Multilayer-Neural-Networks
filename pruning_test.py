"""
    In this file, we provide a simple example of first layer pruning using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import ipdb

# define a sequential model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)

print(model[0].weight.data.shape) # torch.Size([20, 10])

# assume that we are going to prune the first layer in the model
module = model[0]

# Using pruning tools in PyTorch
prune.l1_unstructured(module, 'weight', amount=0.2)

# print the size of the weight in the first layer
print(module.weight.data.shape)  # torch.Size([20, 10])