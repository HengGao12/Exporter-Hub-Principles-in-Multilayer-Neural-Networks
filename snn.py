import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, encoding, surrogate, layer, functional

# This part of code is modified from https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/lif_fc_mnist.py

class SNN(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer1 = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 1200, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            )
        
        self.layer2 = nn.Sequential(
            layer.Flatten(),
            layer.Linear(1200, 1200, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            )
        
        self.layer3 = nn.Sequential(
            layer.Flatten(),
            layer.Linear(1200, 1200, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            )
        
        self.layer4 = nn.Sequential(
            layer.Flatten(),
            layer.Linear(1200, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            )

    def forward(self, x: torch.Tensor):
        self.x1 = self.layer1(x)
        self.x2 = self.layer2(self.x1)
        self.x3 = self.layer3(self.x2)
        self.out = self.layer4(self.x3)
        return self.out