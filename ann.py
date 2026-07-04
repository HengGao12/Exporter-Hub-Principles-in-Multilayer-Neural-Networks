import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(784, 1200)
        self.dropout1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(1200, 1200)
        self.dropout2 = nn.Dropout(0.1)
        self.layer3 = nn.Linear(1200, 1200)
        self.dropout3 = nn.Dropout(0.1)
        self.layer4 = nn.Linear(1200, 10)

    def forward(self, x):
        self.x = x.view(-1, 784)
        self.x1 = F.relu(self.layer1(self.x))
        self.l1 = self.dropout1(self.x1)
        self.x2 = F.relu(self.layer2(self.l1))
        self.l2 = self.dropout2(self.x2)
        self.x3 = F.relu(self.layer3(self.l2))
        self.l3 = self.dropout3(self.x3)
        self.out = self.layer4(self.l3) 
        return self.out