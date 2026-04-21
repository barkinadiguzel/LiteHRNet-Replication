import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightGenerator(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Conv2d(c, c, 1)

    def forward(self, xs):
        x = torch.cat(xs, dim=1)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x
