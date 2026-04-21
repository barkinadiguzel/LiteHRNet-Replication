import torch.nn as nn
from .conv_bn import ConvBN

class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBN(3, 32, k=3, s=2)

    def forward(self, x):
        return self.conv(x)
