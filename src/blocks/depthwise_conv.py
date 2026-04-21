import torch.nn as nn

class DepthwiseConv(nn.Module):
    def __init__(self, channels, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, k, s, p, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv(x))
