import torch
import torch.nn as nn
import torch.nn.functional as F

class CCWBlock(nn.Module):
    def __init__(self, c):
        super().__init__()

        # H: cross-resolution weighting
        self.h1 = nn.Conv2d(c * 2, c, 1)
        self.h2 = nn.Conv2d(c, c, 1)

        # F: spatial weighting
        self.fc1 = nn.Linear(c, c // 4)
        self.fc2 = nn.Linear(c // 4, c)

    def H(self, xs):
        x = torch.cat(xs, dim=1)
        x = F.relu(self.h1(x))
        x = torch.sigmoid(self.h2(x))
        return x

    def F(self, x):
        b, c, h, w = x.size()
        s = x.mean(dim=(2,3))
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return s.view(b, c, 1, 1)

    def forward(self, xs):
        x = xs[-1]
        w = self.H(xs) * self.F(x)
        return x * w
