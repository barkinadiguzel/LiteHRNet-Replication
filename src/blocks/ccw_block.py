import torch
import torch.nn as nn
import torch.nn.functional as F


class CCWBlock(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.h1 = nn.Conv2d(c * 2, c, 1)
        self.h2 = nn.Conv2d(c, c, 1)

        self.fc1 = nn.Linear(c, c // 4)
        self.fc2 = nn.Linear(c // 4, c)

    def H(self, xs):
        x = torch.cat(xs, dim=1)
        x = F.relu(self.h1(x))
        return torch.sigmoid(self.h2(x))

    def F(self, x):
        b, c, h, w = x.size()

        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))

        return s.view(b, c, 1, 1)

    def forward(self, xs):
        out = []

        w_h = self.H(xs)

        for i, x in enumerate(xs):
            w_f = self.F(x)
            w = w_h[:, :x.shape[1], :, :] * w_f
            out.append(x * w)

        return out
