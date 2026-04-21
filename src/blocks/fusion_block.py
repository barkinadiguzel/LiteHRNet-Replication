import torch.nn as nn
import torch.nn.functional as F

class FusionBlock(nn.Module):
    def forward(self, xs):
        out = []
        target_h, target_w = xs[0].shape[2:]

        for x in xs:
            if x.shape[2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            out.append(x)

        return [sum(out)]
