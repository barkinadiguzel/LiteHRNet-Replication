import torch.nn as nn
from ..blocks.ccw_block import CCWBlock
from ..blocks.fusion_block import FusionBlock

class HRNetStage(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ccw = nn.ModuleList([CCWBlock(c) for c in channels])
        self.fusion = FusionBlock()

    def forward(self, xs):
        out = []

        for i, x in enumerate(xs):
            x = self.ccw[i]([x for x in xs])
            out.append(x)

        return self.fusion(out)
