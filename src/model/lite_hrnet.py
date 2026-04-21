import torch.nn as nn
import torch.nn.functional as F

from ..blocks.stem import Stem
from ..modules.hrnet_stage import HRNetStage


class LiteHRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.stem = Stem()

        self.stage2 = HRNetStage(cfg.STAGE_CHANNELS[0])
        self.stage3 = HRNetStage(cfg.STAGE_CHANNELS[1])
        self.stage4 = HRNetStage(cfg.STAGE_CHANNELS[2])

    def forward(self, x):

        x = self.stem(x)

        xs = [x, F.avg_pool2d(x, 2)]
        xs = self.stage2(xs)

        xs = xs + [F.avg_pool2d(xs[-1], 2)]
        xs = self.stage3(xs)

        xs = xs + [F.avg_pool2d(xs[-1], 2)]
        xs = self.stage4(xs)

        return xs
