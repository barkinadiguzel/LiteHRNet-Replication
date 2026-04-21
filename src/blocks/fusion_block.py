import torch.nn.functional as F

class FusionBlock:
    def forward(self, xs):

        target_h, target_w = xs[0].shape[2:]

        resized = []
        for x in xs:
            if x.shape[2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w),
                                  mode="bilinear",
                                  align_corners=False)
            resized.append(x)

        fused = sum(resized)

        return [fused for _ in xs]
