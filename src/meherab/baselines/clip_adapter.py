"""CLIP-Adapter baseline (Gao et al., 2021), adapted to operate on frozen
backbone features (paper Sec. 4). Post-hoc baseline (not one of the four
planned Bonferroni comparisons).

F_out = alpha * ReLU(W_up(ReLU(W_down(F)))) + (1-alpha) * F,
bottleneck=64, alpha=0.2

Extracted verbatim from the original pipeline, Cell 14.
"""
import torch.nn as nn


class CLIPAdapterLayer(nn.Module):
    """CLIP-Adapter: residual feature adapter.

    F_out = alpha * ReLU(W_up(ReLU(W_down(F)))) + (1-alpha) * F
    """

    def __init__(self, d: int, bottleneck: int = 64, alpha: float = 0.2):
        super().__init__()
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        self.act = nn.ReLU(inplace=True)
        self.alpha = alpha
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        h = self.act(self.down(x))
        h = self.act(self.up(h))
        return self.alpha * h + (1.0 - self.alpha) * x
