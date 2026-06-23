"""Houlsby bottleneck adapter baseline (Houlsby et al., ICML 2019), adapted
to operate on frozen backbone features (paper Sec. 4).

F_out = F + W_up(GELU(W_down(F))),  bottleneck=64

Extracted verbatim from the original pipeline, Cell 14.
"""
import torch.nn as nn


class BnAdapter(nn.Module):
    """Houlsby bottleneck adapter: F_out = F + W_up(GELU(W_down(F)))"""

    def __init__(self, d: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(d, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))
