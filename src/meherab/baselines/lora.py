"""Feature-space LoRA baseline (Hu et al., ICLR 2022), adapted to operate on
frozen backbone features rather than weight matrices (paper Sec. 4).

F_out = F + scale * (F @ W_A^T) @ W_B^T,  rank=8, scale=0.1

Extracted verbatim from the original pipeline, Cell 14.
"""
import torch.nn as nn


class LoRALayer(nn.Module):
    """Feature-space LoRA: F_out = F + scale*(F@W_A.T)@W_B.T"""

    def __init__(self, d: int, rank: int):
        super().__init__()
        self.W_A = nn.Linear(d, rank, bias=False)
        self.W_B = nn.Linear(rank, d, bias=False)
        self.scale = 0.1
        nn.init.normal_(self.W_A.weight, std=0.02)
        nn.init.zeros_(self.W_B.weight)

    def forward(self, x):
        return x + self.scale * self.W_B(self.W_A(x))
