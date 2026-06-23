"""Frozen ViT with per-block CLS-token hooks (paper Sec. 3.1).

Wraps any ``timm`` vision transformer, registering a forward hook on every
transformer block that caches the CLS token (``out[:, 0, :]``). Used to
extract the per-block feature matrices ``F_l`` that everything else in
MEHERAB (CKA, the homophilic graph, RASS operations) operates on. The
backbone itself is never modified -- every parameter is frozen, and no
gradient ever touches it anywhere in the pipeline.

``**model_kwargs`` are forwarded to ``timm.create_model`` so callers can
pass e.g. ``img_size=224`` for DINOv2-B (native resolution 518x518; timm
interpolates the position embeddings) -- used in the backbone-
generalisation experiments (paper Sec. 5.3, Appendix C).

Extracted verbatim from the original pipeline, Cell 7.
"""
from typing import Dict, List

import timm
import torch
import torch.nn as nn


class MEHERABBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, **model_kwargs):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **model_kwargs)
        self.embed_dim = self.vit.embed_dim
        self.n_blocks = len(self.vit.blocks)
        self._cache: Dict[int, torch.Tensor] = {}
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self):
        for i, blk in enumerate(self.vit.blocks):
            def _make_hook(idx):
                def _hook(m, inp, out):
                    self._cache[idx] = out[:, 0, :].detach().cpu()

                return _hook

            self._hooks.append(blk.register_forward_hook(_make_hook(i)))

    @torch.no_grad()
    def forward(self, x):
        self._cache.clear()
        final = self.vit(x)
        return final.detach().cpu(), dict(self._cache)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def load_frozen_backbone(model_name: str, device: torch.device, **model_kwargs) -> MEHERABBackbone:
    """Load a backbone, move it to ``device``, freeze every parameter, and
    set it to eval mode -- exactly the setup used for every result in the
    paper (no backbone parameter is ever updated, anywhere, for any method
    including the PEFT baselines, which only train an adapter head).
    """
    backbone = MEHERABBackbone(model_name, pretrained=True, **model_kwargs).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone
