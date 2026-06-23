"""Shared training utilities for the PEFT baselines (paper Sec. 4).

All three baselines (LoRA, BnAdapter, CLIP-Adapter) share the same head
architecture (adapter + linear classifier) and the same training protocol:
Adam, cosine-annealed learning rate, 80 epochs, batch size 64, trained
end-to-end on frozen backbone features -- no backbone parameters are ever
modified.

Extracted and refactored (explicit parameters instead of module-level
``cfg`` global) from the original pipeline, Cell 14.
"""
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..config import set_all_seeds
from .bottleneck_adapter import BnAdapter
from .clip_adapter import CLIPAdapterLayer
from .lora import LoRALayer


class PEFTHead(nn.Module):
    """Adapter + linear classifier, trained end-to-end on frozen features."""

    def __init__(self, adapter: nn.Module, d: int, n_cls: int):
        super().__init__()
        self.adapter = adapter
        self.head = nn.Linear(d, n_cls)

    def forward(self, x):
        return self.head(self.adapter(x))


def _train_peft_model(model, Xtr, ytr, seed, peft_lr=1e-3, peft_epochs=80, peft_batch=64):
    """Shared training loop for all PEFT methods."""
    opt = torch.optim.Adam(model.parameters(), lr=peft_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, peft_epochs)
    crit = nn.CrossEntropyLoss()
    ldr = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=peft_batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    for _ in range(peft_epochs):
        for xb, yb in ldr:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
    return model


def train_peft(
    train_feats,
    train_labels,
    test_feats,
    test_labels,
    adapter_type: str,
    seed: int,
    lora_rank: int = 8,
    adapter_bottle: int = 64,
    peft_lr: float = 1e-3,
    peft_epochs: int = 80,
    peft_batch: int = 64,
) -> float:
    """Train LoRA or BnAdapter baseline and return test accuracy (%)."""
    set_all_seeds(seed)
    scaler = StandardScaler().fit(train_feats)
    Xtr = torch.tensor(scaler.transform(train_feats), dtype=torch.float32)
    Xte = torch.tensor(scaler.transform(test_feats), dtype=torch.float32)
    ytr = torch.tensor(train_labels, dtype=torch.long)
    yte = torch.tensor(test_labels, dtype=torch.long)
    d, n_cls = Xtr.shape[1], int(ytr.max().item()) + 1
    adapter = LoRALayer(d, lora_rank) if adapter_type == "lora" else BnAdapter(d, adapter_bottle)
    model = _train_peft_model(
        PEFTHead(adapter, d, n_cls), Xtr, ytr, seed, peft_lr, peft_epochs, peft_batch
    )
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1).numpy()
    return float((preds == yte.numpy()).mean() * 100)


def train_clip_adapter(
    train_feats,
    train_labels,
    test_feats,
    test_labels,
    seed: int,
    bottleneck: int = 64,
    alpha: float = 0.2,
    peft_lr: float = 1e-3,
    peft_epochs: int = 80,
    peft_batch: int = 64,
) -> float:
    """Train CLIP-Adapter baseline and return test accuracy (%)."""
    set_all_seeds(seed)
    scaler = StandardScaler().fit(train_feats)
    Xtr = torch.tensor(scaler.transform(train_feats), dtype=torch.float32)
    Xte = torch.tensor(scaler.transform(test_feats), dtype=torch.float32)
    ytr = torch.tensor(train_labels, dtype=torch.long)
    yte = torch.tensor(test_labels, dtype=torch.long)
    d, n_cls = Xtr.shape[1], int(ytr.max().item()) + 1
    adapter = CLIPAdapterLayer(d, bottleneck, alpha)
    model = _train_peft_model(
        PEFTHead(adapter, d, n_cls), Xtr, ytr, seed, peft_lr, peft_epochs, peft_batch
    )
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1).numpy()
    return float((preds == yte.numpy()).mean() * 100)
