"""Feature extraction from the frozen backbone (paper Sec. 3.1-3.2).

Two extraction functions:

* ``extract_proxy`` -- collects the unlabeled proxy set used for CKA,
  FDR, and MDS computation. Proxy size adapts to class count
  (``N_prx = max(base_proxy_n, min(proxy_max_n, proxy_per_class * C))``,
  paper notation: ``N_prx = max(256, min(1024, 8C))``) so every class gets
  at least ``proxy_per_class`` samples -- without this, classes with few
  proxy samples produce unreliable per-node FDR estimates.
* ``extract_split`` -- extracts a stratified train/test split of size
  ``n_take`` via ``StratifiedShuffleSplit(random_state=seed)``. Different
  seeds give genuinely different data splits (not a hyperparameter
  sweep). Handles the ``torch.utils.data.Subset`` label-retrieval edge
  case (e.g. EuroSAT, loaded via ``random_split``) and the small-dataset
  edge case where ``n_take`` exceeds the available samples (e.g. DTD,
  train=1880 < n_train=2000).

Extracted verbatim from the original pipeline, Cell 8.
"""
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit


@torch.no_grad()
def extract_proxy(
    model,
    dataset,
    n_classes: int,
    seed: int,
    device: torch.device,
    base_proxy_n: int = 256,
    proxy_max_n: int = 1024,
    proxy_per_class: int = 8,
    proxy_batch: int = 64,
):
    """Extract the unlabeled proxy set used for CKA/FDR/MDS computation."""
    proxy_n = min(proxy_max_n, proxy_per_class * n_classes)
    proxy_n = max(proxy_n, base_proxy_n)

    g = torch.Generator().manual_seed(seed)
    proxy_n = min(proxy_n, len(dataset))  # guard for small datasets
    idxs = torch.randperm(len(dataset), generator=g)[:proxy_n].tolist()
    ldr = DataLoader(Subset(dataset, idxs), batch_size=proxy_batch, shuffle=False, num_workers=2)

    b_feats: Dict[int, List] = defaultdict(list)
    labels = []
    for imgs, lbls in ldr:
        _, cache = model(imgs.to(device))
        for bid, f in cache.items():
            b_feats[bid].append(f)
        labels.extend(lbls.tolist())

    layer_feats = {k: torch.cat(v).float() for k, v in b_feats.items()}
    return layer_feats, np.array(labels)


@torch.no_grad()
def extract_split(model, dataset, n_take: int, seed: int, device: torch.device):
    """Extract a stratified train/test split of size ``n_take``.

    Returns ``(final_block_features, labels, per_block_features_dict)``.
    """
    # Fast, Subset-aware label retrieval. EuroSAT is loaded via
    # random_split -> Subset(full_27k, slice_indices); reading the inner
    # dataset's .targets directly (length 27k) and feeding those indices
    # to StratifiedShuffleSplit would yield indices outside the Subset's
    # range. We must index .targets by dataset.indices first.
    if hasattr(dataset, "indices"):  # Subset / random_split
        inner = dataset.dataset
        if hasattr(inner, "targets") and inner.targets is not None:
            all_lbl = np.array(inner.targets)[list(dataset.indices)]
        elif hasattr(inner, "_labels") and inner._labels is not None:
            all_lbl = np.array(inner._labels)[list(dataset.indices)]
        else:
            all_lbl = []
            for _, lb in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2):
                all_lbl.extend(lb.tolist())
            all_lbl = np.array(all_lbl)
    elif hasattr(dataset, "targets") and dataset.targets is not None:
        all_lbl = np.array(dataset.targets)
    elif hasattr(dataset, "_labels") and dataset._labels is not None:
        all_lbl = np.array(dataset._labels)
    else:
        all_lbl = []
        for _, lb in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2):
            all_lbl.extend(lb.tolist())
        all_lbl = np.array(all_lbl)

    n_take = min(n_take, len(all_lbl))
    # Edge case: dataset has <= n_take samples (e.g. DTD train=1880,
    # n_train=2000). StratifiedShuffleSplit requires train_size < n_samples
    # strictly. When n_take >= n_samples, use ALL samples in seed-shuffled
    # order instead.
    if n_take >= len(all_lbl):
        rng = np.random.default_rng(seed)
        idxs = rng.permutation(len(all_lbl))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_take, random_state=seed)
        idxs, _ = next(sss.split(np.zeros(len(all_lbl)), all_lbl))

    ldr = DataLoader(Subset(dataset, idxs.tolist()), batch_size=64, shuffle=False, num_workers=2)
    finals = []
    bfeats: Dict[int, List] = defaultdict(list)
    lbls_out = []
    for imgs, lb in ldr:
        out, cache = model(imgs.to(device))
        finals.append(out.numpy())
        for bid, f in cache.items():
            bfeats[bid].append(f.numpy())
        lbls_out.extend(lb.tolist())

    return (
        np.concatenate(finals),
        np.array(lbls_out),
        {k: np.concatenate(v) for k, v in bfeats.items()},
    )
