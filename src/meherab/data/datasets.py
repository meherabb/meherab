"""Loaders for the ten benchmark datasets used in the paper (Sec. 4,
Appendix B): Food-101, Oxford-Pets, DTD, Aircraft, Flowers102 (torchvision,
official splits), EuroSAT and Caltech-101 (torchvision, seeded 80/20
``random_split``), and RESISC45, PatternNet, UCMerced (``torchgeo``).

The three torchgeo datasets each needed a non-default split strategy to
avoid a class-sorted train/test mismatch (the underlying image lists are
sorted by class, so a naive sequential 80/20 split puts entirely different
classes in train vs. test):

* **RESISC45** -- torchgeo ships a built-in ``split='train'/'test'``
  argument with predefined balanced splits; we use that directly.
* **PatternNet** / **UCMerced** -- no built-in split exists, so we extract
  fast class labels from the file list and run a
  ``StratifiedShuffleSplit`` ourselves to guarantee every class appears in
  both train and test.

Extracted and refactored (paths are now function parameters instead of
hardcoded ``/kaggle/working/...`` globals) from the original pipeline,
Cell 6.
"""
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def default_transforms(img_size: int = 224):
    infer_tf = T.Compose(
        [T.Resize(256), T.CenterCrop(img_size), T.ToTensor(), T.Normalize(_MEAN, _STD)]
    )
    infer_tf_rgb = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
        ]
    )
    return infer_tf, infer_tf_rgb


class _TGSplitW(torch.utils.data.Dataset):
    """Wraps a torchgeo dataset loaded with a built-in split parameter."""

    def __init__(self, tg_ds, transform=None):
        self.ds = tg_ds
        self.transform = transform
        if hasattr(tg_ds, "files") and tg_ds.files:
            try:
                self.targets = [int(f["label"]) for f in tg_ds.files]
                self._labels = self.targets
                return
            except (KeyError, TypeError):
                pass
        n = len(tg_ds)
        n_cls = len(getattr(tg_ds, "classes", []))
        if n_cls > 0 and n % n_cls == 0:
            n_per = n // n_cls
            self.targets = [i // n_per for i in range(n)]
        else:
            print(f"    [labels] iterating {n} items ...")
            self.targets = [int(tg_ds[i]["label"]) for i in range(n)]
        self._labels = self.targets

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        lbl = int(item["label"])
        if self.transform:
            img = self.transform(T.ToPILImage()(img.float()))
        return img, lbl


class _TGSubW(torch.utils.data.Dataset):
    """Wraps a torchgeo dataset with an explicit index list from a
    stratified split.
    """

    def __init__(self, tg_ds, idxs, lbls, tf=None):
        self.ds = tg_ds
        self.idxs = idxs
        self.transform = tf
        self.targets = [lbls[i] for i in idxs]
        self._labels = self.targets

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        item = self.ds[self.idxs[idx]]
        img = item["image"]
        lbl = int(item["label"])
        if self.transform:
            img = self.transform(T.ToPILImage()(img.float()))
        return img, lbl


def _tg_lbl_fast(tg_ds, n_cls):
    """Fast label list for uniform class-sorted torchgeo datasets (no image load)."""
    n = len(tg_ds)
    if n % n_cls == 0:
        return [i // (n // n_cls) for i in range(n)]
    if hasattr(tg_ds, "files") and tg_ds.files:
        try:
            return [int(f["label"]) for f in tg_ds.files]
        except (KeyError, TypeError):
            pass
    print(f"    [labels] iterating {n} items ...")
    return [int(tg_ds[i]["label"]) for i in range(n)]


def load_all_datasets(
    data_root: str = "./data",
    tg_root: str = "./data/torchgeo",
    img_size: int = 224,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Tuple]:
    """Load all ten benchmarks. Returns ``{name: (train_ds, test_ds, n_classes)}``.

    Datasets that fail to load (e.g. a download mirror is temporarily
    unavailable) are skipped with a warning rather than raising -- matching
    the original notebook's resilience to flaky dataset mirrors.
    """
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(tg_root, exist_ok=True)
    infer_tf, infer_tf_rgb = default_transforms(img_size)
    all_datasets: Dict[str, Tuple] = {}

    def log(msg):
        if verbose:
            print(msg)

    log("[Data] Loading 10 datasets ...")

    torchvision_specs = [
        (
            "Food-101",
            lambda: (
                dsets.Food101(data_root, "train", download=True, transform=infer_tf),
                dsets.Food101(data_root, "test", download=True, transform=infer_tf),
                101,
            ),
        ),
        (
            "Oxford-Pets",
            lambda: (
                dsets.OxfordIIITPet(data_root, "trainval", download=True, transform=infer_tf),
                dsets.OxfordIIITPet(data_root, "test", download=True, transform=infer_tf),
                37,
            ),
        ),
        (
            "DTD",
            lambda: (
                dsets.DTD(data_root, "train", download=True, transform=infer_tf),
                dsets.DTD(data_root, "test", download=True, transform=infer_tf),
                47,
            ),
        ),
        (
            "Aircraft",
            lambda: (
                dsets.FGVCAircraft(data_root, "train", download=True, transform=infer_tf),
                dsets.FGVCAircraft(data_root, "test", download=True, transform=infer_tf),
                100,
            ),
        ),
        (
            "Flowers102",
            lambda: (
                dsets.Flowers102(data_root, "train", download=True, transform=infer_tf),
                dsets.Flowers102(data_root, "test", download=True, transform=infer_tf),
                102,
            ),
        ),
    ]
    for name, loader_fn in torchvision_specs:
        try:
            tr, te, nc = loader_fn()
            all_datasets[name] = (tr, te, nc)
            log(f"  [OK] {name:<14} train={len(tr):>7,}  test={len(te):>6,}  cls={nc}")
        except Exception as ex:
            log(f"  [!!] {name:<14} {ex}")

    # EuroSAT -- no official split; seeded 80/20 random_split.
    try:
        full = dsets.EuroSAT(data_root, download=True, transform=infer_tf)
        n_tr = int(0.8 * len(full))
        g = torch.Generator().manual_seed(seed)
        tr, te = torch.utils.data.random_split(full, [n_tr, len(full) - n_tr], generator=g)
        all_datasets["EuroSAT"] = (tr, te, 10)
        log(f"  [OK] EuroSAT         train={len(tr):>7,}  test={len(te):>6,}  cls=10")
    except Exception as ex:
        log(f"  [!!] EuroSAT: {ex}")

    # Caltech-101 -- seeded 80/20 random_split, with .targets patched onto
    # the Subset so downstream stratified extraction works.
    try:
        _cal = dsets.Caltech101(data_root, download=True, transform=infer_tf_rgb)
        n_cal = len(_cal)
        n_tr_c = int(0.8 * n_cal)
        g_c = torch.Generator().manual_seed(seed)
        tr_c, te_c = torch.utils.data.random_split(_cal, [n_tr_c, n_cal - n_tr_c], generator=g_c)
        tr_c.targets = [_cal.y[i] for i in tr_c.indices]
        te_c.targets = [_cal.y[i] for i in te_c.indices]
        tr_c._labels = tr_c.targets
        te_c._labels = te_c.targets
        all_datasets["Caltech-101"] = (tr_c, te_c, 101)
        log(f"  [OK] Caltech-101     train={len(tr_c):>7,}  test={len(te_c):>6,}  cls=101")
    except Exception as ex:
        log(f"  [--] Caltech-101: {ex}")

    # RESISC45 -- torchgeo built-in split (avoids class-sorted train/test mismatch).
    try:
        from torchgeo.datasets import RESISC45 as _TG_R45

        _r45_root = os.path.join(tg_root, "resisc45")
        os.makedirs(_r45_root, exist_ok=True)
        _r45_tr = _TG_R45(_r45_root, split="train", download=True)
        _r45_te = _TG_R45(_r45_root, split="test", download=True)
        all_datasets["RESISC45"] = (
            _TGSplitW(_r45_tr, infer_tf),
            _TGSplitW(_r45_te, infer_tf),
            45,
        )
        log(f"  [OK] RESISC45        train={len(_r45_tr):>7,}  test={len(_r45_te):>6,}  cls=45  [split]")
    except Exception as ex:
        log(f"  [!!] RESISC45: {ex}")

    # PatternNet -- StratifiedShuffleSplit (no built-in split; class-sorted
    # file list means a sequential split would separate classes).
    try:
        from torchgeo.datasets import PatternNet as _TG_PN

        _pn_root = os.path.join(tg_root, "patternnet")
        os.makedirs(_pn_root, exist_ok=True)
        _tg_pn = _TG_PN(_pn_root, download=True)
        _pn_lbl = _tg_lbl_fast(_tg_pn, 38)
        _sss_pn = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
        _tr_i, _te_i = next(_sss_pn.split(np.zeros(len(_pn_lbl)), _pn_lbl))
        all_datasets["PatternNet"] = (
            _TGSubW(_tg_pn, list(_tr_i), _pn_lbl, infer_tf),
            _TGSubW(_tg_pn, list(_te_i), _pn_lbl, infer_tf),
            38,
        )
        log(f"  [OK] PatternNet      train={len(_tr_i):>7,}  test={len(_te_i):>6,}  cls=38  [stratified]")
    except Exception as ex:
        log(f"  [!!] PatternNet: {ex}")

    # UCMerced -- StratifiedShuffleSplit, same reasoning as PatternNet.
    try:
        from torchgeo.datasets import UCMerced as _TG_UCM

        _ucm_root = os.path.join(tg_root, "ucmerced")
        os.makedirs(_ucm_root, exist_ok=True)
        _tg_ucm = _TG_UCM(_ucm_root, download=True)
        _ucm_lbl = _tg_lbl_fast(_tg_ucm, 21)
        _sss_ucm = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
        _tr_i2, _te_i2 = next(_sss_ucm.split(np.zeros(len(_ucm_lbl)), _ucm_lbl))
        all_datasets["UCMerced"] = (
            _TGSubW(_tg_ucm, list(_tr_i2), _ucm_lbl, infer_tf),
            _TGSubW(_tg_ucm, list(_te_i2), _ucm_lbl, infer_tf),
            21,
        )
        log(f"  [OK] UCMerced        train={len(_tr_i2):>7,}  test={len(_te_i2):>6,}  cls=21  [stratified]")
    except Exception as ex:
        log(f"  [!!] UCMerced: {ex}")

    log(f"\n[Data] {len(all_datasets)} / 10 datasets loaded")
    return all_datasets


# Domain labels used for figure coloring (paper Fig. 4, Appendix F.4/F.6/F.7).
# Matches the original pipeline's Cell 4 definition exactly.
DOMAIN_LABEL = {
    "Food-101": "Object/Scene",
    "Oxford-Pets": "Object/Scene",
    "Caltech-101": "Object/Scene",
    "Flowers102": "Object/Scene",
    "DTD": "Texture",
    "Aircraft": "Fine-grained",
    "EuroSAT": "Remote sensing",
    "RESISC45": "Remote sensing",
    "PatternNet": "Remote sensing",
    "UCMerced": "Aerial",
}
