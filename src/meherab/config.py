"""All MEHERAB hyperparameters in one place.

Extracted verbatim (logic unchanged) from the original experiment pipeline,
Cell 5. Every tuneable value used anywhere in the method lives in this single
dataclass -- there are no magic numbers elsewhere in the package. Any result
in the paper can be reproduced by constructing one ``MEHERABConfig`` and
threading it through the pipeline.

See ``configs/default.yaml`` for the same values in YAML form, and
``docs/PAPER_MAP.md`` for the equation/section each field corresponds to.
"""
import os
import random
from dataclasses import dataclass, fields

import numpy as np
import torch


@dataclass
class MEHERABConfig:
    # Backbone
    backbone: str = "vit_base_patch16_224"
    img_size: int = 224

    # Proxy set (Sec. 3.2). proxy_n scales with class count so every class
    # gets at least `proxy_per_class` samples in the unlabeled proxy set
    # (N_prx = max(256, min(1024, 8*C)) in the paper).
    base_proxy_n: int = 256
    proxy_max_n: int = 1024
    proxy_per_class: int = 8
    proxy_batch: int = 64

    # Homophilic graph (Sec. 3.2)
    n_clusters: int = 4
    # Hyperedge criterion: joint FDR gain must exceed this delta for a
    # multi-node hyperedge to be selected over the single highest-FDR pair.
    hyperedge_delta: float = 0.005

    # RASS operations (Sec. 3.3)
    adapter_ranks: tuple = (4, 8, 16)
    sc_keep_ratios: tuple = (0.5, 0.75, 1.0)
    adapter_scale: float = 0.3
    n_ops: int = 3

    # Modality Discriminability Score (Sec. 3.4, Eq. 5-7)
    mds_alpha: float = 0.5

    # Evolutionary search / Algorithm 1 (Sec. 3.5)
    evo_pop: int = 20
    evo_gens: int = 15
    evo_mutation: float = 0.3
    evo_elite: int = 5
    evo_tournament: int = 3

    # Evaluation protocol (Sec. 4). Probe regularisation C is selected by an
    # inner 3-fold GridSearchCV per seed -- never tuned against the test set.
    n_train: int = 2000
    n_test: int = 1000
    pca_dim: int = 128
    probe_C_grid: tuple = (0.1, 0.5, 1.0, 2.0, 5.0)
    n_seeds: int = 5
    eval_max_iter: int = 1000

    # PEFT baselines (Sec. 4) -- LoRA, bottleneck adapter, CLIP-Adapter
    lora_rank: int = 8
    adapter_bottle: int = 64
    peft_lr: float = 1e-3
    peft_epochs: int = 80
    peft_batch: int = 64

    # MDS proxy-validation experiment (Sec. 5.2)
    n_corr_cands: int = 50


def print_config(cfg: MEHERABConfig) -> None:
    """Pretty-print a config, matching the original notebook's startup banner."""
    col_w = max(len(f.name) for f in fields(cfg)) + 2
    print("=" * 52)
    print("  MEHERAB Configuration")
    print("=" * 52)
    for f in fields(cfg):
        print(f"  {f.name:<{col_w}}{str(getattr(cfg, f.name)):<30}")
    print("=" * 52)


# Reproducibility seeds (Cell 3). Five independent seeds, each driving a
# distinct StratifiedShuffleSplit of the data -- genuine replication across
# data splits, not a hyperparameter sweep masquerading as seeds.
GLOBAL_SEED = 42
EVAL_SEEDS = [42, 1337, 2024, 314159, 99991]


def set_all_seeds(seed: int) -> None:
    """Seed every RNG used anywhere in the pipeline (Cell 3)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

