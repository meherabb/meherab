"""NASWOT-adapted and SynFlow-adapted proxy scores (paper Sec. 5.2).

Competing zero-gradient proxies used only for the proxy-validation
experiment (Fig. 3 / Table 2) -- they rank the same RASS candidates as MDS,
enabling a direct comparison of proxy quality (Spearman rho vs. actual
accuracy). MDS achieves rho>=0.5 on 9 of 10 datasets; both baselines fail
on at least one headline remote-sensing dataset (NASWOT: rho=-0.036 on
PatternNet, rho=0.114 on EuroSAT).

* **NASWOT-adapted** (Mellor et al., ICML 2021): log-det of the feature
  Gram matrix K = F F^T. Measures linear independence / feature diversity.
* **SynFlow-adapted** (Tanaka et al., NeurIPS 2020): single-step synaptic
  flow of a freshly initialized linear probe. Measures how strongly
  features align with class labels.

Extracted verbatim from the original pipeline, Cell 12.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .mds import apply_rass_proxy, compute_mds


def naswot_score(features: np.ndarray) -> float:
    """log-det of the feature Gram matrix K = F @ F.T.

    Higher score -> more expressive / linearly independent feature space.
    """
    K = features @ features.T
    n = K.shape[0]
    K_reg = K + 1e-4 * np.eye(n)
    sign, ld = np.linalg.slogdet(K_reg)
    return float(ld) if sign > 0 else -1e9


def synflow_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Synaptic flow of a single-epoch linear probe.

    Higher score -> stronger class signal already present in the features.
    """
    F = StandardScaler().fit_transform(features)
    n, d = F.shape
    n_cls = int(np.max(labels)) + 1

    Ft = torch.tensor(F, dtype=torch.float32)
    yt = torch.tensor(labels, dtype=torch.long)
    W = nn.Parameter(torch.randn(n_cls, d) * 0.01)
    b = nn.Parameter(torch.zeros(n_cls))

    loss = nn.CrossEntropyLoss()(Ft @ W.T + b, yt)
    loss.backward()

    saliency = (W.detach().abs() * W.grad.abs()).sum().item()
    saliency += (b.detach().abs() * b.grad.abs()).sum().item()
    return float(saliency)


def score_candidate_all_proxies(candidate, layer_feats, pretrain_ref, labels, hg, baseline_fdr):
    """Compute MDS, NASWOT-adapted, and SynFlow-adapted for one candidate."""
    adapted = apply_rass_proxy(candidate.ops, layer_feats, hg)
    adapted_norm = StandardScaler().fit_transform(adapted)
    return {
        "MDS": compute_mds(candidate, layer_feats, pretrain_ref, labels, hg, baseline_fdr),
        "NASWOT": naswot_score(adapted_norm),
        "SynFlow": synflow_score(adapted_norm, labels),
    }
