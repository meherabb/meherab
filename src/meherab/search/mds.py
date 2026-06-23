"""Modality Discriminability Score (paper Sec. 3.4, Eq. 5-7).

MDS(c) = alpha * TaskAlign(c) - (1-alpha) * ManifoldCollapse(c), alpha=0.5

* **TaskAlign** (Eq. 6) = tanh(FDR(adapted) / (FDR(final-block) + eps)) --
  saturates at 1 as transformed features exceed final-block discriminability.
  Normalized *relative to the final-block baseline FDR*, not a hardcoded
  constant -- this relative normalization is what Eq. 6 specifies.
* **ManifoldCollapse** (Eq. 7) = 1 - CKA(pretrained_ref, adapted) -- penalizes
  candidates that distort the feature manifold rather than exposing genuine
  structure.

MDS is evaluated entirely on the unlabeled proxy set with zero gradients and
zero backbone modification -- this is what makes the evolutionary search in
``meherab.search.evolutionary`` "zero-gradient."

Extracted and refactored (explicit parameters instead of module-level
``cfg``/``GLOBAL_SEED`` globals) from the original pipeline, Cell 11.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..graph.cka import linear_cka
from ..graph.fdr import fisher_discriminant_ratio
from ..rass.operations import TType


def _node_mean_proxy(layer_feats, hg, nid):
    return torch.stack([layer_feats[m] for m in hg.node_members[nid]]).mean(0).numpy()


def apply_rass_proxy(ops, layer_feats, hg, adapter_scale: float = 0.3, random_state: int = 42):
    """Apply RASS ops to *proxy* features for MDS scoring.

    This is a torch-feature-dict variant used only during the zero-gradient
    search loop; it does not use ``FittedRASSTransform`` (which expects
    numpy block-feature dicts already split into train/test). PCA bases are
    re-fit per call since the proxy set never changes during search.
    """
    parts = []
    for op in ops:
        if op.op_type == TType.SC:
            nf = _node_mean_proxy(layer_feats, hg, op.node_id)
            k = max(1, int(nf.shape[1] * op.param))
            idx = nf.var(0).argsort()[::-1][:k]
            parts.append(nf[:, idx])
        elif op.op_type == TType.CF:
            edge = list(hg.hyperedges[op.edge_id])
            stk = np.stack([_node_mean_proxy(layer_feats, hg, n) for n in edge])
            nrm = np.linalg.norm(stk, axis=2, keepdims=True) + 1e-8
            parts.append((stk / nrm).mean(0))
        elif op.op_type == TType.AI:
            nf = _node_mean_proxy(layer_feats, hg, op.node_id)
            rank = int(op.param)
            nc = min(rank, nf.shape[0] - 1, nf.shape[1] - 1)
            if nc < 1:
                parts.append(nf)
                continue
            pca = PCA(n_components=nc, random_state=random_state)
            recon = pca.inverse_transform(pca.fit_transform(nf))
            parts.append(nf + adapter_scale * (recon - nf))
    if not parts:
        return np.zeros((list(layer_feats.values())[0].shape[0], 1))
    return np.concatenate(parts, axis=1)


def compute_mds(
    candidate,
    layer_feats,
    pretrain_ref,
    labels,
    hg,
    baseline_fdr: float,
    alpha: float = 0.5,
    adapter_scale: float = 0.3,
    random_state: int = 42,
) -> float:
    """Modality Discriminability Score -- zero-shot, no gradients (Eq. 5).

    Parameters
    ----------
    baseline_fdr : float
        FDR of the final-block features on the proxy set (the rho_base of
        Eq. 3, reused here for TaskAlign's relative normalization).
    """
    adapted = apply_rass_proxy(candidate.ops, layer_feats, hg, adapter_scale, random_state)
    adapted_norm = StandardScaler().fit_transform(adapted)

    # TaskAlign (Eq. 6) -- relative to baseline_fdr.
    fdr = fisher_discriminant_ratio(adapted_norm, labels)
    ta = float(np.tanh(fdr / (baseline_fdr + 1e-8)))

    # ManifoldCollapse (Eq. 7): 1 - CKA(adapted, pretrained).
    pf, af = pretrain_ref, adapted
    nc = min(min(pf.shape[1], af.shape[1]), pf.shape[0] - 1, af.shape[0] - 1, 64)
    if nc > 1:
        pf_r = PCA(nc, random_state=random_state).fit_transform(pf)
        af_r = PCA(nc, random_state=random_state).fit_transform(af)
    else:
        d = min(pf.shape[1], af.shape[1])
        pf_r, af_r = pf[:, :d], af[:, :d]
    mc = 1.0 - linear_cka(
        torch.tensor(pf_r, dtype=torch.float32), torch.tensor(af_r, dtype=torch.float32)
    )

    return float(alpha * ta - (1.0 - alpha) * mc)
