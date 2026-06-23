"""Fisher Discriminant Ratio and Gini coefficient.

FDR (paper Eq. 2) measures per-node class-discriminability and is the
quantity Proposition 1 connects to probe error. Gini measures how evenly
discriminability is spread across nodes (Appendix F.2 -- a weaker predictor
of MEHERAB's gain than the deficit ratio gamma itself, kept here for
completeness / the FDR-balance ablation).

Extracted verbatim from the original pipeline, Cell 9.
"""
import numpy as np


def fisher_discriminant_ratio(F: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Fisher Discriminant Ratio, rho = tr(S_B) / tr(S_W) (Eq. 2).

    Parameters
    ----------
    F : np.ndarray, shape (N, d)
        Feature matrix (already standardized upstream).
    y : np.ndarray, shape (N,)
        Class labels.
    """
    classes = np.unique(y)
    mu = F.mean(0)
    sb = np.zeros(F.shape[1])
    sw = np.zeros(F.shape[1])
    for c in classes:
        mask = y == c
        if mask.sum() < 2:
            continue
        mu_c = F[mask].mean(0)
        sb += mask.sum() * (mu_c - mu) ** 2
        sw += ((F[mask] - mu_c) ** 2).sum(0)
    return float(np.sum(sb) / (np.sum(sw) + 1e-8))


def gini_coefficient(values) -> float:
    """Gini coefficient of a list of FDR values across nodes.

    0 = perfectly balanced discriminability across nodes,
    1 = all discriminative signal concentrated in one node.
    """
    v = np.array(sorted(values))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1).dot(v) / (n * v.sum()))


def fdr_deficit_ratio(node_fdrs: dict, final_block_node_id: int) -> float:
    """The FDR deficit ratio gamma (Eq. 3): max-node FDR / final-block FDR.

    gamma > 1.5 is the paper's empirical threshold for "MEHERAB likely
    helps" (Sec. 6.1, Table 3).
    """
    rho_base = node_fdrs[final_block_node_id]
    return max(node_fdrs.values()) / (rho_base + 1e-8)
