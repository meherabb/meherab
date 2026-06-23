"""Linear CKA (Centered Kernel Alignment) between block representations.

Implements Eq. 1 in the paper. Extracted verbatim from the original
pipeline, Cell 9.
"""
import numpy as np
import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA similarity between two feature matrices (N x d each).

    Paper Eq. 1::

        K_ll' = ||F_l'^T F_l||_F^2 / (||F_l^T F_l||_F * ||F_l'^T F_l'||_F)
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    n = X.shape[0]
    xy = ((X.T @ Y) ** 2).sum() / (n - 1) ** 2
    xx = ((X.T @ X) ** 2).sum() / (n - 1) ** 2
    yy = ((Y.T @ Y) ** 2).sum() / (n - 1) ** 2
    return (xy / (xx.sqrt() * yy.sqrt()).clamp(1e-8)).item()


def compute_cka_matrix(layer_feats: dict) -> np.ndarray:
    """Pairwise linear-CKA similarity matrix across all transformer blocks.

    Parameters
    ----------
    layer_feats : dict[int, torch.Tensor]
        Mapping block index -> (N, d) CLS-token feature matrix on the
        proxy set.

    Returns
    -------
    np.ndarray
        L x L symmetric similarity matrix (paper Fig. 1a).
    """
    ids = sorted(layer_feats.keys())
    n = len(ids)
    mat = np.eye(n)
    for i, bi in enumerate(ids):
        for j, bj in enumerate(ids):
            if i < j:
                mat[i, j] = mat[j, i] = linear_cka(layer_feats[bi], layer_feats[bj])
    return mat
