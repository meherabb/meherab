"""Leakage-safe RASS feature transformation (paper Sec. 3.3).

``FittedRASSTransform`` fits every operation's parameters (PCA bases for
Adapter-Inject, variance-ranked indices for Semantic-Compress) **once on
training features only**, then freezes them for application to both train
and test sets -- this is what prevents train/test leakage in the reported
numbers.

MEHERAB features are the concatenation ``[final_block_features ||
RASS_output]`` (Sec. 3.1): the final-block features are always retained, so
the RASS transform only ever *adds* discriminative structure on top of the
linear-probe baseline. No ``max(rass, lp)`` floor is applied anywhere --
the reported MEHERAB numbers are the raw value of this concatenation.

Extracted and refactored (explicit ``adapter_scale``/``random_state``
parameters instead of module-level ``cfg``/``GLOBAL_SEED`` globals) from
the original pipeline, Cell 10.
"""
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA

from .operations import TType


class FittedRASSTransform:
    """Fit-once RASS transform: parameters fixed on training data."""

    def __init__(self, ops, hg, block_feats_train, adapter_scale: float = 0.3, random_state: int = 42):
        self.ops = ops
        self.hg = hg
        self.adapter_scale = adapter_scale
        self.random_state = random_state
        self._fits: Dict = {}
        self._build(block_feats_train)

    def _node_mean(self, bf, nid):
        return np.stack([bf[m] for m in self.hg.node_members[nid]]).mean(0)

    def _build(self, bf):
        # Pre-fit all transform parameters on training features only.
        for op in self.ops:
            if op.op_type == TType.SC:
                nf = self._node_mean(bf, op.node_id)
                k = max(1, int(nf.shape[1] * op.param))
                idx = nf.var(0).argsort()[::-1][:k]
                self._fits[op] = idx.copy()
            elif op.op_type == TType.AI:
                nf = self._node_mean(bf, op.node_id)
                rank = int(op.param)
                nc = min(rank, nf.shape[0] - 1, nf.shape[1] - 1)
                pca = PCA(n_components=max(1, nc), random_state=self.random_state)
                pca.fit(nf)
                self._fits[op] = pca

    def apply(self, block_feats):
        """Apply the frozen transform to any feature set (train or test)."""
        parts = []
        for op in self.ops:
            if op.op_type == TType.SC:
                nf = self._node_mean(block_feats, op.node_id)
                parts.append(nf[:, self._fits[op]])
            elif op.op_type == TType.CF:
                edge = list(self.hg.hyperedges[op.edge_id])
                stk = np.stack([self._node_mean(block_feats, n) for n in edge])
                nrm = np.linalg.norm(stk, axis=2, keepdims=True) + 1e-8
                parts.append((stk / nrm).mean(0))
            elif op.op_type == TType.AI:
                nf = self._node_mean(block_feats, op.node_id)
                pca = self._fits[op]
                recon = pca.inverse_transform(pca.transform(nf))
                parts.append(nf + self.adapter_scale * (recon - nf))
        if not parts:
            return np.zeros((list(block_feats.values())[0].shape[0], 1))
        return np.concatenate(parts, axis=1)


def build_meherab_features(tf: FittedRASSTransform, block_feats, final_feats):
    """MEHERAB features = concat(backbone_final, RASS_output). No LP floor."""
    rass_out = tf.apply(block_feats)
    return np.concatenate([final_feats, rass_out], axis=1)
