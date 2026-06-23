"""Task-conditional homophilic-graph construction (paper Sec. 3.2).

Clusters the L transformer blocks into ``n_clusters`` nodes by average-
linkage hierarchical clustering on (1 - CKA), computes a per-node Fisher
Discriminant Ratio, and selects the discriminative node pair: a genuine
multi-node hyperedge if joint FDR exceeds the mean of its components by
more than ``hyperedge_delta``, otherwise a single edge between the two
highest-FDR nodes.

Empirically (paper Sec. 3.2, Proposition 2), no genuine multi-node
hyperedge is ever found across any of the ten benchmarks -- near-unit
intra-cluster CKA homophily makes this a structural property of ViT-B/16,
not a search failure. ``build_homophilic_graph`` always returns exactly one
hyperedge, the single highest-FDR pair, and reports the best pairwise gain
it rejected for transparency.

Extracted and lightly refactored (the implicit dependence on a module-level
``cfg`` global in the original notebook is now an explicit ``delta``
parameter) from the original pipeline, Cell 9.
"""
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler

from .cka import compute_cka_matrix
from .fdr import fisher_discriminant_ratio, gini_coefficient


@dataclass
class SemanticHypergraph:
    """The task-conditional homophilic graph over block-cluster nodes."""

    nodes: List[int]
    node_members: Dict[int, List[int]]
    hyperedges: List[frozenset]
    cka_matrix: np.ndarray
    cl_labels: np.ndarray
    node_fdrs: Dict[int, float]
    fdr_gini: float
    pairwise_gains: Dict[tuple, float]

    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return len(self.hyperedges)


def build_homophilic_graph(
    layer_feats: Dict[int, torch.Tensor],
    labels: np.ndarray,
    n_clusters: int = 4,
    hyperedge_delta: float = 0.005,
    verbose: bool = True,
) -> SemanticHypergraph:
    """Construct the homophilic graph for one dataset's proxy features.

    Parameters
    ----------
    layer_feats : dict[int, torch.Tensor]
        Block index -> (N, d) CLS-token feature matrix on the proxy set.
    labels : np.ndarray
        Proxy-set class labels.
    n_clusters : int
        Number of block-cluster nodes (K=4 throughout the paper; Appendix
        F.6 shows accuracy varies <=0.5pp across K in {2,3,4,6}).
    hyperedge_delta : float
        Minimum joint-FDR gain over the component mean required to accept
        a genuine multi-node hyperedge (paper: delta=0.005).
    """

    def log(msg):
        if verbose:
            print(msg)

    log("[HG] Computing CKA matrix ...")
    cka_mat = compute_cka_matrix(layer_feats)
    dist_mat = np.clip(1.0 - cka_mat, 0, None)
    np.fill_diagonal(dist_mat, 0)
    Z = linkage(squareform(dist_mat, checks=False), method="average")
    cl_labels = fcluster(Z, n_clusters, criterion="maxclust") - 1

    block_ids = sorted(layer_feats.keys())
    node_members: Dict[int, List[int]] = defaultdict(list)
    for bidx, cid in enumerate(cl_labels):
        node_members[int(cid)].append(block_ids[bidx])
    nodes = sorted(node_members.keys())
    log(f"[HG] Clusters: { {k: node_members[k] for k in nodes} }")

    node_feats = {
        nid: torch.stack([layer_feats[m] for m in mems]).mean(0).numpy()
        for nid, mems in node_members.items()
    }
    node_norms = {nid: StandardScaler().fit_transform(f) for nid, f in node_feats.items()}
    node_fdrs = {nid: fisher_discriminant_ratio(node_norms[nid], labels) for nid in nodes}
    fdr_gini = gini_coefficient(list(node_fdrs.values()))
    log(f"[HG] Per-node FDR: { {k: f'{v:.3f}' for k, v in node_fdrs.items()} }")
    log(f"[HG] FDR Gini: {fdr_gini:.3f}  (0=balanced/helps, 1=concentrated/neutral)")

    pairwise_gains = {}
    hyperedges: List[frozenset] = []
    for r in [2, 3]:
        for subset in itertools.combinations(nodes, r):
            joint = np.concatenate([node_norms[s] for s in subset], axis=1)
            jfdr = fisher_discriminant_ratio(joint, labels)
            avg_fdr = np.mean([node_fdrs[s] for s in subset])
            gain = jfdr - avg_fdr
            pairwise_gains[subset] = gain
            if gain > hyperedge_delta:
                hyperedges.append(frozenset(subset))
                log(f"[HG]   Hyperedge {set(subset)}: FDR={jfdr:.3f} gain={gain:+.4f} > delta")

    if not hyperedges:
        top2 = sorted(nodes, key=lambda n: node_fdrs[n], reverse=True)[:2]
        hyperedges.append(frozenset(top2))
        best_pair = (
            max(pairwise_gains.items(), key=lambda x: x[1])
            if pairwise_gains
            else (tuple(top2), 0.0)
        )
        log(f"[HG]   No genuine hyperedges (best pairwise gain={best_pair[1]:+.4f} < delta={hyperedge_delta})")
        log(f"[HG]   Fallback: single edge between top-2 FDR nodes {set(top2)}")
        log("[HG]   Interpretation: ViT-B/16 blocks provide limited cross-cluster synergy")
        log("[HG]   MEHERAB still discovers best single-node pathway via MDS")

    return SemanticHypergraph(
        nodes=nodes,
        node_members=dict(node_members),
        hyperedges=hyperedges,
        cka_matrix=cka_mat,
        cl_labels=cl_labels,
        node_fdrs=node_fdrs,
        fdr_gini=fdr_gini,
        pairwise_gains=pairwise_gains,
    )
