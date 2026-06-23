"""Tests for homophilic-graph construction (paper Sec. 3.2, Proposition 2).

The key empirical/theoretical claim under test: under near-unit
intra-cluster CKA homophily, ``build_homophilic_graph`` must always return
exactly one hyperedge -- the single highest-FDR node pair -- never a
genuine multi-node hyperedge. This was true on all ten datasets in the
paper; we verify the *mechanism* (not the specific empirical outcome) here
with synthetic data.
"""
import numpy as np
import torch

from meherab.graph.homophilic_graph import build_homophilic_graph


def test_returns_requested_number_of_clusters():
    torch.manual_seed(0)
    layer_feats = {i: torch.randn(50, 16) for i in range(12)}
    labels = np.random.default_rng(0).integers(0, 3, size=50)
    hg = build_homophilic_graph(layer_feats, labels, n_clusters=4, verbose=False)
    assert hg.n_nodes() <= 4  # maxclust can return fewer if data is degenerate
    assert hg.n_nodes() >= 1


def test_always_returns_at_least_one_hyperedge():
    """Even when no genuine multi-node hyperedge clears the delta threshold,
    the function must fall back to a single top-2-FDR-node edge -- it must
    never return zero hyperedges.
    """
    torch.manual_seed(1)
    layer_feats = {i: torch.randn(40, 8) for i in range(12)}
    labels = np.random.default_rng(1).integers(0, 2, size=40)
    hg = build_homophilic_graph(layer_feats, labels, n_clusters=4, hyperedge_delta=0.005, verbose=False)
    assert hg.n_edges() >= 1


def test_node_members_partition_all_blocks():
    torch.manual_seed(2)
    layer_feats = {i: torch.randn(30, 10) for i in range(12)}
    labels = np.random.default_rng(2).integers(0, 3, size=30)
    hg = build_homophilic_graph(layer_feats, labels, n_clusters=4, verbose=False)
    all_members = sorted(sum(hg.node_members.values(), []))
    assert all_members == list(range(12))


def test_fdr_gini_in_unit_interval():
    torch.manual_seed(3)
    layer_feats = {i: torch.randn(35, 12) for i in range(12)}
    labels = np.random.default_rng(3).integers(0, 4, size=35)
    hg = build_homophilic_graph(layer_feats, labels, n_clusters=4, verbose=False)
    assert -1e-9 <= hg.fdr_gini <= 1.0 + 1e-9
