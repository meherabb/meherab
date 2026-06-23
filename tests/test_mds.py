"""Tests for the Modality Discriminability Score (paper Eq. 5-7)."""
import numpy as np
import torch

from meherab.graph.fdr import fisher_discriminant_ratio
from meherab.graph.homophilic_graph import build_homophilic_graph
from meherab.rass.operations import RASSFactory
from meherab.search.mds import compute_mds, apply_rass_proxy
from sklearn.preprocessing import StandardScaler


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    layer_feats = {i: torch.randn(60, 20) for i in range(12)}
    labels = rng.integers(0, 3, size=60)
    hg = build_homophilic_graph(layer_feats, labels, n_clusters=4, verbose=False)
    factory = RASSFactory(hg)
    pretrain_ref = np.stack([lf.numpy() for lf in layer_feats.values()]).mean(0)
    final_id = max(layer_feats.keys())
    baseline_fdr = fisher_discriminant_ratio(
        StandardScaler().fit_transform(layer_feats[final_id].numpy()), labels
    )
    return layer_feats, labels, hg, factory, pretrain_ref, baseline_fdr


def test_mds_returns_finite_scalar():
    layer_feats, labels, hg, factory, pretrain_ref, baseline_fdr = _setup()
    cand = factory.random_candidate()
    score = compute_mds(cand, layer_feats, pretrain_ref, labels, hg, baseline_fdr)
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_mds_alpha_zero_is_pure_negative_manifold_collapse():
    """alpha=0 -> MDS = -(1)*ManifoldCollapse, i.e. TaskAlign term has zero
    weight. We check this by comparing alpha=0 vs alpha=1 give different
    scores (proving alpha actually changes the weighting, per Eq. 5).
    """
    layer_feats, labels, hg, factory, pretrain_ref, baseline_fdr = _setup(seed=1)
    cand = factory.random_candidate()
    score_a0 = compute_mds(cand, layer_feats, pretrain_ref, labels, hg, baseline_fdr, alpha=0.0)
    score_a1 = compute_mds(cand, layer_feats, pretrain_ref, labels, hg, baseline_fdr, alpha=1.0)
    assert score_a0 != score_a1


def test_apply_rass_proxy_output_shape_matches_n_samples():
    layer_feats, labels, hg, factory, pretrain_ref, baseline_fdr = _setup(seed=2)
    cand = factory.random_candidate()
    adapted = apply_rass_proxy(cand.ops, layer_feats, hg)
    n_samples = list(layer_feats.values())[0].shape[0]
    assert adapted.shape[0] == n_samples


def test_mds_is_deterministic_given_fixed_random_state():
    layer_feats, labels, hg, factory, pretrain_ref, baseline_fdr = _setup(seed=3)
    cand = factory.random_candidate()
    s1 = compute_mds(cand, layer_feats, pretrain_ref, labels, hg, baseline_fdr, random_state=42)
    s2 = compute_mds(cand, layer_feats, pretrain_ref, labels, hg, baseline_fdr, random_state=42)
    assert s1 == s2
