"""Tests for ``FittedRASSTransform`` -- specifically its anti-leakage
property (paper Sec. 3.3): every operation's parameters (PCA bases for
Adapter-Inject, variance-ranked indices for Semantic-Compress) must be fit
**once, on training features only**, then frozen for application to any
other feature set, including test data.
"""
import numpy as np
import torch

from meherab.graph.homophilic_graph import build_homophilic_graph
from meherab.rass.operations import RASSFactory
from meherab.rass.transform import FittedRASSTransform, build_meherab_features


def _make_graph_and_factory(seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    layer_feats = {i: torch.randn(50, 24) for i in range(12)}
    labels = rng.integers(0, 3, size=50)
    hg = build_homophilic_graph(layer_feats, labels, n_clusters=4, verbose=False)
    factory = RASSFactory(hg)
    return hg, factory


def test_transform_is_deterministic_after_fitting():
    """Applying the same frozen transform twice to the same data must give
    identical output -- proves no parameter is being refit at apply-time.
    """
    hg, factory = _make_graph_and_factory(seed=0)
    cand = factory.random_candidate()
    rng = np.random.default_rng(1)
    block_feats_train = {i: rng.normal(size=(40, 24)).astype(np.float32) for i in range(12)}
    block_feats_other = {i: rng.normal(size=(15, 24)).astype(np.float32) * 3 + 50 for i in range(12)}

    tf = FittedRASSTransform(cand.ops, hg, block_feats_train, adapter_scale=0.3, random_state=42)
    out_1 = tf.apply(block_feats_other)
    out_2 = tf.apply(block_feats_other)
    assert np.allclose(out_1, out_2)


def test_transform_params_unaffected_by_application_data():
    """Fitting on train, then applying to two *different* "test-like" sets
    must use the exact same frozen parameters -- i.e. the transform's
    learned indices/PCA bases must not change between calls.
    """
    hg, factory = _make_graph_and_factory(seed=2)
    cand = factory.random_candidate()
    rng = np.random.default_rng(3)
    block_feats_train = {i: rng.normal(size=(40, 24)).astype(np.float32) for i in range(12)}
    tf = FittedRASSTransform(cand.ops, hg, block_feats_train, adapter_scale=0.3, random_state=42)
    fits_snapshot_before = dict(tf._fits)

    # Apply to a wildly different distribution -- must not alter stored fits.
    block_feats_weird = {i: rng.normal(size=(10, 24)).astype(np.float32) * 100 + 999 for i in range(12)}
    tf.apply(block_feats_weird)

    for op, fitted in tf._fits.items():
        before = fits_snapshot_before[op]
        if isinstance(fitted, np.ndarray):
            assert np.array_equal(fitted, before)
        else:
            # PCA object: compare a key fitted attribute.
            assert np.array_equal(fitted.components_, before.components_)


def test_build_meherab_features_concatenates_final_and_rass():
    hg, factory = _make_graph_and_factory(seed=4)
    cand = factory.random_candidate()
    rng = np.random.default_rng(5)
    n_samples, final_dim = 20, 768
    block_feats = {i: rng.normal(size=(n_samples, 24)).astype(np.float32) for i in range(12)}
    final_feats = rng.normal(size=(n_samples, final_dim)).astype(np.float32)

    tf = FittedRASSTransform(cand.ops, hg, block_feats, adapter_scale=0.3, random_state=42)
    meherab_feats = build_meherab_features(tf, block_feats, final_feats)

    # MEHERAB features must contain the final-block features as a strict
    # prefix (Sec. 3.1: "final-block features are always retained").
    assert meherab_feats.shape[0] == n_samples
    assert meherab_feats.shape[1] > final_dim
    assert np.allclose(meherab_feats[:, :final_dim], final_feats)
