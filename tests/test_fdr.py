"""Correctness tests for the Fisher Discriminant Ratio and Gini coefficient
(paper Eq. 2-3, Proposition 1).
"""
import numpy as np

from meherab.graph.fdr import fisher_discriminant_ratio, gini_coefficient, fdr_deficit_ratio


def test_fdr_higher_for_well_separated_classes():
    """FDR must be substantially higher when class means are far apart
    relative to within-class scatter (the whole premise of Proposition 1:
    higher Fisher criterion -> strictly lower probe error).
    """
    rng = np.random.default_rng(0)
    F_separated = np.vstack([rng.normal(5, 1, (50, 8)), rng.normal(-5, 1, (50, 8))])
    F_overlapping = rng.normal(0, 1, (100, 8))
    y = np.array([0] * 50 + [1] * 50)

    fdr_sep = fisher_discriminant_ratio(F_separated, y)
    fdr_overlap = fisher_discriminant_ratio(F_overlapping, y)
    assert fdr_sep > fdr_overlap


def test_fdr_is_nonnegative():
    rng = np.random.default_rng(1)
    F = rng.normal(size=(60, 5))
    y = rng.integers(0, 3, size=60)
    assert fisher_discriminant_ratio(F, y) >= 0


def test_gini_zero_for_balanced_values():
    assert gini_coefficient([1.0, 1.0, 1.0, 1.0]) < 1e-9


def test_gini_increases_with_concentration():
    balanced = gini_coefficient([1.0, 1.0, 1.0, 1.0])
    concentrated = gini_coefficient([0.01, 0.01, 0.01, 10.0])
    assert concentrated > balanced


def test_gini_handles_empty_and_zero():
    assert gini_coefficient([]) == 0.0
    assert gini_coefficient([0.0, 0.0, 0.0]) == 0.0


def test_fdr_deficit_ratio_matches_definition():
    """gamma = max-node FDR / final-block FDR (Eq. 3)."""
    node_fdrs = {0: 0.5, 1: 0.9, 2: 0.3, 3: 0.384}  # node 3 = final block
    gamma = fdr_deficit_ratio(node_fdrs, final_block_node_id=3)
    # fdr_deficit_ratio adds a 1e-8 epsilon to the denominator for numerical
    # safety against a zero baseline FDR, so allow a correspondingly small
    # tolerance rather than expecting exact equality.
    expected = max(node_fdrs.values()) / node_fdrs[3]
    assert abs(gamma - expected) < 1e-6
    assert gamma > 1.0  # the highest-FDR node (0.9) exceeds the final block (0.384)
