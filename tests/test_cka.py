"""Correctness tests for linear CKA (paper Eq. 1).

These are mathematical-property tests, not full-pipeline tests -- they use
synthetic data and run in well under a second, with no GPU or dataset
download required.
"""
import torch

from meherab.graph.cka import linear_cka, compute_cka_matrix


def test_cka_self_similarity_is_one():
    """CKA(X, X) must equal 1.0 -- a representation is maximally similar to itself."""
    torch.manual_seed(0)
    X = torch.randn(100, 32)
    assert abs(linear_cka(X, X.clone()) - 1.0) < 1e-4


def test_cka_is_symmetric():
    torch.manual_seed(1)
    X = torch.randn(80, 16)
    Y = torch.randn(80, 16)
    assert abs(linear_cka(X, Y) - linear_cka(Y, X)) < 1e-6


def test_cka_invariant_to_orthogonal_transform():
    """CKA is invariant to orthogonal transforms of either input (Kornblith et al., 2019)."""
    torch.manual_seed(2)
    X = torch.randn(60, 20)
    # Random orthogonal matrix via QR decomposition.
    Q, _ = torch.linalg.qr(torch.randn(20, 20))
    X_rotated = X @ Q
    Y = torch.randn(60, 20)
    assert abs(linear_cka(X, Y) - linear_cka(X_rotated, Y)) < 1e-4


def test_cka_bounded_in_unit_interval():
    torch.manual_seed(3)
    X = torch.randn(50, 10)
    Y = torch.randn(50, 10)
    val = linear_cka(X, Y)
    assert -1e-6 <= val <= 1.0 + 1e-6


def test_compute_cka_matrix_diagonal_is_one():
    """Every diagonal entry of the L x L CKA matrix must be 1 (self-similarity)."""
    torch.manual_seed(4)
    layer_feats = {i: torch.randn(40, 12) for i in range(6)}
    mat = compute_cka_matrix(layer_feats)
    for i in range(6):
        assert abs(mat[i, i] - 1.0) < 1e-4
    # Symmetric
    assert (abs(mat - mat.T) < 1e-6).all()
