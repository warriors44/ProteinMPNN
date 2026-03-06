from __future__ import annotations

import math

import torch

from protein_mpnn_lo_utils import gumbel_top_k, plackett_luce_log_prob


def test_gumbel_top_k_returns_permutation() -> None:
    """gumbel_top_k should return valid indices without repetition."""
    logits = torch.zeros(4, 10)
    perm = gumbel_top_k(logits)
    assert perm.shape == (4, 10)
    for b in range(perm.size(0)):
        uniq = torch.unique(perm[b])
        assert uniq.numel() == perm.size(1)
        assert int(perm[b].min()) >= 0
        assert int(perm[b].max()) < logits.size(1)


def test_plackett_luce_log_prob_matches_manual_small_case() -> None:
    """plackett_luce_log_prob should match a manual computation on N=3."""
    # B=1, N=3
    logits = torch.tensor([[0.2, -0.3, 1.1]])
    perm = torch.tensor([[2, 0, 1]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])

    # Manual per-step log prob:
    # step0 pick 2: log exp(l2) / (exp(l0)+exp(l1)+exp(l2))
    # step1 pick 0: log exp(l0) / (exp(l0)+exp(l1))
    # step2 pick 1: log exp(l1) / exp(l1) = 0
    l0, l1, l2 = logits[0].tolist()
    step0 = l2 - math.log(math.exp(l0) + math.exp(l1) + math.exp(l2))
    step1 = l0 - math.log(math.exp(l0) + math.exp(l1))
    step2 = 0.0
    expected = torch.tensor([[step0, step1, step2]], dtype=logits.dtype)

    got = plackett_luce_log_prob(logits, perm, mask)
    assert torch.allclose(got, expected, atol=1e-6)

