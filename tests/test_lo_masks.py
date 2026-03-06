from __future__ import annotations

import torch

from protein_mpnn_lo_utils import ProteinMPNN_LO


def _all_neighbors_eidx(batch_size: int, n: int) -> torch.Tensor:
    """Build E_idx where each node attends to all nodes (K=n)."""
    base = torch.arange(n).view(1, 1, n).repeat(batch_size, n, 1)
    return base


def test_generalized_ar_mask_respects_rank_ordering() -> None:
    """Generalized AR mask should allow attending only to earlier-ranked nodes."""
    b, n = 1, 4
    e_idx = _all_neighbors_eidx(b, n)
    # perm[step] = position decoded at that step.
    perm = torch.tensor([[2, 0, 3, 1]])
    mask = ProteinMPNN_LO._build_generalized_ar_mask(e_idx, perm)  # [B,N,K]
    assert mask.shape == (b, n, n)

    # Build ranks: rank[pos] = step
    rank = torch.empty(n, dtype=torch.long)
    for step in range(n):
        rank[int(perm[0, step])] = step

    # For each node i and neighbor j, mask=1 iff rank[j] < rank[i]
    for i in range(n):
        for j in range(n):
            expected = 1.0 if int(rank[j]) < int(rank[i]) else 0.0
            got = float(mask[0, i, j].item())
            assert got == expected


def test_partial_ar_mask_blocks_future_among_remaining_nodes() -> None:
    """Partial AR mask should hide remaining nodes from each other."""
    b, n = 1, 5
    e_idx = _all_neighbors_eidx(b, n)
    full_perm = torch.tensor([[0, 3, 1, 4, 2]])
    # i_samples is 1-indexed: i=3 -> decoded steps are {0,1} and remaining are {2,3,4}
    i_samples = torch.tensor([3])
    mask = ProteinMPNN_LO._build_partial_ar_mask(e_idx, full_perm, i_samples)
    assert mask.shape == (b, n, n)

    # decoded positions at steps < i-1:
    decoded = {int(full_perm[0, 0]), int(full_perm[0, 1])}
    remaining = set(range(n)).difference(decoded)
    assert len(decoded) == 2
    assert len(remaining) == 3

    # Remaining nodes should only see decoded nodes, not other remaining nodes.
    for i in remaining:
        for j in remaining:
            assert float(mask[0, i, j].item()) == 0.0
        for j in decoded:
            assert float(mask[0, i, j].item()) == 1.0

