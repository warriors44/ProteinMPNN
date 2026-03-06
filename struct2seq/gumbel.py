from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_gumbel(shape: tuple[int, ...], device: torch.device, eps: float = 1e-20) -> torch.Tensor:
    """Sample from Gumbel(0, 1) distribution."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_top_k(logits: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """Sample a permutation from the Plackett-Luce distribution using the
    Gumbel Top-K trick (Yellott 1977, Kool et al. 2019).

    Args:
        logits: Unnormalized log-probabilities [B, N].
        k: Number of elements to select.  If None, returns full permutation.

    Returns:
        permutation: [B, k] tensor where permutation[b, i] is the index of the
            element selected at step i.
    """
    if k is None:
        k = logits.size(-1)
    gumbel_noise = sample_gumbel(logits.shape, device=logits.device)
    perturbed = logits + gumbel_noise
    _, permutation = perturbed.topk(k, dim=-1)
    return permutation


def plackett_luce_log_prob(
    logits: torch.Tensor,
    permutation: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute log q(z_i | z_{<i}) for each step i under the Plackett-Luce model.

    The Plackett-Luce probability of selecting item z_i at step i is:
        q(z_i | z_{<i}) = exp(h_{z_i}) / sum_{j not in z_{<i}} exp(h_j)

    Args:
        logits: Per-position scores [B, N].
        permutation: Sampled ordering [B, N] where permutation[b, i] = index
            of the position decoded at step i.
        mask: Optional padding mask [B, N] (1 = valid, 0 = padding).

    Returns:
        log_probs: [B, N] where log_probs[b, i] = log q(z_i | z_{<i}).
    """
    B, N = logits.shape
    K = permutation.size(1)

    ordered_logits = torch.gather(logits, 1, permutation)

    if mask is not None:
        valid_mask = torch.gather(mask, 1, permutation)
    else:
        valid_mask = torch.ones(B, K, device=logits.device)

    log_numerator = ordered_logits

    ordered_logits_masked = ordered_logits.clone()
    ordered_logits_masked[valid_mask == 0] = float('-inf')

    flipped = torch.flip(ordered_logits_masked, [1])
    log_cumsum = torch.flip(torch.logcumsumexp(flipped, dim=1), [1])

    log_probs = torch.where(
        valid_mask.bool(),
        log_numerator - log_cumsum,
        torch.zeros_like(log_numerator),
    )

    return log_probs
