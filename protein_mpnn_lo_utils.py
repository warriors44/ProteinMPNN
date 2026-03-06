from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_mpnn_utils import (
    gather_edges,
    gather_nodes,
    gather_nodes_t,
    cat_neighbors_nodes,
    EncLayer,
    DecLayer,
    ProteinFeatures,
    CA_ProteinFeatures,
)


# ======================================================================
# Gumbel utilities (ported from struct2seq/gumbel.py)
# ======================================================================

def _sample_gumbel(
    shape: Tuple[int, ...],
    device: torch.device,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Sample from Gumbel(0, 1) distribution."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_top_k(
    logits: torch.Tensor,
    k: Optional[int] = None,
) -> torch.Tensor:
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
    gumbel_noise = _sample_gumbel(logits.shape, device=logits.device)
    perturbed = logits + gumbel_noise
    _, permutation = perturbed.topk(k, dim=-1)
    return permutation


def plackett_luce_log_prob(
    logits: torch.Tensor,
    permutation: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute log q(z_i | z_{<i}) for each step i under the Plackett-Luce model.

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


# ======================================================================
# ProteinMPNN_LO  --  Learning-Order ProteinMPNN
# ======================================================================

class ProteinMPNN_LO(nn.Module):
    """ProteinMPNN with Learning-Order autoregressive decoding (LO-ARM).

    Extends the ProteinMPNN architecture with a learned decoding order
    based on the Plackett-Luce distribution and Gumbel Top-K sampling,
    following the framework of Wang et al. (arXiv:2503.05979).

    Three probability components:
      - p_theta(x_{z_i} | x_{z_{<i}}, S): token classifier (W_out)
      - p_theta(z_i | z_{<i}, x_{z_{<i}}, S): order-policy prior (W_order_p)
      - q_theta(z_i | z_{<i}, x, S): variational order posterior (W_order_q)
    """

    def __init__(
        self,
        num_letters: int = 21,
        node_features: int = 128,
        edge_features: int = 128,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 21,
        k_neighbors: int = 64,
        augment_eps: float = 0.05,
        dropout: float = 0.1,
        ca_only: bool = False,
        num_samples: int = 2,
        separate_q_decoder: bool = False,
    ) -> None:
        super(ProteinMPNN_LO, self).__init__()

        if num_samples < 2:
            raise ValueError("num_samples must be >= 2 for RLOO estimator.")

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.separate_q_decoder = separate_q_decoder
        self.ca_only = ca_only

        # ---- Featurization ----
        if ca_only:
            self.features = CA_ProteinFeatures(
                node_features, edge_features,
                top_k=k_neighbors, augment_eps=augment_eps,
            )
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(
                node_features, edge_features,
                top_k=k_neighbors, augment_eps=augment_eps,
            )

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # ---- Encoder (shared) ----
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # ---- p_theta decoder ----
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        # Token prediction head
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Order-policy heads
        self.W_order_p = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.W_order_q = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ---- Optional separate q_theta decoder ----
        if self.separate_q_decoder:
            self.q_decoder_layers = nn.ModuleList([
                DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
                for _ in range(num_decoder_layers)
            ])
            self.W_order_q_sep = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        # ---- Parameter initialization ----
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ==================================================================
    # Encoder
    # ==================================================================

    def _encode(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run structure encoder.

        Returns:
            h_V_enc: encoded node features [B, L, H]
            h_E: encoded edge features [B, L, K, H]
            E_idx: neighbor indices [B, L, K]
        """
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros(
            (E.shape[0], E.shape[1], E.shape[-1]), device=E.device,
        )
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V, h_E, E_idx

    # ==================================================================
    # Generalized autoregressive masks  (rank-based, [B,N,K])
    # ==================================================================

    @staticmethod
    def _build_generalized_ar_mask(
        E_idx: torch.Tensor,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        """Build autoregressive mask from an arbitrary decoding permutation.

        Args:
            E_idx: neighbor indices [B, N, K].
            permutation: [B, N] where permutation[b, i] = position decoded at step i.

        Returns:
            mask: [B, N, K] with 1 where neighbor was decoded earlier.
        """
        B, N = permutation.shape
        device = permutation.device

        rank = torch.zeros(B, N, dtype=torch.long, device=device)
        rank.scatter_(
            1, permutation,
            torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
        )

        rank_self = rank.unsqueeze(-1)
        rank_flat = rank.unsqueeze(1).expand(-1, N, -1)
        rank_neighbors = torch.gather(rank_flat, 2, E_idx)

        mask = (rank_neighbors < rank_self).float()
        return mask

    @staticmethod
    def _build_partial_ar_mask(
        E_idx: torch.Tensor,
        full_perm: torch.Tensor,
        i_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Build autoregressive mask for partial decode (z_{<i}).

        Decoded positions j=z_k see only z_{<k}. Remaining positions see
        z_{<i} and not each other.

        Args:
            E_idx: neighbor indices [B, N, K].
            full_perm: full permutation [B, N].
            i_samples: [B] number of decoded steps so far (1-indexed).

        Returns:
            mask: [B, N, K] with 1 where neighbor is in the visible past.
        """
        B, N = full_perm.shape
        device = full_perm.device

        rank = torch.zeros(B, N, dtype=torch.long, device=device)
        rank.scatter_(
            1, full_perm,
            torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
        )

        i_minus_1 = (i_samples - 1).clamp(min=0).unsqueeze(1)
        rank_corrected = torch.where(
            rank < i_minus_1,
            rank,
            i_minus_1.expand(-1, N),
        )

        rank_self = rank_corrected.unsqueeze(-1)
        rank_flat = rank_corrected.unsqueeze(1).expand(-1, N, -1)
        rank_neighbors = torch.gather(rank_flat, 2, E_idx)

        mask = (rank_neighbors < rank_self).float()
        return mask

    # ==================================================================
    # q_theta path: unmasked decoder for variational posterior
    # ==================================================================

    def forward_q(
        self,
        h_V_enc: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        design_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute q_theta order logits using the unmasked (full info) decoder.

        All sequence information is visible (no causal mask).

        Args:
            h_V_enc: encoder output [B, L, H].
            h_E: edge embeddings [B, L, K, H].
            E_idx: neighbor indices [B, L, K].
            S: ground-truth sequence [B, L].
            mask: padding mask [B, L].
            design_mask: [B, L] 1 for designable positions.

        Returns:
            q_logits: [B, L] per-position order logits (-inf for non-designable).
        """
        h_V = h_V_enc.clone()
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        if self.separate_q_decoder:
            decoder_layers = self.q_decoder_layers
            order_head = self.W_order_q_sep
        else:
            decoder_layers = self.decoder_layers
            order_head = self.W_order_q

        for dec_layer in decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_V = dec_layer(h_V, h_ESV, mask)

        q_logits = order_head(h_V).squeeze(-1)
        q_logits = q_logits.masked_fill(design_mask == 0, float('-inf'))
        return q_logits

    # ==================================================================
    # p_theta path: generalized causal decoder
    # ==================================================================

    def forward_p(
        self,
        h_V_enc: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        design_mask: torch.Tensor,
        permutation: Optional[torch.Tensor] = None,
        ar_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run decoder with a generalized autoregressive mask.

        Args:
            h_V_enc: encoder output [B, L, H].
            h_E: edge embeddings [B, L, K, H].
            E_idx: neighbor indices [B, L, K].
            S: ground-truth sequence [B, L] (teacher forcing).
            mask: padding mask [B, L].
            design_mask: [B, L] 1 for designable positions.
            permutation: decoding order [B, L]. Required if ar_mask is None.
            ar_mask: precomputed mask [B, L, K].

        Returns:
            log_probs: [B, L, vocab] log-probabilities for each position.
            p_order_logits: [B, L] per-position order logits for p_theta.
        """
        if ar_mask is not None:
            mask_attend = ar_mask.unsqueeze(-1)
        elif permutation is not None:
            ar_mask = self._build_generalized_ar_mask(E_idx, permutation)
            mask_attend = ar_mask.unsqueeze(-1)
        else:
            raise ValueError("Either permutation or ar_mask must be provided")

        h_V = h_V_enc.clone()
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for dec_layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = dec_layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)

        p_order_logits = self.W_order_p(h_V).squeeze(-1)
        p_order_logits = p_order_logits.masked_fill(design_mask == 0, float('-inf'))

        return log_probs, p_order_logits

    # ==================================================================
    # F_theta with exact expectation over z_i  (Eq. 8)
    # ==================================================================

    @staticmethod
    def _compute_F_theta(
        log_probs: torch.Tensor,
        p_order_logits: torch.Tensor,
        q_logits: torch.Tensor,
        S: torch.Tensor,
        remaining_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute F_theta(z_{<i}, x) with exact expectation over z_i.

        F = sum_{j in remaining} q(z_i=j | z_{<i}, x)
              * [ log p(x_j | x_{z_{<i}}, s)
                + log p(z_i=j | z_{<i}, x_{z_{<i}}, s)
                - log q(z_i=j | z_{<i}, x, s) ]

        Args:
            log_probs: [B, L, vocab] from forward_p (with partial ar_mask).
            p_order_logits: [B, L] from forward_p.
            q_logits: [B, L] from forward_q (not detached).
            S: ground-truth sequence [B, L].
            remaining_mask: [B, L] binary, 1 for undecoded valid positions.

        Returns:
            F: [B] scalar F_theta per batch element.
        """
        log_p_token = torch.gather(
            log_probs, 2, S.unsqueeze(-1),
        ).squeeze(-1)

        neg_inf = float('-inf')
        log_p_order = F.log_softmax(
            p_order_logits.masked_fill(remaining_mask == 0, neg_inf), dim=-1,
        )
        log_q_order = F.log_softmax(
            q_logits.masked_fill(remaining_mask == 0, neg_inf), dim=-1,
        )

        q_weights = torch.where(
            remaining_mask.bool(),
            torch.exp(log_q_order),
            torch.zeros_like(log_q_order),
        )

        inner = torch.where(
            remaining_mask.bool(),
            log_p_token + log_p_order - log_q_order,
            torch.zeros_like(log_p_token),
        )
        F_val = (q_weights * inner).sum(-1)
        return F_val

    # ==================================================================
    # ELBO training  (Algorithm 1, Eqs. 8/9/11)
    # ==================================================================

    def compute_elbo(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """ELBO loss following Algorithm 1 of Wang et al. (arXiv:2503.05979).

        Args:
            X: coordinates [B, L, 4, 3].
            S: ground-truth sequence [B, L].
            mask: padding mask [B, L].
            chain_M: chain design mask [B, L] (1 = designable).
            residue_idx: residue indices [B, L].
            chain_encoding_all: chain encoding [B, L].

        Returns:
            loss: scalar loss to minimise.
            info: monitoring dict.
        """
        B, N = S.shape
        device = S.device
        design_mask = chain_M * mask

        h_V_enc, h_E, E_idx = self._encode(X, mask, residue_idx, chain_encoding_all)

        q_logits = self.forward_q(h_V_enc, h_E, E_idx, S, mask, design_mask)

        L_design = design_mask.sum(dim=-1).clamp(min=1.0)
        num_fixed = ((1.0 - design_mask) * mask).sum(dim=-1).long()

        # Sample i_design ~ Uniform(1, ..., L_designable).
        # In the full permutation, designable positions start after fixed ones,
        # so the absolute index is i_full = i_design + num_fixed.
        i_design = (torch.rand(B, device=device) * L_design).long() + 1
        i_full = i_design + num_fixed

        F_values: List[torch.Tensor] = []
        log_q_values: List[torch.Tensor] = []

        K = self.num_samples
        step_indices = torch.arange(N, device=device).unsqueeze(0)

        for _ in range(K):
            full_perm = self._build_fixed_first_perm(
                design_mask, mask, q_logits.detach(),
            )

            ar_mask = self._build_partial_ar_mask(E_idx, full_perm, i_full)
            log_probs_k, p_order_logits_k = self.forward_p(
                h_V_enc, h_E, E_idx, S, mask, design_mask, ar_mask=ar_mask,
            )

            rank = torch.zeros(B, N, dtype=torch.long, device=device)
            rank.scatter_(
                1, full_perm,
                torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
            )
            decoded_mask = (rank < (i_full - 1).unsqueeze(1)).float()
            remaining_mask = (1.0 - decoded_mask) * design_mask

            F_k = self._compute_F_theta(
                log_probs_k, p_order_logits_k, q_logits, S, remaining_mask,
            )
            F_values.append(F_k)

            log_q_all = plackett_luce_log_prob(q_logits, full_perm, mask)
            # Only sum over designable steps (after fixed positions, before i_full)
            step_mask = (
                (step_indices >= num_fixed.unsqueeze(1))
                & (step_indices < (i_full - 1).unsqueeze(1))
            ).float()
            # Fixed position steps have -inf log_q; zero them out
            log_q_safe = log_q_all.masked_fill(~torch.isfinite(log_q_all), 0.0)
            log_q_partial = (log_q_safe * step_mask).sum(-1)
            log_q_values.append(log_q_partial)

        F_stack = torch.stack(F_values, dim=0)       # [K, B]
        log_q_stack = torch.stack(log_q_values, dim=0)  # [K, B]

        F_mean = F_stack.mean(dim=0)  # [B]

        sum_F = F_stack.sum(dim=0)
        F_minus_k = (sum_F.unsqueeze(0) - F_stack) / float(K - 1)  # [K, B]

        adv = (F_stack - F_minus_k).detach()  # [K, B]

        rloo_term = (adv * log_q_stack).mean(dim=0)  # [B]

        loss_per_elem = -L_design * (F_mean + rloo_term)
        loss = loss_per_elem.sum() / L_design.sum()

        with torch.no_grad():
            elbo_per_res = F_mean.mean()
            delta_F_abs = (F_stack - F_mean.unsqueeze(0)).abs().mean()

            randn = torch.randn(chain_M.shape, device=device)
            log_probs_all = self.forward(
                X, S, mask, chain_M, residue_idx, chain_encoding_all, randn,
            )
            log_p_token_all = torch.gather(
                log_probs_all, 2, S.unsqueeze(-1),
            ).squeeze(-1)
            nll_avg = -(log_p_token_all * design_mask).sum() / design_mask.sum()

        info: Dict[str, torch.Tensor] = {
            'elbo': elbo_per_res,
            'nll': nll_avg,
            'F_mean': F_mean.mean(),
            'delta_F_abs': delta_F_abs,
            'i_mean': i_design.float().mean(),
        }
        return loss, info

    # ==================================================================
    # Helper: build permutation with fixed positions first
    # ==================================================================

    @staticmethod
    def _build_fixed_first_perm(
        design_mask: torch.Tensor,
        mask: torch.Tensor,
        q_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Build a full permutation placing fixed/padded positions first.

        Fixed and padded positions are ordered randomly among themselves and
        placed at the beginning of the permutation.  Designable positions are
        ordered according to a Gumbel-top-k sample from q_logits.

        Permutation convention: perm[b, step] = position decoded at that step.
        Fixed positions are "already decoded" so they occupy the first steps.

        Args:
            design_mask: [B, L] 1 for designable positions.
            mask: [B, L] padding mask.
            q_logits: [B, L] order logits (already -inf for non-designable).

        Returns:
            full_perm: [B, L] permutation tensor.
        """
        B, N = design_mask.shape
        device = design_mask.device

        gumbel_noise = _sample_gumbel(q_logits.shape, device=device)
        # Fixed/padded positions get very HIGH scores so topk selects them
        # first (= earliest ranks = "already decoded").
        # Designable positions get Gumbel-perturbed q_logits (finite).
        # NOTE: The "high" constant must be representable in the current dtype
        # (e.g. float16 under mixed precision), so we derive it from finfo
        # instead of hard-coding something like 1e9 which would overflow.
        high_val = torch.finfo(q_logits.dtype).max / 10.0
        scores = torch.where(
            design_mask.bool(),
            q_logits + gumbel_noise,
            torch.full_like(q_logits, high_val) + torch.rand_like(q_logits),
        )
        _, full_perm = scores.topk(N, dim=-1)
        return full_perm

    # ==================================================================
    # Standard forward  (backward-compatible NLL evaluation)
    # ==================================================================

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        randn: torch.Tensor,
        use_input_decoding_order: bool = False,
        decoding_order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard forward pass with random (or given) decoding order.

        Compatible with the original ProteinMPNN interface for NLL evaluation.
        """
        device = X.device
        design_mask = chain_M * mask

        h_V_enc, h_E, E_idx = self._encode(X, mask, residue_idx, chain_encoding_all)

        if not use_input_decoding_order:
            decoding_order = torch.argsort(
                (chain_M + 0.0001) * torch.abs(randn),
            )

        ar_mask = self._build_generalized_ar_mask(E_idx, decoding_order)
        log_probs, _ = self.forward_p(
            h_V_enc, h_E, E_idx, S, mask, design_mask,
            ar_mask=ar_mask,
        )
        return log_probs

    # ==================================================================
    # Sampling with learned order
    # ==================================================================

    def sample(
        self,
        X: torch.Tensor,
        randn: torch.Tensor,
        S_true: torch.Tensor,
        chain_mask: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        residue_idx: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        omit_AAs_np: Optional[np.ndarray] = None,
        bias_AAs_np: Optional[np.ndarray] = None,
        chain_M_pos: Optional[torch.Tensor] = None,
        omit_AA_mask: Optional[torch.Tensor] = None,
        pssm_coef: Optional[torch.Tensor] = None,
        pssm_bias: Optional[torch.Tensor] = None,
        pssm_multi: Optional[float] = None,
        pssm_log_odds_flag: Optional[bool] = None,
        pssm_log_odds_mask: Optional[torch.Tensor] = None,
        pssm_bias_flag: Optional[bool] = None,
        bias_by_res: Optional[torch.Tensor] = None,
        order_temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Autoregressive sampling using the learned order (p_theta).

        At each step:
          1. Select next position via p_theta order head.
          2. Sample amino acid at that position.
          3. Update hidden states.

        Fixed positions (chain_mask == 0) are pre-filled with S_true.

        Returns:
            dict with keys "S", "probs", "decoding_order".
        """
        device = X.device
        h_V_enc, h_E, E_idx = self._encode(X, mask, residue_idx, chain_encoding_all)

        chain_mask = chain_mask * chain_M_pos * mask
        N_batch, N_nodes = X.size(0), X.size(1)

        all_probs = torch.zeros(
            (N_batch, N_nodes, 21), device=device, dtype=torch.float32,
        )
        h_S = torch.zeros_like(h_V_enc, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        ordering = torch.zeros(
            (N_batch, N_nodes), dtype=torch.long, device=device,
        )
        decoded_mask = torch.zeros(N_batch, N_nodes, device=device)

        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask is not None

        # Pre-fill fixed positions
        for b in range(N_batch):
            for j in range(N_nodes):
                if chain_mask[b, j].item() < 0.5 and mask[b, j].item() > 0.5:
                    S[b, j] = S_true[b, j]
                    h_S[b, j, :] = self.W_s(S_true[b, j:j+1]).squeeze(0)
                    decoded_mask[b, j] = 1.0

        h_V_stack = [h_V_enc] + [
            torch.zeros_like(h_V_enc, device=device)
            for _ in range(len(self.decoder_layers))
        ]

        h_EX_encoder = cat_neighbors_nodes(
            torch.zeros_like(h_S), h_E, E_idx,
        )
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        step_count = 0
        for t_ in range(N_nodes):
            remaining = chain_mask * (1.0 - decoded_mask)
            if remaining.sum() == 0:
                break

            # --- Order selection via p_theta ---
            h_V_current = h_V_stack[-1]
            order_logits = self.W_order_p(h_V_current).squeeze(-1)
            order_logits = order_logits.masked_fill(
                remaining == 0, float('-inf'),
            )
            order_logits = order_logits / order_temperature

            order_probs = F.softmax(order_logits, dim=-1)
            t = torch.multinomial(order_probs, 1).squeeze(-1)  # [B]
            ordering[:, step_count] = t
            step_count += 1

            # --- Decode amino acid at selected position ---
            E_idx_t = torch.gather(
                E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]),
            )
            h_E_t = torch.gather(
                h_E, 1,
                t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]),
            )
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)

            neighbor_decoded = torch.gather(
                decoded_mask.unsqueeze(1).expand(-1, 1, E_idx_t.size(-1)),
                -1,
                E_idx_t[:, 0:1, :],
            ).unsqueeze(-1)  # [B, 1, K, 1]
            mask_bw_t = neighbor_decoded
            mask_fw_t = 1.0 - neighbor_decoded

            h_EXV_encoder_t = torch.gather(
                h_EXV_encoder, 1,
                t[:, None, None, None].repeat(
                    1, 1, h_EXV_encoder.shape[-2], h_EXV_encoder.shape[-1],
                ),
            )
            h_EXV_encoder_fw_t = mask_fw_t * h_EXV_encoder_t

            mask_t = torch.gather(mask, 1, t[:, None])
            for l, layer in enumerate(self.decoder_layers):
                h_ESV_decoder_t = cat_neighbors_nodes(
                    h_V_stack[l], h_ES_t, E_idx_t,
                )
                h_V_t = torch.gather(
                    h_V_stack[l], 1,
                    t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]),
                )
                h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_fw_t
                h_V_stack[l + 1].scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, h_V_enc.shape[-1]),
                    layer(h_V_t, h_ESV_t, mask_V=mask_t),
                )

            h_V_t_final = torch.gather(
                h_V_stack[-1], 1,
                t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]),
            )[:, 0]
            token_logits = self.W_out(h_V_t_final) / temperature

            bias_by_res_gathered = torch.gather(
                bias_by_res, 1,
                t[:, None, None].repeat(1, 1, 21),
            )[:, 0, :]

            probs = F.softmax(
                token_logits
                - constant[None, :] * 1e8
                + constant_bias[None, :] / temperature
                + bias_by_res_gathered / temperature,
                dim=-1,
            )
            if pssm_bias_flag:
                pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:, None])[:, 0]
                pssm_bias_gathered = torch.gather(
                    pssm_bias, 1,
                    t[:, None, None].repeat(1, 1, pssm_bias.shape[-1]),
                )[:, 0]
                probs = (
                    (1 - pssm_multi * pssm_coef_gathered[:, None]) * probs
                    + pssm_multi * pssm_coef_gathered[:, None] * pssm_bias_gathered
                )
            if pssm_log_odds_flag:
                pssm_log_odds_mask_gathered = torch.gather(
                    pssm_log_odds_mask, 1,
                    t[:, None, None].repeat(1, 1, pssm_log_odds_mask.shape[-1]),
                )[:, 0]
                probs_masked = probs * pssm_log_odds_mask_gathered
                probs_masked += probs * 0.001
                probs = probs_masked / torch.sum(
                    probs_masked, dim=-1, keepdim=True,
                )
            if omit_AA_mask_flag:
                omit_AA_mask_gathered = torch.gather(
                    omit_AA_mask, 1,
                    t[:, None, None].repeat(1, 1, omit_AA_mask.shape[-1]),
                )[:, 0]
                probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                probs = probs_masked / torch.sum(
                    probs_masked, dim=-1, keepdim=True,
                )

            S_t = torch.multinomial(probs, 1)
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])
            all_probs.scatter_(
                1,
                t[:, None, None].repeat(1, 1, 21),
                (chain_mask_gathered[:, :, None] * probs[:, None, :]).float(),
            )
            S_true_gathered = torch.gather(S_true, 1, t[:, None])
            S_t = (
                S_t * chain_mask_gathered
                + S_true_gathered * (1.0 - chain_mask_gathered)
            ).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(
                1,
                t[:, None, None].repeat(1, 1, temp1.shape[-1]),
                temp1,
            )
            S.scatter_(1, t[:, None], S_t)
            decoded_mask.scatter_(1, t[:, None], 1.0)

        output_dict: Dict[str, Any] = {
            "S": S,
            "probs": all_probs,
            "decoding_order": ordering,
        }
        return output_dict

    # ==================================================================
    # Tied sampling (group-based, for oligomers)
    # ==================================================================

    def tied_sample(
        self,
        X: torch.Tensor,
        randn: torch.Tensor,
        S_true: torch.Tensor,
        chain_mask: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        residue_idx: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        omit_AAs_np: Optional[np.ndarray] = None,
        bias_AAs_np: Optional[np.ndarray] = None,
        chain_M_pos: Optional[torch.Tensor] = None,
        omit_AA_mask: Optional[torch.Tensor] = None,
        pssm_coef: Optional[torch.Tensor] = None,
        pssm_bias: Optional[torch.Tensor] = None,
        pssm_multi: Optional[float] = None,
        pssm_log_odds_flag: Optional[bool] = None,
        pssm_log_odds_mask: Optional[torch.Tensor] = None,
        pssm_bias_flag: Optional[bool] = None,
        tied_pos: Optional[Any] = None,
        tied_beta: Optional[torch.Tensor] = None,
        bias_by_res: Optional[torch.Tensor] = None,
        order_temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Sampling with learned order for tied (oligomeric) positions.

        Groups of tied positions are decoded simultaneously with a shared AA.
        The order-policy selects among groups (mean logits within group).
        """
        device = X.device
        h_V_enc, h_E, E_idx = self._encode(X, mask, residue_idx, chain_encoding_all)

        chain_mask = chain_mask * chain_M_pos * mask

        # Build group schedule: list of lists of position indices
        # First, use random order to establish initial group ordering,
        # then re-order groups by learned order-policy during decoding.
        groups: List[List[int]] = []
        seen: set = set()
        N_nodes = X.size(1)
        for pos in range(N_nodes):
            if pos in seen:
                continue
            group = [item for item in tied_pos if pos in item]
            if group:
                groups.append(group[0])
                seen.update(group[0])
            else:
                groups.append([pos])
                seen.add(pos)

        N_batch = X.size(0)
        all_probs = torch.zeros(
            (N_batch, N_nodes, 21), device=device, dtype=torch.float32,
        )
        h_S = torch.zeros_like(h_V_enc, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        decoded_mask = torch.zeros(N_batch, N_nodes, device=device)
        ordering = torch.zeros(
            (N_batch, N_nodes), dtype=torch.long, device=device,
        )

        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask is not None

        # Pre-fill fixed positions
        for b in range(N_batch):
            for j in range(N_nodes):
                if chain_mask[b, j].item() < 0.5 and mask[b, j].item() > 0.5:
                    S[b, j] = S_true[b, j]
                    h_S[b, j, :] = self.W_s(S_true[b, j:j + 1]).squeeze(0)
                    decoded_mask[b, j] = 1.0

        h_V_stack = [h_V_enc] + [
            torch.zeros_like(h_V_enc, device=device)
            for _ in range(len(self.decoder_layers))
        ]
        h_EX_encoder = cat_neighbors_nodes(
            torch.zeros_like(h_S), h_E, E_idx,
        )
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        remaining_groups = list(range(len(groups)))
        step_count = 0

        while remaining_groups:
            # --- Order selection via p_theta at group level ---
            h_V_current = h_V_stack[-1]
            pos_order_logits = self.W_order_p(h_V_current).squeeze(-1)

            group_scores = torch.full(
                (N_batch, len(groups)), float('-inf'), device=device,
            )
            for gi in remaining_groups:
                grp = groups[gi]
                designable_in_grp = [
                    t for t in grp
                    if chain_mask[0, t].item() > 0.5 and mask[0, t].item() > 0.5
                ]
                if not designable_in_grp:
                    continue
                grp_logits = pos_order_logits[:, designable_in_grp].mean(dim=-1)
                group_scores[:, gi] = grp_logits

            if (group_scores == float('-inf')).all():
                break

            group_probs = F.softmax(
                group_scores / order_temperature, dim=-1,
            )
            selected_gi = torch.multinomial(group_probs, 1).squeeze(-1)  # [B]
            gi_val = selected_gi[0].item()
            t_list = groups[gi_val]
            remaining_groups.remove(gi_val)

            # Check if all members are padding/fixed
            all_fixed = all(
                (mask[:, t] == 0).all() or (chain_mask[:, t] == 0).all()
                for t in t_list
            )
            if all_fixed:
                for t in t_list:
                    S[:, t] = S_true[:, t]
                    h_S[:, t, :] = self.W_s(S_true[:, t:t + 1]).squeeze(1)
                    decoded_mask[:, t] = 1.0
                    ordering[:, step_count] = t
                    step_count += 1
                continue

            # --- Decode each member, accumulate logits ---
            logits_accum = torch.zeros((N_batch, 21), device=device)
            for t in t_list:
                if (mask[:, t] == 0).all():
                    continue

                E_idx_t = E_idx[:, t:t + 1, :]
                h_E_t = h_E[:, t:t + 1, :, :]
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = h_EXV_encoder[:, t:t + 1, :, :]

                neighbor_decoded = torch.gather(
                    decoded_mask.unsqueeze(1).expand(-1, 1, E_idx_t.size(-1)),
                    -1,
                    E_idx_t[:, 0:1, :],
                ).unsqueeze(-1)
                mask_bw_t = neighbor_decoded
                mask_fw_t = 1.0 - neighbor_decoded
                h_EXV_encoder_fw_t = mask_fw_t * h_EXV_encoder_t

                mask_t = mask[:, t:t + 1]
                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(
                        h_V_stack[l], h_ES_t, E_idx_t,
                    )
                    h_V_t = h_V_stack[l][:, t:t + 1, :]
                    h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_fw_t
                    h_V_stack[l + 1][:, t, :] = layer(
                        h_V_t, h_ESV_t, mask_V=mask_t,
                    ).squeeze(1)

                h_V_t_final = h_V_stack[-1][:, t, :]
                beta_t = tied_beta[t] if tied_beta is not None else 1.0
                logits_accum += (
                    beta_t
                    * (self.W_out(h_V_t_final) / temperature)
                    / len(t_list)
                )

            bias_by_res_gathered = bias_by_res[:, t_list[-1], :]
            probs = F.softmax(
                logits_accum
                - constant[None, :] * 1e8
                + constant_bias[None, :] / temperature
                + bias_by_res_gathered / temperature,
                dim=-1,
            )
            if pssm_bias_flag:
                t_rep = t_list[-1]
                pssm_coef_gathered = pssm_coef[:, t_rep]
                pssm_bias_gathered = pssm_bias[:, t_rep]
                probs = (
                    (1 - pssm_multi * pssm_coef_gathered[:, None]) * probs
                    + pssm_multi * pssm_coef_gathered[:, None] * pssm_bias_gathered
                )
            if pssm_log_odds_flag:
                t_rep = t_list[-1]
                pssm_log_odds_mask_gathered = pssm_log_odds_mask[:, t_rep]
                probs_masked = probs * pssm_log_odds_mask_gathered
                probs_masked += probs * 0.001
                probs = probs_masked / torch.sum(
                    probs_masked, dim=-1, keepdim=True,
                )
            if omit_AA_mask_flag:
                t_rep = t_list[-1]
                omit_AA_mask_gathered = omit_AA_mask[:, t_rep]
                probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                probs = probs_masked / torch.sum(
                    probs_masked, dim=-1, keepdim=True,
                )

            S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
            S_t_repeat = (
                chain_mask[:, t_list[0]] * S_t_repeat
                + (1 - chain_mask[:, t_list[0]]) * S_true[:, t_list[0]]
            ).long()

            for t in t_list:
                h_S[:, t, :] = self.W_s(S_t_repeat[:, None]).squeeze(1)
                S[:, t] = S_t_repeat
                all_probs[:, t, :] = probs.float()
                decoded_mask[:, t] = 1.0
                ordering[:, step_count] = t
                step_count += 1

        output_dict: Dict[str, Any] = {
            "S": S,
            "probs": all_probs,
            "decoding_order": ordering,
        }
        return output_dict

    # ==================================================================
    # Importance-sampling log-likelihood estimate
    # ==================================================================

    def compute_loglik_is_q(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        num_samples_eval: int = 8,
    ) -> torch.Tensor:
        """Estimate log p_theta(x | structure) via importance sampling with q_theta.

        Args:
            X: coordinates [B, L, 4, 3].
            S: ground-truth sequence [B, L].
            mask: padding mask [B, L].
            chain_M: chain design mask [B, L].
            residue_idx: residue indices [B, L].
            chain_encoding_all: chain encoding [B, L].
            num_samples_eval: number of importance samples K.

        Returns:
            loglik_per_res: [B] per-residue log-likelihood estimates.
        """
        B, N = S.shape
        device = S.device
        design_mask = chain_M * mask

        h_V_enc, h_E, E_idx = self._encode(X, mask, residue_idx, chain_encoding_all)
        q_logits = self.forward_q(h_V_enc, h_E, E_idx, S, mask, design_mask)

        L_design = design_mask.sum(dim=-1).clamp(min=1.0)

        log_terms: List[torch.Tensor] = []

        for _ in range(num_samples_eval):
            full_perm = self._build_fixed_first_perm(
                design_mask, mask, q_logits.detach(),
            )

            log_probs_k, p_order_logits_k = self.forward_p(
                h_V_enc, h_E, E_idx, S, mask, design_mask,
                permutation=full_perm,
            )

            log_p_token = torch.gather(
                log_probs_k, 2, S.unsqueeze(-1),
            ).squeeze(-1)
            log_p_x_given_z = (log_p_token * design_mask).sum(-1)

            # Step-level mask: which steps correspond to designable positions
            step_design = torch.gather(design_mask, 1, full_perm)

            log_q_all = plackett_luce_log_prob(q_logits, full_perm, mask)
            log_q_safe = log_q_all.masked_fill(~torch.isfinite(log_q_all), 0.0)
            log_q_z = (log_q_safe * step_design).sum(-1)

            log_p_all = plackett_luce_log_prob(
                p_order_logits_k, full_perm, mask,
            )
            log_p_safe = log_p_all.masked_fill(~torch.isfinite(log_p_all), 0.0)
            log_p_z = (log_p_safe * step_design).sum(-1)

            log_weight = log_p_z - log_q_z
            log_terms.append(log_weight + log_p_x_given_z)

        log_terms_stack = torch.stack(log_terms, dim=0)
        log_p_x = (
            torch.logsumexp(log_terms_stack, dim=0)
            - math.log(num_samples_eval)
        )
        loglik_per_res = log_p_x / L_design
        return loglik_per_res

    # ==================================================================
    # Conditional probabilities  (per-position, same as original)
    # ==================================================================

    def conditional_probs(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        randn: torch.Tensor,
        backbone_only: bool = False,
    ) -> torch.Tensor:
        """Compute conditional probabilities p(s_i | rest, structure)."""
        device = X.device
        design_mask = chain_M * mask

        h_V_enc, h_E, E_idx = self._encode(X, mask, residue_idx, chain_encoding_all)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        chain_M_eff = chain_M * mask
        chain_M_np = chain_M_eff.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0, :] == 1)[:, 0]
        log_conditional_probs = torch.zeros(
            [X.shape[0], chain_M.shape[1], 21], device=device,
        ).float()

        for idx in idx_to_loop:
            h_V = torch.clone(h_V_enc)
            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1], device=device).float()
                order_mask[idx] = 0.0
            else:
                order_mask = torch.zeros(chain_M.shape[1], device=device).float()
                order_mask[idx] = 1.0
            decoding_order = torch.argsort(
                (order_mask[None, ] + 0.0001) * torch.abs(randn),
            )
            ar_mask = self._build_generalized_ar_mask(E_idx, decoding_order)
            mask_attend = ar_mask.unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_conditional_probs[:, idx, :] = log_probs[:, idx, :]
        return log_conditional_probs

    # ==================================================================
    # Unconditional probabilities  (same as original)
    # ==================================================================

    def unconditional_probs(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unconditional probabilities p(s_i | structure only)."""
        device = X.device

        h_V_enc, h_E, E_idx = self._encode(X, mask, residue_idx, chain_encoding_all)

        h_EX_encoder = cat_neighbors_nodes(
            torch.zeros_like(h_V_enc), h_E, E_idx,
        )
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        mask_attend = torch.zeros(
            [X.shape[0], X.shape[1], E_idx.shape[-1]], device=device,
        ).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        h_V = h_V_enc.clone()
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
