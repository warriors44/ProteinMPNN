from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_attention import (
    TransformerLayer,
    MPNNLayer,
    cat_neighbors_nodes,
    gather_nodes,
)
from .protein_features import ProteinFeatures


class Struct2SeqAO(nn.Module):
    """Struct2Seq with any-order (AO-ARM) training objective.

    This variant shares the same encoder / decoder architecture as the
    original Struct2Seq / Struct2SeqLO models, but uses the any-order ELBO
    from Eq. (10) in Wang et al. (arXiv:2503.05979) for training.

    Differences from Struct2SeqLO:
      - No learned order heads (no W_order_p / W_order_q).
      - No variational order posterior q_theta.
      - Training only uses token log-likelihoods under a uniform prior
        over permutations (AO-ARM).
    """

    def __init__(
        self,
        num_letters: int,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 20,
        k_neighbors: int = 30,
        protein_features: str = "full",
        augment_eps: float = 0.0,
        dropout: float = 0.1,
        forward_attention_decoder: bool = True,
        encoder_arch: str = "transformer",
        decoder_arch: str = "transformer",
    ) -> None:
        super(Struct2SeqAO, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        if encoder_arch not in ("transformer", "mpnn"):
            raise ValueError(
                f"encoder_arch must be 'transformer' or 'mpnn', got {encoder_arch!r}.",
            )
        if decoder_arch not in ("transformer", "mpnn"):
            raise ValueError(
                f"decoder_arch must be 'transformer' or 'mpnn', got {decoder_arch!r}.",
            )

        self.encoder_arch = encoder_arch
        self.decoder_arch = decoder_arch

        # Featurization
        self.features = ProteinFeatures(
            node_features,
            edge_features,
            top_k=k_neighbors,
            features_type=protein_features,
            augment_eps=augment_eps,
            dropout=dropout,
        )

        # Embedding layers (shared)
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        enc_layer = TransformerLayer if encoder_arch == "transformer" else MPNNLayer
        dec_layer = TransformerLayer if decoder_arch == "transformer" else MPNNLayer

        # Encoder
        self.encoder_layers = nn.ModuleList(
            [enc_layer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers)]
        )

        # Decoder
        self.forward_attention_decoder = forward_attention_decoder
        self.decoder_layers = nn.ModuleList(
            [dec_layer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_decoder_layers)]
        )

        # Token prediction head (shared)
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _encode(
        self,
        X: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run structure encoder.

        Returns:
            h_V: encoded node features [B, N, H]
            h_E: encoded edge features [B, N, K, H]
            E_idx: neighbor indices [B, N, K]
            mask_attend_enc: encoder attention mask [B, N, K]
        """
        V, E, E_idx = self.features(X, L, mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for enc_layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = enc_layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        return h_V, h_E, E_idx, mask_attend

    # ------------------------------------------------------------------
    # Generalized autoregressive masks
    # ------------------------------------------------------------------

    def _generalized_autoregressive_mask(
        self,
        E_idx: torch.Tensor,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        """Build autoregressive mask from an arbitrary decoding permutation.

        Args:
            E_idx: neighbor indices [B, N, K].
            permutation: [B, N] where permutation[b, i] = index of position
                decoded at step i.

        Returns:
            mask: [B, N, K] with 1 where neighbor was decoded earlier in
                the permutation (i.e. is in the \"past\").
        """
        B, N = permutation.shape
        device = permutation.device

        rank = torch.zeros(B, N, dtype=torch.long, device=device)
        rank.scatter_(1, permutation, torch.arange(N, device=device).unsqueeze(0).expand(B, -1))

        rank_self = rank.unsqueeze(-1)
        rank_flat = rank.unsqueeze(1).expand(-1, N, -1)
        rank_neighbors = torch.gather(rank_flat, 2, E_idx)

        mask = (rank_neighbors < rank_self).float()
        return mask

    def _build_partial_ar_mask(
        self,
        E_idx: torch.Tensor,
        full_perm: torch.Tensor,
        i_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Build autoregressive mask for partial decode (z_{<i}).

        Decoded positions j=z_k see only z_{<k}. Remaining positions see
        z_{<i} and not each other. This matches the probabilistic model
        used in the LO-ARM paper and enables exact expectation over z_i.

        Args:
            E_idx: neighbor indices [B, N, K].
            full_perm: full permutation [B, N] where full_perm[b, k] is the
                position decoded at step k.
            i_samples: [B] number of decoded steps so far (1-indexed).

        Returns:
            mask: [B, N, K] with 1 where neighbor is in the visible past.
        """
        B, N = full_perm.shape
        device = full_perm.device

        rank = torch.zeros(B, N, dtype=torch.long, device=device)
        rank.scatter_(
            1,
            full_perm,
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

    # ------------------------------------------------------------------
    # Generalized causal decoder
    # ------------------------------------------------------------------

    def forward_p(
        self,
        h_V_enc: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        permutation: torch.Tensor | None = None,
        ar_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run decoder with a generalized autoregressive mask.

        This is identical to the p_theta path in Struct2SeqLO, except that
        order logits are not produced (AO-ARM uses a uniform order-policy).

        Args:
            h_V_enc: encoder output [B, N, H].
            h_E: edge embeddings [B, N, K, H].
            E_idx: neighbor indices [B, N, K].
            S: ground-truth sequence [B, N] (teacher forcing).
            mask: padding mask [B, N].
            permutation: decoding order [B, N]. Required if ar_mask is None.
            ar_mask: optional precomputed mask [B, N, K]. Used for partial
                decode (e.g. with _build_partial_ar_mask).

        Returns:
            log_probs: [B, N, vocab] log-probabilities for each position.
            None: placeholder for compatibility with Struct2SeqLO interface.
        """
        if ar_mask is not None:
            mask_attend = ar_mask.unsqueeze(-1)
        elif permutation is not None:
            ar_mask = self._generalized_autoregressive_mask(E_idx, permutation)
            mask_attend = ar_mask.unsqueeze(-1)
        else:
            raise ValueError("Either permutation or ar_mask must be provided")

        h_V = h_V_enc.clone()
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        h_ES_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx)

        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend

        if self.forward_attention_decoder:
            mask_fw = mask_1D * (1.0 - mask_attend)
            h_ESV_encoder_fw = mask_fw * h_ESV_encoder
        else:
            h_ESV_encoder_fw = 0

        for dec_layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = dec_layer(h_V, h_ESV, mask_V=mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, None

    # ------------------------------------------------------------------
    # AO-ARM ELBO (Eq. 10) with uniform order-policy
    # ------------------------------------------------------------------

    def compute_elbo_ao(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute AO-ARM ELBO (Eq. 10) with uniform ordering.

        For each batch element b:
          1. Sample i_b ~ Uniform(1, ..., L_b).
          2. Sample a random permutation z (uniform over positions).
          3. Construct a partial AR mask so that:
             - decoded positions j = z_k see z_{<k}
             - remaining positions see z_{<i} but not each other
          4. Compute log p_theta(x_j | x_{z_{<i}}, s) for all j.
          5. Apply Eq. (10):
               F_b = (L_b / (L_b - i_b + 1)) * sum_{j in z_{>=i}} log p(x_j | x_{z_{<i}}, s)
          6. Loss is -F_b, averaged over residues.
        """
        B, N = S.shape
        device = S.device

        h_V_enc, h_E, E_idx, _ = self._encode(X, L, mask)

        # Lengths per batch element
        L_tensor = torch.tensor(L, dtype=torch.float, device=device)

        # Sample i ~ Uniform(1, ..., L_b)
        i_samples = (torch.rand(B, device=device) * L_tensor).long() + 1

        # Sample a full permutation per batch element, restricted to valid
        # (non-padded) positions according to `mask`. Padded positions are
        # always placed at the end of the permutation.
        full_perm = torch.empty(B, N, dtype=torch.long, device=device)
        for b in range(B):
            valid_idx = torch.nonzero(mask[b] > 0.5, as_tuple=False).squeeze(-1)
            invalid_idx = torch.nonzero(mask[b] <= 0.5, as_tuple=False).squeeze(-1)

            num_valid = int(valid_idx.numel())
            perm_valid = valid_idx[torch.randperm(num_valid, device=device)]

            if invalid_idx.numel() > 0:
                full_perm[b] = torch.cat([perm_valid, invalid_idx], dim=0)
            else:
                full_perm[b] = perm_valid

        # Build partial AR mask and run decoder
        ar_mask = self._build_partial_ar_mask(E_idx, full_perm, i_samples)
        log_probs, _ = self.forward_p(h_V_enc, h_E, E_idx, S, mask, ar_mask=ar_mask)

        # log p(x_j | x_{z<i}, s)
        log_p_token = torch.gather(log_probs, 2, S.unsqueeze(-1)).squeeze(-1)

        # Determine decoded vs remaining positions from permutation / i_samples
        rank = torch.zeros(B, N, dtype=torch.long, device=device)
        rank.scatter_(
            1,
            full_perm,
            torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
        )
        decoded_mask = (rank < (i_samples - 1).unsqueeze(1)).float()
        remaining_mask = (1.0 - decoded_mask) * mask

        # Sum over remaining positions and rescale by L / (L - i + 1)
        sum_rem = (log_p_token * remaining_mask).sum(-1)
        cnt_rem = remaining_mask.sum(-1)  # should equal L_b - i_b + 1

        # Guard against any numerical edge cases (should not occur in practice)
        cnt_rem = cnt_rem.clamp(min=1.0)

        F_b = (L_tensor / cnt_rem) * sum_rem  # [B]

        loss_per_elem = -F_b
        loss = loss_per_elem.sum() / L_tensor.sum()

        with torch.no_grad():
            # Per-residue ELBO (for logging only)
            elbo_per_res = (F_b / L_tensor).mean()

            # Standard NLL under fixed N->C ordering for monitoring
            log_probs_all = self.forward(X, S, L, mask)
            log_p_token_all = torch.gather(
                log_probs_all, 2, S.unsqueeze(-1)
            ).squeeze(-1)
            nll_avg = -(log_p_token_all * mask).sum() / mask.sum()

        info = {
            "elbo": elbo_per_res,
            "nll": nll_avg,
            "i_mean": i_samples.float().mean(),
        }
        return loss, info

    # ------------------------------------------------------------------
    # Standard forward (fixed N->C order) for evaluation
    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
        permutation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard forward pass using sequential (N->C) ordering.

        This matches the original Struct2Seq interface and is used for
        evaluation / validation NLL.
        """
        if permutation is None:
            B, N = S.shape
            device = S.device
            permutation = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        else:
            if permutation.dim() != 2:
                raise ValueError("permutation must be a [B, N] tensor")
            if permutation.shape != S.shape:
                raise ValueError(
                    f"permutation shape {tuple(permutation.shape)} must match S shape {tuple(S.shape)}"
                )

        h_V_enc, h_E, E_idx, _ = self._encode(X, L, mask)
        log_probs, _ = self.forward_p(h_V_enc, h_E, E_idx, S, mask, permutation)
        return log_probs

