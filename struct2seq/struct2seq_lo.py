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
from .gumbel import gumbel_top_k, plackett_luce_log_prob


class Struct2SeqLO(nn.Module):
    """Struct2Seq with Learning-Order autoregressive decoding.

    Extends the original Struct2Seq architecture with a learned decoding order
    based on the Plackett-Luce distribution and Gumbel Top-K sampling,
    following the framework of Wang et al. (arXiv:2503.05979).

    The model has three probability components:
      - p_theta(x_{z_i} | x_{z_{<i}}, s): token prediction (shared decoder)
      - p_theta(z_i | z_{<i}, x_{z_{<i}}, s): order prior (W_order_p head)
      - q_theta(z_i | z_{<i}, x, s): variational order posterior (W_order_q head)
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
        protein_features: str = 'full',
        augment_eps: float = 0.,
        dropout: float = 0.1,
        forward_attention_decoder: bool = True,
        num_samples: int = 2,
        p_encoder_arch: str = "transformer",
        p_decoder_arch: str = "transformer",
        q_encoder_arch: str = "transformer",
        q_decoder_arch: str = "transformer",
        separate_encoder: bool = False,
        separate_decoder: bool = False,
    ) -> None:
        super(Struct2SeqLO, self).__init__()

        if num_samples < 2:
            raise ValueError("num_samples must be >= 2 for RLOO estimator.")

        for name, val in [
            ("p_encoder_arch", p_encoder_arch),
            ("p_decoder_arch", p_decoder_arch),
            ("q_encoder_arch", q_encoder_arch),
            ("q_decoder_arch", q_decoder_arch),
        ]:
            if val not in ("transformer", "mpnn"):
                raise ValueError(
                    f"{name} must be 'transformer' or 'mpnn', got {val!r}.",
                )

        if not separate_encoder and p_encoder_arch != q_encoder_arch:
            raise ValueError(
                "p_encoder_arch and q_encoder_arch must match when "
                "separate_encoder is False.",
            )
        if not separate_decoder and p_decoder_arch != q_decoder_arch:
            raise ValueError(
                "p_decoder_arch and q_decoder_arch must match when "
                "separate_decoder is False.",
            )

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.p_encoder_arch = p_encoder_arch
        self.p_decoder_arch = p_decoder_arch
        self.q_encoder_arch = q_encoder_arch
        self.q_decoder_arch = q_decoder_arch
        self.separate_encoder = separate_encoder
        self.separate_decoder = separate_decoder

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

        _layer_cls = {
            "transformer": TransformerLayer,
            "mpnn": MPNNLayer,
        }

        # p_theta encoder
        self.encoder_layers = nn.ModuleList([
            _layer_cls[p_encoder_arch](hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # p_theta decoder
        self.forward_attention_decoder = forward_attention_decoder
        self.decoder_layers = nn.ModuleList([
            _layer_cls[p_decoder_arch](hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        # Token prediction head
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Order prediction heads
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

        # Separate q_theta encoder
        if self.separate_encoder:
            self.q_encoder_layers = nn.ModuleList([
                _layer_cls[q_encoder_arch](hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ])

        # Separate q_theta decoder
        if self.separate_decoder:
            self.q_decoder_layers = nn.ModuleList([
                _layer_cls[q_decoder_arch](hidden_dim, hidden_dim * 3, dropout=dropout)
                for _ in range(num_decoder_layers)
            ])
            self.W_order_q_sep = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

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
        """Run p_theta structure encoder.

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

    def _encode_q(
        self,
        X: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run q_theta structure encoder.

        When separate_encoder is True, uses dedicated q_encoder_layers.
        Otherwise delegates to _encode (shared layers).

        Returns:
            h_V: encoded node features [B, N, H]
            h_E: encoded edge features [B, N, K, H]
            E_idx: neighbor indices [B, N, K]
            mask_attend_enc: encoder attention mask [B, N, K]
        """
        if not self.separate_encoder:
            return self._encode(X, L, mask)

        V, E, E_idx = self.features(X, L, mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for enc_layer in self.q_encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = enc_layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        return h_V, h_E, E_idx, mask_attend

    # ------------------------------------------------------------------
    # Generalized autoregressive mask
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
                the permutation (i.e. is in the "past").
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
        and sampling behavior.

        Args:
            E_idx: neighbor indices [B, N, K].
            full_perm: full permutation [B, N] where full_perm[b, k] = position
                decoded at step k.
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

    # ------------------------------------------------------------------
    # q_theta path: unmasked decoder for variational posterior
    # ------------------------------------------------------------------

    def forward_q(
        self,
        h_V_enc: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute q_theta order logits using the unmasked (full info) decoder.

        All sequence information is visible (no causal mask).

        Args:
            h_V_enc: encoder output [B, N, H].
            h_E: edge embeddings [B, N, K, H].
            E_idx: neighbor indices [B, N, K].
            S: ground-truth sequence [B, N].
            mask: padding mask [B, N].

        Returns:
            q_logits: [B, N] per-position order logits.
        """
        h_V = h_V_enc.clone()
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        if self.separate_decoder:
            decoder_layers = self.q_decoder_layers
            order_head = self.W_order_q_sep
        else:
            decoder_layers = self.decoder_layers
            order_head = self.W_order_q

        for dec_layer in decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_V = dec_layer(h_V, h_ESV, mask_V=mask, mask_attend=mask_attend)

        q_logits = order_head(h_V).squeeze(-1)
        q_logits = q_logits.masked_fill(mask == 0, float('-inf'))
        return q_logits

    # ------------------------------------------------------------------
    # p_theta path: generalized causal decoder
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run decoder with a generalized autoregressive mask.

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
            p_order_logits: [B, N] per-position order logits for p_theta.
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
            mask_fw = mask_1D * (1. - mask_attend)
            h_ESV_encoder_fw = mask_fw * h_ESV_encoder
        else:
            h_ESV_encoder_fw = 0

        for dec_layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = dec_layer(h_V, h_ESV, mask_V=mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)

        p_order_logits = self.W_order_p(h_V).squeeze(-1)
        p_order_logits = p_order_logits.masked_fill(mask == 0, float('-inf'))

        return log_probs, p_order_logits

    # ------------------------------------------------------------------
    # F_theta with exact expectation over z_i (Eq. 8)
    # ------------------------------------------------------------------

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
            log_probs: [B, N, vocab] from forward_p (with partial ar_mask).
            p_order_logits: [B, N] from forward_p.
            q_logits: [B, N] from forward_q (not detached).
            S: ground-truth sequence [B, N].
            remaining_mask: [B, N] binary, 1 for undecoded valid positions.

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

    # ------------------------------------------------------------------
    # Paper-faithful ELBO (Algorithm 1, Eqs. 8/9/11)
    # ------------------------------------------------------------------

    def compute_elbo_paper(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """ELBO loss following Algorithm 1 of Wang et al. (arXiv:2503.05979).

        1. Sample i ~ Uniform(1, ..., L_b) per batch element.
        2. Draw K independent partial permutations z^k_{<i} of length i-1
           from the Plackett-Luce q_theta via Gumbel Top-K, where
           K = self.num_samples (RLOO samples).
        3. Compute F_theta(z^k_{<i}, x) with the *exact* expectation over z_i
           (Eq. 8: sum over all remaining positions).
        4. Construct the loss whose gradient equals the REINFORCE
           leave-one-out estimator of Eq. 11.

        Args:
            X: coordinates [B, N, 4, 3].
            S: ground-truth sequence [B, N].
            L: lengths array (np).
            mask: padding mask [B, N].

        Returns:
            loss: scalar loss to minimise.
            info: monitoring dict.
        """
        B, N = S.shape
        device = S.device

        h_V_enc, h_E, E_idx, _ = self._encode(X, L, mask)

        if self.separate_encoder:
            h_V_enc_q, _, _, _ = self._encode_q(X, L, mask)
        else:
            h_V_enc_q = h_V_enc
        q_logits = self.forward_q(h_V_enc_q, h_E, E_idx, S, mask)

        L_tensor = torch.tensor(L, dtype=torch.float, device=device)

        i_samples = (
            torch.rand(B, device=device) * L_tensor
        ).long() + 1

        F_values: list[torch.Tensor] = []
        log_q_values: list[torch.Tensor] = []

        K = self.num_samples

        for _ in range(K):
            full_perm = gumbel_top_k(q_logits.detach())

            ar_mask = self._build_partial_ar_mask(E_idx, full_perm, i_samples)
            log_probs_k, p_order_logits_k = self.forward_p(
                h_V_enc, h_E, E_idx, S, mask, ar_mask=ar_mask,
            )

            rank = torch.zeros(B, N, dtype=torch.long, device=device)
            rank.scatter_(
                1, full_perm,
                torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
            )
            decoded_mask = (
                rank < (i_samples - 1).unsqueeze(1)
            ).float()
            remaining_mask = (1.0 - decoded_mask) * mask

            F_k = self._compute_F_theta(
                log_probs_k, p_order_logits_k, q_logits, S, remaining_mask,
            )
            F_values.append(F_k)

            log_q_all = plackett_luce_log_prob(q_logits, full_perm, mask)
            step_mask = (
                torch.arange(N, device=device).unsqueeze(0)
                < (i_samples - 1).unsqueeze(1)
            ).float()
            log_q_partial = (log_q_all * step_mask).sum(-1)
            log_q_values.append(log_q_partial)

        # Stack along new sample dimension: [K, B]
        F_stack = torch.stack(F_values, dim=0)
        log_q_stack = torch.stack(log_q_values, dim=0)

        if K < 2:
            raise ValueError("RLOO estimator requires num_samples >= 2.")

        # Mean F over samples (p_theta ELBO part)
        F_mean = F_stack.mean(dim=0)  # [B]

        # Leave-one-out baselines for each sample k
        sum_F = F_stack.sum(dim=0)  # [B]
        F_minus_k = (sum_F.unsqueeze(0) - F_stack) / float(K - 1)  # [K, B]

        # Advantages for q_theta (no gradient through F_minus_k)
        adv = (F_stack - F_minus_k).detach()  # [K, B]

        # RLOO term: average over K samples
        rloo_term = (adv * log_q_stack).mean(dim=0)  # [B]

        loss_per_elem = -L_tensor * (F_mean + rloo_term)
        loss = loss_per_elem.sum() / L_tensor.sum()

        with torch.no_grad():
            elbo_per_res = F_mean.mean()
            delta_F_abs = (F_stack - F_mean.unsqueeze(0)).abs().mean()

            # Decoder-based NLL under fixed N->C ordering for monitoring
            log_probs_all = self.forward(X, S, L, mask)
            log_p_token_all = torch.gather(
                log_probs_all, 2, S.unsqueeze(-1),
            ).squeeze(-1)
            nll_avg = -(log_p_token_all * mask).sum() / mask.sum()

        info = {
            'elbo': elbo_per_res,
            'nll': nll_avg,
            'F_mean': F_mean.mean(),
            'delta_F_abs': delta_F_abs,
            'i_mean': i_samples.float().mean(),
        }
        return loss, info

    # ------------------------------------------------------------------
    # q-based importance sampling estimate of log p_theta(x | s)
    # ------------------------------------------------------------------

    def compute_loglik_is_q(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
        num_samples_eval: int | None = None,
    ) -> torch.Tensor:
        """Estimate log p_theta(x | s) via importance sampling with q_theta(z | x, s).

        For each batch element b:
          1. Sample K permutations z^(k) ~ q_theta(z | x, s) via Plackett-Luce.
          2. For each z^(k), run forward_p with permutation=z^(k) under teacher forcing
             to obtain log p_theta(x | z^(k), s).
          3. Compute log p_theta(z^(k) | s) and log q_theta(z^(k) | x, s) using the
             Plackett-Luce log-probabilities.
          4. Form importance weights w_k = p(z^(k) | s) / q(z^(k) | x, s) and use
             log-sum-exp to approximate log p_theta(x | s).

        Args:
            X: coordinates [B, N, 4, 3].
            S: ground-truth sequence [B, N].
            L: lengths array (np).
            mask: padding mask [B, N].
            num_samples_eval: number of importance samples K. If None, a default
                value of 8 is used.

        Returns:
            loglik_per_res: [B] per-residue log-likelihood estimates
                log p_theta(x | s) / L_b for each batch element.
        """
        B, N = S.shape
        device = S.device

        h_V_enc, h_E, E_idx, _ = self._encode(X, L, mask)

        if self.separate_encoder:
            h_V_enc_q, _, _, _ = self._encode_q(X, L, mask)
        else:
            h_V_enc_q = h_V_enc
        q_logits = self.forward_q(h_V_enc_q, h_E, E_idx, S, mask)

        L_tensor = torch.tensor(L, dtype=torch.float32, device=device)

        K_eval = num_samples_eval if num_samples_eval is not None else 8
        if K_eval < 1:
            raise ValueError("num_samples_eval must be >= 1.")

        log_terms: list[torch.Tensor] = []

        for _ in range(K_eval):
            full_perm = gumbel_top_k(q_logits.detach())

            log_probs_k, p_order_logits_k = self.forward_p(
                h_V_enc,
                h_E,
                E_idx,
                S,
                mask,
                permutation=full_perm,
            )

            log_p_token = torch.gather(
                log_probs_k,
                2,
                S.unsqueeze(-1),
            ).squeeze(-1)
            log_p_x_given_z = (log_p_token * mask).sum(-1)

            log_q_all = plackett_luce_log_prob(q_logits, full_perm, mask)
            log_q_z = (log_q_all * mask).sum(-1)

            log_p_all = plackett_luce_log_prob(p_order_logits_k, full_perm, mask)
            log_p_z = (log_p_all * mask).sum(-1)

            log_weight = log_p_z - log_q_z
            log_terms.append(log_weight + log_p_x_given_z)

        log_terms_stack = torch.stack(log_terms, dim=0)
        log_p_x_given_s = torch.logsumexp(log_terms_stack, dim=0) - float(np.log(K_eval))

        loglik_per_res = log_p_x_given_s / L_tensor
        return loglik_per_res

    # ------------------------------------------------------------------
    # p_theta-only Monte Carlo estimate of log p_theta(x | s)
    # ------------------------------------------------------------------

    def compute_loglik_mc_p(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
        num_samples_eval: int | None = None,
    ) -> torch.Tensor:
        """Estimate log p_theta(x | s) via Monte Carlo over orders from p_theta.

        For each batch element b:
          1. Sequentially sample a permutation z^(k) ~ p_theta(z | s) using the
             p_theta order head with partial autoregressive masks:
               - Start with no decoded positions.
               - At each step i, build a partial AR mask corresponding to the
                 current prefix z_{<i} via _build_partial_ar_mask.
               - Run forward_p under teacher forcing to obtain order logits and
                 sample the next position z_i among remaining positions.
          2. After a full permutation is sampled, run forward_p once with the
             full permutation to compute log p_theta(x | z^(k), s).
          3. Average over K_eval such samples in log-space to approximate
             log p_theta(x | s).

        Args:
            X: coordinates [B, N, 4, 3].
            S: ground-truth sequence [B, N].
            L: lengths array (np).
            mask: padding mask [B, N].
            num_samples_eval: number of Monte Carlo samples K_eval. If None,
                a default value of 8 is used.

        Returns:
            loglik_per_res: [B] per-residue log-likelihood estimates
                log p_theta(x | s) / L_b for each batch element.
        """
        B, N = S.shape
        device = S.device

        h_V_enc, h_E, E_idx, _ = self._encode(X, L, mask)

        L_tensor = torch.tensor(L, dtype=torch.float32, device=device)

        K_eval = num_samples_eval if num_samples_eval is not None else 8
        if K_eval < 1:
            raise ValueError("num_samples_eval must be >= 1.")

        log_terms: list[torch.Tensor] = []

        for _ in range(K_eval):
            # Initialise decoded mask and permutation for this sample.
            decoded_mask = torch.zeros(B, N, dtype=torch.float32, device=device)
            full_perm = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).clone()

            while True:
                any_active = False

                for b in range(B):
                    length_b = int(L[b])
                    mask_b = mask[b]  # [N]

                    # Remaining valid, undecoded positions for this batch element.
                    remaining_mask_b = mask_b * (1.0 - decoded_mask[b])
                    if remaining_mask_b.sum().item() == 0:
                        continue

                    any_active = True

                    decoded_count_b = int((decoded_mask[b] * mask_b).sum().item())
                    if decoded_count_b >= length_b:
                        continue

                    i_samples_b = torch.tensor(
                        [min(decoded_count_b + 1, length_b)],
                        dtype=torch.long,
                        device=device,
                    )

                    # Build partial AR mask for this prefix using current permutation.
                    ar_mask_b = self._build_partial_ar_mask(
                        E_idx[b:b+1],
                        full_perm[b:b+1],
                        i_samples_b,
                    )

                    # Run forward_p for this single element to get order logits.
                    log_probs_step_b, p_order_logits_step_b = self.forward_p(
                        h_V_enc[b:b+1],
                        h_E[b:b+1],
                        E_idx[b:b+1],
                        S[b:b+1],
                        mask[b:b+1],
                        ar_mask=ar_mask_b,
                    )
                    # Silence unused variable warning.
                    _ = log_probs_step_b

                    remaining_mask_b_2d = remaining_mask_b.unsqueeze(0)
                    logits_step_b = p_order_logits_step_b.clone()
                    logits_step_b = logits_step_b.masked_fill(
                        remaining_mask_b_2d == 0,
                        float('-inf'),
                    )

                    order_probs_b = F.softmax(logits_step_b, dim=-1)
                    pos_b = torch.multinomial(order_probs_b, 1).squeeze(0).squeeze(-1)

                    # Maintain consistency between permutation and decoded mask by
                    # swapping the newly selected position into the next prefix slot.
                    decoded_count_b_tensor = int(
                        (decoded_mask[b] * mask_b).sum().item(),
                    )
                    perm_b = full_perm[b]
                    idx_in_perm = (perm_b == pos_b).nonzero(as_tuple=False)[0, 0]

                    full_perm[b, idx_in_perm] = perm_b[decoded_count_b_tensor]
                    full_perm[b, decoded_count_b_tensor] = pos_b

                    decoded_mask[b, pos_b] = 1.0

                if not any_active:
                    break

            # Full permutation for this Monte Carlo sample; compute log p(x | z, s).
            log_probs_k, _ = self.forward_p(
                h_V_enc,
                h_E,
                E_idx,
                S,
                mask,
                permutation=full_perm,
            )

            log_p_token = torch.gather(
                log_probs_k,
                2,
                S.unsqueeze(-1),
            ).squeeze(-1)
            log_p_x_given_z = (log_p_token * mask).sum(-1)

            log_terms.append(log_p_x_given_z)

        log_terms_stack = torch.stack(log_terms, dim=0)
        log_p_x_given_s = torch.logsumexp(log_terms_stack, dim=0) - float(np.log(K_eval))

        loglik_per_res = log_p_x_given_s / L_tensor
        return loglik_per_res

    # ------------------------------------------------------------------
    # p-theta-only Monte Carlo estimate of log p_theta(x | s)
    # ------------------------------------------------------------------

    def compute_loglik_mc_p(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
        num_samples_eval: int | None = None,
    ) -> torch.Tensor:
        """Estimate log p_theta(x | s) via Monte Carlo sampling from p_theta(z | x_{z_{<i}}, s).

        For each batch element b:
          1. Sequentially sample a permutation z^(k) under the learned order prior p_theta,
             using the partial autoregressive mask (_build_partial_ar_mask) and teacher forcing.
          2. Given the completed permutation, run forward_p once with the full permutation
             to obtain log p_theta(x | z^(k), s).
          3. Average over K permutations with a simple Monte Carlo estimator:
                 log p_theta(x | s) ~= log (1/K sum_k p_theta(x | z^(k), s)).

        Args:
            X: coordinates [B, N, 4, 3].
            S: ground-truth sequence [B, N].
            L: lengths array (np).
            mask: padding mask [B, N].
            num_samples_eval: number of Monte Carlo samples K. If None, a default
                value of 8 is used.

        Returns:
            loglik_per_res: [B] per-residue log-likelihood estimates
                log p_theta(x | s) / L_b for each batch element.
        """
        B, N = S.shape
        device = S.device

        h_V_enc, h_E, E_idx, _ = self._encode(X, L, mask)

        L_tensor = torch.tensor(L, dtype=torch.float32, device=device)

        K_eval = num_samples_eval if num_samples_eval is not None else 8
        if K_eval < 1:
            raise ValueError("num_samples_eval must be >= 1.")

        max_steps = int(L_tensor.max().item())

        log_terms: list[torch.Tensor] = []

        for _ in range(K_eval):
            # decode_step[b, j] stores the step index at which position j was decoded
            # (0-based). Undecoded positions have value N.
            decode_step = torch.full(
                (B, N),
                fill_value=N,
                dtype=torch.long,
                device=device,
            )
            decoded_mask = torch.zeros(B, N, dtype=torch.float32, device=device)

            # Sequentially sample a permutation under p_theta
            for _step in range(max_steps):
                remaining = mask * (1.0 - decoded_mask)

                # If no valid positions remain in any sequence, stop.
                if remaining.sum() == 0:
                    break

                # Build a full permutation consistent with current decoded order.
                # Positions are ordered by decode_step (decoded first, in order of
                # their decode time; undecoded and padded positions follow).
                full_perm = torch.argsort(decode_step, dim=1)

                decoded_count = decoded_mask.sum(dim=1).long()
                i_samples = torch.clamp(
                    decoded_count + 1,
                    max=L_tensor.long(),
                )

                ar_mask = self._build_partial_ar_mask(E_idx, full_perm, i_samples)

                log_probs_step, p_order_logits_step = self.forward_p(
                    h_V_enc,
                    h_E,
                    E_idx,
                    S,
                    mask,
                    ar_mask=ar_mask,
                )

                # Sample next position z_i from p_theta(z_i | z_{<i}, x_{z_{<i}}, s)
                order_logits = p_order_logits_step.masked_fill(
                    remaining == 0,
                    float("-inf"),
                )

                # Only sample for batch elements that still have remaining positions.
                active = remaining.sum(dim=1) > 0
                if not torch.any(active):
                    break

                order_logits_active = order_logits[active]
                order_probs_active = F.softmax(order_logits_active, dim=-1)

                pos_active = torch.multinomial(order_probs_active, 1).squeeze(-1)

                batch_idx = torch.arange(B, device=device)[active]
                decoded_mask[batch_idx, pos_active] = 1.0
                decoded_count_active = decoded_count[batch_idx]
                decode_step[batch_idx, pos_active] = decoded_count_active

            # After finishing sequential sampling, construct the final permutation
            # from the decode_step matrix.
            full_perm_final = torch.argsort(decode_step, dim=1)

            log_probs_k, _ = self.forward_p(
                h_V_enc,
                h_E,
                E_idx,
                S,
                mask,
                permutation=full_perm_final,
            )

            log_p_token = torch.gather(
                log_probs_k,
                2,
                S.unsqueeze(-1),
            ).squeeze(-1)
            log_p_x_given_z = (log_p_token * mask).sum(-1)

            log_terms.append(log_p_x_given_z)

        log_terms_stack = torch.stack(log_terms, dim=0)
        log_p_x_given_s = torch.logsumexp(log_terms_stack, dim=0) - float(np.log(K_eval))

        loglik_per_res = log_p_x_given_s / L_tensor
        return loglik_per_res

    # ------------------------------------------------------------------
    # Forward (convenience wrapper matching original Struct2Seq interface)
    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Standard forward pass using sequential (N->C) ordering.

        Compatible with the original Struct2Seq interface for evaluation.
        """
        B, N = S.shape
        device = S.device
        permutation = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

        h_V_enc, h_E, E_idx, _ = self._encode(X, L, mask)
        log_probs, _ = self.forward_p(h_V_enc, h_E, E_idx, S, mask, permutation)
        return log_probs

    # ------------------------------------------------------------------
    # Sampling with learned order
    # ------------------------------------------------------------------

    def sample(
        self,
        X: torch.Tensor,
        L: np.ndarray,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive sampling using the learned order (p_theta only).

        At each step:
          1. Compute order logits for remaining positions -> select next position.
          2. Compute amino acid logits for the selected position -> sample token.

        Args:
            X: coordinates [B, N, 4, 3].
            L: lengths array.
            mask: padding mask [B, N].
            temperature: sampling temperature.

        Returns:
            S: sampled sequence [B, N].
            ordering: [B, N] the order in which positions were decoded.
        """
        h_V_enc, h_E, E_idx, _ = self._encode(X, L, mask)

        B, N_nodes = X.size(0), X.size(1)
        device = X.device

        S = torch.zeros(B, N_nodes, dtype=torch.long, device=device)
        ordering = torch.zeros(B, N_nodes, dtype=torch.long, device=device)
        decoded_mask = torch.zeros(B, N_nodes, device=device)
        h_S = torch.zeros_like(h_V_enc)

        h_V_stack = [h_V_enc.clone()] + [
            torch.zeros_like(h_V_enc) for _ in range(len(self.decoder_layers))
        ]

        for t in range(N_nodes):
            remaining = (mask - decoded_mask)

            if remaining.sum() == 0:
                break

            # --- Select next position via p_theta order head ---
            h_V_current = h_V_stack[-1]
            order_logits = self.W_order_p(h_V_current).squeeze(-1)
            order_logits = order_logits.masked_fill(remaining == 0, float('-inf'))
            order_logits = order_logits / temperature

            order_probs = F.softmax(order_logits, dim=-1)
            pos = torch.multinomial(order_probs, 1).squeeze(-1)
            ordering[:, t] = pos

            # --- Decode amino acid at the selected position ---
            for b in range(B):
                p = pos[b].item()
                E_idx_p = E_idx[b:b+1, p:p+1, :]
                h_E_p = h_E[b:b+1, p:p+1, :, :]
                h_ES_p = cat_neighbors_nodes(
                    h_S[b:b+1], h_E_p, E_idx_p,
                )

                ar_mask_p = decoded_mask[b:b+1].unsqueeze(0)
                neighbor_decoded = torch.gather(
                    decoded_mask[b:b+1].unsqueeze(1).expand(-1, 1, E_idx_p.size(-1)),
                    -1,
                    E_idx_p[:, 0:1, :].squeeze(1).unsqueeze(0),
                ).unsqueeze(-1)
                mask_bw_p = neighbor_decoded
                mask_fw_p = 1.0 - neighbor_decoded

                h_ES_encoder_p = cat_neighbors_nodes(
                    torch.zeros_like(h_S[b:b+1]), h_E_p, E_idx_p,
                )
                h_ESV_encoder_p = cat_neighbors_nodes(
                    h_V_enc[b:b+1], h_ES_encoder_p, E_idx_p,
                )
                h_ESV_encoder_fw_p = mask_fw_p * h_ESV_encoder_p

                for l, dec_layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_p = cat_neighbors_nodes(
                        h_V_stack[l][b:b+1], h_ES_p, E_idx_p,
                    )
                    h_V_p = h_V_stack[l][b:b+1, p:p+1, :]
                    h_ESV_p = mask_bw_p * h_ESV_decoder_p + h_ESV_encoder_fw_p
                    h_V_stack[l + 1][b, p, :] = dec_layer(
                        h_V_p, h_ESV_p, mask_V=mask[b:b+1, p:p+1],
                    ).squeeze(0).squeeze(0)

                h_V_final = h_V_stack[-1][b, p, :]
                token_logits = self.W_out(h_V_final) / temperature
                token_probs = F.softmax(token_logits, dim=-1)
                S_p = torch.multinomial(token_probs.unsqueeze(0), 1).squeeze(-1).squeeze(0)

                S[b, p] = S_p
                h_S[b, p, :] = self.W_s(S_p)

            decoded_mask.scatter_(1, pos.unsqueeze(-1), 1.0)

        return S, ordering
