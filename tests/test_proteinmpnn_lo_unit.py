from __future__ import annotations

from typing import Any

import numpy as np
import torch

from protein_mpnn_lo_utils import ProteinMPNN_LO


def _dummy_batch(b: int = 2, l: int = 32) -> dict[str, Any]:
    """Create a small synthetic batch for unit testing."""
    x = torch.randn(b, l, 4, 3)
    s = torch.randint(0, 21, (b, l))
    mask = torch.ones(b, l)
    chain_m = torch.ones(b, l)
    # Make first few positions fixed (non-designable)
    chain_m[:, :5] = 0.0
    residue_idx = torch.arange(l).unsqueeze(0).repeat(b, 1)
    chain_encoding_all = torch.ones(b, l)
    return {
        "X": x,
        "S": s,
        "mask": mask,
        "chain_M": chain_m,
        "residue_idx": residue_idx,
        "chain_encoding_all": chain_encoding_all,
    }


def test_forward_shapes_and_finiteness() -> None:
    """forward() should return finite log_probs with expected shape."""
    batch = _dummy_batch()
    model = ProteinMPNN_LO(num_samples=2)
    model.eval()
    with torch.no_grad():
        log_probs = model.forward(
            batch["X"],
            batch["S"],
            batch["mask"],
            batch["chain_M"],
            batch["residue_idx"],
            batch["chain_encoding_all"],
            randn=torch.randn_like(batch["chain_M"]),
        )
    assert log_probs.shape == (batch["S"].shape[0], batch["S"].shape[1], 21)
    assert torch.isfinite(log_probs).all()


def test_compute_elbo_is_finite_and_backward_works() -> None:
    """compute_elbo() should produce finite loss and backprop gradients."""
    batch = _dummy_batch()
    model = ProteinMPNN_LO(num_samples=2)
    model.train()

    loss, info = model.compute_elbo(
        batch["X"],
        batch["S"],
        batch["mask"],
        batch["chain_M"],
        batch["residue_idx"],
        batch["chain_encoding_all"],
    )
    assert torch.isfinite(loss)
    assert "elbo" in info
    loss.backward()

    # Ensure some gradients exist.
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


def test_sample_preserves_fixed_positions_and_returns_order() -> None:
    """sample() should keep fixed positions equal to S_true."""
    batch = _dummy_batch(b=2, l=24)
    model = ProteinMPNN_LO(num_samples=2)
    model.eval()

    with torch.no_grad():
        out = model.sample(
            batch["X"],
            randn=torch.randn_like(batch["chain_M"]),
            S_true=batch["S"],
            chain_mask=batch["chain_M"],
            chain_encoding_all=batch["chain_encoding_all"],
            residue_idx=batch["residue_idx"],
            mask=batch["mask"],
            temperature=0.2,
            omit_AAs_np=np.zeros(21, dtype=np.float32),
            bias_AAs_np=np.zeros(21, dtype=np.float32),
            chain_M_pos=torch.ones_like(batch["chain_M"]),
            omit_AA_mask=None,
            pssm_coef=None,
            pssm_bias=None,
            pssm_multi=0.0,
            pssm_log_odds_flag=False,
            pssm_log_odds_mask=None,
            pssm_bias_flag=False,
            bias_by_res=torch.zeros(batch["S"].shape[0], batch["S"].shape[1], 21),
            order_temperature=1.0,
        )

    s_sample = out["S"]
    order = out["decoding_order"]
    assert s_sample.shape == batch["S"].shape
    assert order.shape == (batch["S"].shape[0], batch["S"].shape[1])

    # Fixed prefix should be identical
    assert torch.equal(s_sample[:, :5], batch["S"][:, :5])


def test_compute_loglik_is_q_is_finite() -> None:
    """IS log-likelihood estimate should be finite."""
    batch = _dummy_batch(b=2, l=20)
    model = ProteinMPNN_LO(num_samples=2)
    model.eval()
    with torch.no_grad():
        loglik = model.compute_loglik_is_q(
            batch["X"],
            batch["S"],
            batch["mask"],
            batch["chain_M"],
            batch["residue_idx"],
            batch["chain_encoding_all"],
            num_samples_eval=4,
        )
    assert loglik.shape == (batch["S"].shape[0],)
    assert torch.isfinite(loglik).all()


def _dummy_batch_with_padding(b: int = 2, l: int = 32) -> dict[str, Any]:
    """Batch where sequences have different lengths (mask includes padding).

    Batch element 0: length 32 (full), 5 fixed positions
    Batch element 1: length 20 (12 padding positions), 5 fixed positions

    This means for element 1:
      num_fixed = 5, but num_non_design = N - L_design = 32 - 15 = 17
    The gap (17 - 5 = 12) is the padding count.
    """
    x = torch.randn(b, l, 4, 3)
    s = torch.randint(0, 21, (b, l))
    mask = torch.ones(b, l)
    mask[1, 20:] = 0.0

    chain_m = torch.ones(b, l)
    chain_m[:, :5] = 0.0
    chain_m[1, 20:] = 0.0

    residue_idx = torch.arange(l).unsqueeze(0).repeat(b, 1)
    chain_encoding_all = torch.ones(b, l)
    return {
        "X": x,
        "S": s,
        "mask": mask,
        "chain_M": chain_m,
        "residue_idx": residue_idx,
        "chain_encoding_all": chain_encoding_all,
    }


def test_compute_elbo_with_padding_is_finite() -> None:
    """compute_elbo() should work correctly when batch has variable-length
    sequences with padding (mask=0 for padded positions)."""
    batch = _dummy_batch_with_padding()
    model = ProteinMPNN_LO(num_samples=2)
    model.train()

    loss, info = model.compute_elbo(
        batch["X"],
        batch["S"],
        batch["mask"],
        batch["chain_M"],
        batch["residue_idx"],
        batch["chain_encoding_all"],
    )
    assert torch.isfinite(loss), f"ELBO loss is not finite: {loss}"
    assert "elbo" in info
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


def test_compute_loglik_is_q_with_padding_is_finite() -> None:
    """IS log-likelihood should not crash with padded batches.

    This test specifically exercises the num_non_design offset fix:
    without the fix, step = num_fixed + d overflows N for the shorter
    sequence, causing an index-out-of-bounds error in torch.gather.
    """
    batch = _dummy_batch_with_padding()
    model = ProteinMPNN_LO(num_samples=2)
    model.eval()
    with torch.no_grad():
        loglik = model.compute_loglik_is_q(
            batch["X"],
            batch["S"],
            batch["mask"],
            batch["chain_M"],
            batch["residue_idx"],
            batch["chain_encoding_all"],
            num_samples_eval=4,
        )
    assert loglik.shape == (batch["S"].shape[0],)
    assert torch.isfinite(loglik).all(), f"IS loglik contains non-finite: {loglik}"

