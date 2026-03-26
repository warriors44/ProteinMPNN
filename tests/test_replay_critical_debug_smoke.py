"""Smoke test: replay_critical_debug runs on a minimal synthetic snapshot."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


def test_replay_critical_debug_exits_zero(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO))
    from protein_mpnn_lo_utils import ProteinMPNN_LO

    device = torch.device("cpu")
    model = ProteinMPNN_LO(
        num_samples=2,
        node_features=32,
        edge_features=32,
        hidden_dim=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=8,
        dropout=0.0,
        augment_eps=0.05,
        separate_q_decoder=False,
        ca_only=False,
    ).to(device)

    B, L = 2, 12
    batch = {
        "X": torch.randn(B, L, 4, 3),
        "S": torch.randint(0, 21, (B, L)),
        "mask": torch.ones(B, L),
        "chain_M": torch.ones(B, L),
        "residue_idx": torch.arange(L).unsqueeze(0).expand(B, -1),
        "chain_encoding_all": torch.ones(B, L),
    }
    stem = "first_critical_event_epoch1_step1"
    torch.save(batch, tmp_path / f"{stem}_batch.pt")

    ckpt = {
        "epoch": 1,
        "step": 1,
        "seed": 123,
        "model_state_dict": model.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_random_state": random.getstate(),
        "lambda_entropy": 0.0,
    }
    torch.save(ckpt, tmp_path / f"{stem}.pt")

    meta = {
        "seed": 123,
        "args": {
            "hidden_dim": 32,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "num_neighbors": 8,
            "dropout": 0.0,
            "backbone_noise": 0.05,
            "num_lo_samples": 2,
            "separate_q_decoder": 0,
            "ca_only": 0,
            "lambda_entropy": 0.0,
        },
    }
    with open(tmp_path / f"{stem}_meta.json", "w") as f:
        json.dump(meta, f)

    script = REPO / "training_lo" / "replay_critical_debug.py"
    r = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(tmp_path / f"{stem}.pt"),
            "--batch",
            str(tmp_path / f"{stem}_batch.pt"),
            "--meta",
            str(tmp_path / f"{stem}_meta.json"),
            "--device",
            "cpu",
            "--no_hooks",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.fail(f"replay_critical_debug failed:\n{r.stderr}\n{r.stdout}")
    assert r.returncode == 0
