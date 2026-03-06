from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAINING_DIR = _REPO_ROOT / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from model_utils import get_std_opt  # type: ignore[import-not-found]


def test_noamopt_with_gradscaler_updates_params_and_decreases_loss() -> None:
    """Ensure GradScaler + NoamOpt actually updates parameters and reduces loss.

    This test mirrors the mixed-precision training branch in
    `training_lo/training_lo.py` but uses a tiny synthetic regression problem.
    With the previous bug (`scaler.step(optimizer.optimizer)`), the internal
    Adam optimizer inside NoamOpt kept a learning rate of zero, so parameters
    never updated and the loss did not decrease. Using `scaler.step(optimizer)`
    should call NoamOpt.step(), update the learning rate schedule, and perform
    real optimization steps.
    """
    device = torch.device("cpu")

    # Tiny deterministic regression task: y = 2 * x.
    x = torch.randn(16, 4, device=device)
    y = 2.0 * x.sum(dim=-1, keepdim=True)

    model = torch.nn.Linear(4, 1).to(device)
    initial_state = copy.deepcopy(model.state_dict())

    optimizer = get_std_opt(model.parameters(), d_model=128, step=0)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses: list[float] = []

    for _ in range(5):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().cpu().item()))

    # Learning rate inside the wrapped Adam optimizer should have been updated
    # away from zero by NoamOpt.step().
    lr_values = [g["lr"] for g in optimizer.optimizer.param_groups]
    assert all(lr > 0.0 for lr in lr_values)

    # Parameters should have changed compared to the initial state.
    updated_state = model.state_dict()
    param_changed = any(
        not torch.allclose(updated_state[k], v) for k, v in initial_state.items()
    )
    assert param_changed

    # Loss should decrease over the course of a few optimization steps.
    assert losses[-1] < losses[0]

