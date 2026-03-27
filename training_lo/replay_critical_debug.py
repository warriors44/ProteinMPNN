#!/usr/bin/env python3
"""Replay a saved critical-event snapshot and trace where non-finite values appear.

This script loads outputs from ``training_lo_debug._save_first_critical_event_snapshot``:

  - ``model_weights/first_critical_event_epoch{E}_step{S}.pt`` — weights, optimizer,
    RNG state, and training hyperparameters.
  - ``first_critical_event_epoch{E}_step{S}_batch.pt`` — featurized tensors
    ``X, S, mask, chain_M, residue_idx, chain_encoding_all``.
  - ``first_critical_event_epoch{E}_step{S}_meta.json`` — ``seed`` and full ``args``.

**RNG:** Checkpoints written by current ``training_lo_debug`` include
``*_before_compute_elbo`` RNG keys (captured immediately before ``compute_elbo``) plus
legacy ``torch_rng_state`` / ``numpy_rng_state`` / ... taken after ``compute_elbo`` and
before ``backward``. This script restores ``*_before_compute_elbo`` first when present,
so ``compute_elbo`` internal randomness (``i_design``, Gumbel order, etc.) can be replayed
bit-for-bit together with ``torch`` / ``numpy`` / ``python`` / CUDA RNG. Older checkpoints
without the ``*_before_compute_elbo`` keys only restore the post-``compute_elbo`` state.

**Modes**

- **Stages** (``--run_stages``): high-level checks after ``_encode``, ``forward_q``,
  and one ``forward_p`` pass (deterministic sub-seed for permutation sampling).
- **Hooks** (``--run_hooks``): register forward hooks on every ``nn.Module`` and report
  the first submodule whose output contains non-finite values (optionally ``--verbose``
  to print every module).
- **ELBO** (``--run_compute_elbo``): full ``compute_elbo`` with ``return_debug=True``
  and print diagnostic scalars (same path as training).

Example::

    python training_lo/replay_critical_debug.py \\
        --checkpoint exp/model_weights/first_critical_event_epoch65_step96921.pt \\
        --batch exp/first_critical_event_epoch65_step96921_batch.pt

If ``--meta`` is omitted, ``../{stem}_meta.json`` next to ``model_weights/`` is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast


def _setup_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    training_dir = repo_root / "training"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return repo_root


_ = _setup_paths()

from protein_mpnn_lo_utils import ProteinMPNN_LO  # noqa: E402


def infer_sibling_from_checkpoint(checkpoint_path: str, suffix: str) -> Path:
    """Map ``.../model_weights/{stem}.pt`` to ``.../{stem}_{suffix}``."""
    p = Path(checkpoint_path).resolve()
    stem = p.stem
    return p.parent.parent / f"{stem}_{suffix}"


def load_meta(meta_path: Path) -> Dict[str, Any]:
    with open(meta_path, "r") as f:
        return json.load(f)


def namespace_from_args_dict(raw: Dict[str, Any]) -> SimpleNamespace:
    """Build a namespace usable like training ``args`` (bool/int coercion)."""
    d = dict(raw)
    if "separate_q_decoder" in d:
        d["separate_q_decoder"] = int(d["separate_q_decoder"])
    if "ca_only" in d:
        d["ca_only"] = int(d["ca_only"])
    return SimpleNamespace(**d)


def build_model(args: SimpleNamespace, device: torch.device) -> ProteinMPNN_LO:
    """Construct ``ProteinMPNN_LO`` the same way as ``training_lo_debug.main``."""
    return ProteinMPNN_LO(
        ca_only=bool(args.ca_only),
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
        num_samples=args.num_lo_samples,
        separate_q_decoder=bool(args.separate_q_decoder),
    ).to(device)


def load_batch(batch_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        raw = torch.load(batch_path, map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(batch_path, map_location=device)
    return {k: v.to(device) for k, v in raw.items()}


def _torch_cpu_rng_state(state: Any) -> torch.Tensor:
    """Return CPU tensor RNG state for ``torch.set_rng_state``.

    Checkpoints loaded with ``map_location=cuda`` move all tensors to GPU, but
    ``set_rng_state`` only accepts a CPU ``ByteTensor``.
    """
    if not torch.is_tensor(state):
        raise TypeError(f"Expected Tensor for torch RNG state, got {type(state)}")
    return state.detach().cpu()


def _cuda_rng_byte_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """CUDA ``set_rng_state`` requires a ``ByteTensor`` on the target CUDA device.

    ``torch.load(..., map_location=...)`` can leave RNG state tensors on the wrong
    device or dtype; normalize before ``torch.cuda.set_rng_state_all``.
    """
    out = t.detach()
    if out.dtype != torch.uint8:
        out = out.to(dtype=torch.uint8)
    return out.to(device=device)


def _cuda_rng_states_on_device(states: Any, device: torch.device) -> Any:
    """Place CUDA RNG state tensors on ``device`` as ``uint8`` (ByteTensor)."""
    if device.type != "cuda":
        return states
    if isinstance(states, (list, tuple)):
        return [
            _cuda_rng_byte_tensor(s, device) if torch.is_tensor(s) else s
            for s in states
        ]
    if torch.is_tensor(states):
        return _cuda_rng_byte_tensor(states, device)
    return states


def restore_rng_from_checkpoint(ckpt: Dict[str, Any], *, device: torch.device) -> None:
    """Restore Python / NumPy / Torch (and CUDA) RNG state if present.

    Prefer ``*_before_compute_elbo`` keys (replay ``compute_elbo`` draws) over legacy
    post-``compute_elbo`` keys.

    Args:
        ckpt: Loaded checkpoint dict (may have tensors on GPU from ``map_location``).
        device: Replay device; CUDA RNG is restored only when ``device.type == "cuda"``.
    """
    import random

    if "torch_rng_state_before_compute_elbo" in ckpt:
        torch.set_rng_state(_torch_cpu_rng_state(ckpt["torch_rng_state_before_compute_elbo"]))
        if "numpy_rng_state_before_compute_elbo" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state_before_compute_elbo"])
        if "python_random_state_before_compute_elbo" in ckpt:
            random.setstate(ckpt["python_random_state_before_compute_elbo"])
        if (
            device.type == "cuda"
            and torch.cuda.is_available()
            and "cuda_rng_state_all_before_compute_elbo" in ckpt
        ):
            st = _cuda_rng_states_on_device(
                ckpt["cuda_rng_state_all_before_compute_elbo"],
                device,
            )
            torch.cuda.set_rng_state_all(st)
        return

    if "seed" in ckpt:
        s = int(ckpt["seed"])
        torch.manual_seed(s)
        np.random.seed(s)
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(_torch_cpu_rng_state(ckpt["torch_rng_state"]))
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
    if "python_random_state" in ckpt:
        random.setstate(ckpt["python_random_state"])
    if device.type == "cuda" and torch.cuda.is_available() and "cuda_rng_state_all" in ckpt:
        st = _cuda_rng_states_on_device(ckpt["cuda_rng_state_all"], device)
        torch.cuda.set_rng_state_all(st)


def iter_tensors(obj: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from iter_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from iter_tensors(x)


def check_finite_tensor(tag: str, t: torch.Tensor) -> bool:
    if t.numel() == 0:
        return True
    ok = bool(torch.isfinite(t).all().item())
    if not ok:
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        print(f"[NONFINITE] {tag}  shape={tuple(t.shape)}  nan={n_nan}  inf={n_inf}")
    else:
        print(f"[ok] {tag}  shape={tuple(t.shape)}  dtype={t.dtype}")
    return ok


def run_stages(
    model: ProteinMPNN_LO,
    X: torch.Tensor,
    S: torch.Tensor,
    mask: torch.Tensor,
    chain_M: torch.Tensor,
    residue_idx: torch.Tensor,
    chain_encoding_all: torch.Tensor,
    *,
    stage_seed: int,
) -> None:
    """Named forward stages with finite checks (one deterministic forward_p sample)."""
    device = X.device
    design_mask = chain_M * mask
    print("--- stages: _encode ---")
    h_V_enc, h_E, E_idx = model._encode(X, mask, residue_idx, chain_encoding_all)
    check_finite_tensor("encode.h_V_enc", h_V_enc)
    check_finite_tensor("encode.h_E", h_E)

    print("--- stages: forward_q ---")
    q_logits = model.forward_q(h_V_enc, h_E, E_idx, S, mask, design_mask)
    check_finite_tensor("forward_q.q_logits", q_logits)

    B, N = S.shape
    L_design = design_mask.sum(dim=-1).clamp(min=1.0)
    num_non_design = N - design_mask.sum(dim=-1).long()

    torch.manual_seed(stage_seed)
    g = torch.Generator(device=device)
    g.manual_seed(stage_seed)
    i_design = (torch.rand(B, device=device, generator=g) * L_design).long() + 1
    i_full = i_design + num_non_design

    print("--- stages: permutation + forward_p (one sample) ---")
    full_perm = model._build_fixed_first_perm(
        design_mask, mask, q_logits.detach(),
    )
    check_finite_tensor("perm.full_perm", full_perm.float())

    ar_mask = model._build_partial_ar_mask(E_idx, full_perm, i_full)
    log_probs_k, p_order_logits_k = model.forward_p(
        h_V_enc, h_E, E_idx, S, mask, design_mask, ar_mask=ar_mask,
    )
    check_finite_tensor("forward_p.log_probs", log_probs_k)
    check_finite_tensor("forward_p.p_order_logits", p_order_logits_k)


def _make_module_hook(
    verbose: bool,
    first_hit: List[Optional[str]],
) -> Callable[..., None]:
    def hook(module: nn.Module, inp: Any, out: Any) -> None:
        name = getattr(module, "_hook_debug_name", module.__class__.__name__)
        if first_hit[0] is not None:
            return
        for ti, t in enumerate(iter_tensors(out)):
            if not torch.is_tensor(t) or t.numel() == 0:
                continue
            if not torch.isfinite(t).all():
                first_hit[0] = f"{name} (tensor index {ti} in output)"
                if not verbose:
                    return
        if verbose:
            for ti, t in enumerate(iter_tensors(out)):
                if torch.is_tensor(t) and t.numel() > 0:
                    finite = bool(torch.isfinite(t).all().item())
                    print(f"  [hook] {name}  out[{ti}]  finite={finite}  shape={tuple(t.shape)}")

    return hook


def register_nan_hooks(
    model: nn.Module,
    *,
    verbose: bool,
) -> Tuple[List[Any], List[Optional[str]]]:
    """Register forward hooks; first non-finite output sets ``first_hit[0]``."""
    handles: List[Any] = []
    first_hit: List[Optional[str]] = [None]
    for name, m in model.named_modules():
        display = name if name else "(root)"
        m._hook_debug_name = display  # type: ignore[attr-defined]
        h = m.register_forward_hook(_make_module_hook(verbose, first_hit))
        handles.append(h)
    return handles, first_hit


def remove_hooks(handles: List[Any]) -> None:
    for h in handles:
        h.remove()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay critical-event snapshot and trace non-finite forwards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to first_critical_event_epoch*_step*.pt",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default="",
        help="Path to *_batch.pt (default: infer from checkpoint stem)",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="",
        help="Path to *_meta.json (default: infer next to checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Wrap compute_elbo / hook run in autocast (match AMP training).",
    )
    parser.add_argument(
        "--manual_seed",
        type=int,
        default=-1,
        help="If >=0, override meta/checkpoint seed before running.",
    )
    parser.add_argument(
        "--stage_seed",
        type=int,
        default=42,
        help="Seed for i_design / permutation in --run_stages only.",
    )
    parser.add_argument(
        "--no_stages",
        action="store_true",
        help="Skip named stage tracing (_encode, forward_q, forward_p).",
    )
    parser.add_argument(
        "--no_hooks",
        action="store_true",
        help="Skip submodule forward hooks during compute_elbo.",
    )
    parser.add_argument(
        "--verbose_hooks",
        action="store_true",
        help="Print per-module finite status (noisy).",
    )
    parser.add_argument(
        "--no_compute_elbo",
        action="store_true",
        help="Skip full compute_elbo (only useful with stages).",
    )
    args = parser.parse_args()

    run_stages_flag = not args.no_stages
    run_hooks_flag = not args.no_hooks
    run_elbo_flag = not args.no_compute_elbo

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    batch_path = Path(args.batch).resolve() if args.batch else infer_sibling_from_checkpoint(
        str(ckpt_path), "batch.pt",
    )
    meta_path = Path(args.meta).resolve() if args.meta else infer_sibling_from_checkpoint(
        str(ckpt_path), "meta.json",
    )
    if not batch_path.is_file():
        raise SystemExit(f"Batch file not found: {batch_path}")
    if not meta_path.is_file():
        raise SystemExit(f"Meta file not found: {meta_path}")

    meta = load_meta(meta_path)
    raw_args = meta.get("args", {})
    ns = namespace_from_args_dict(raw_args)

    device = torch.device(args.device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model(ns, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    batch = load_batch(batch_path, device)
    X = batch["X"]
    S = batch["S"]
    mask = batch["mask"]
    chain_M = batch["chain_M"]
    residue_idx = batch["residue_idx"]
    chain_encoding_all = batch["chain_encoding_all"]

    seed_use = int(args.manual_seed) if args.manual_seed >= 0 else int(meta.get("seed", ckpt.get("seed", 0)))
    torch.manual_seed(seed_use)
    np.random.seed(seed_use)
    restore_rng_from_checkpoint(ckpt, device=device)

    lam = float(getattr(ns, "lambda_entropy", ckpt.get("lambda_entropy", 0.0)))

    model.train()

    if run_stages_flag:
        print("======== STAGE TRACE ========")
        run_stages(
            model, X, S, mask, chain_M, residue_idx, chain_encoding_all,
            stage_seed=args.stage_seed,
        )

    handles: List[Any] = []
    first_hit: List[Optional[str]] = [None]

    def run_elbo_inner() -> Tuple[torch.Tensor, Dict[str, Any]]:
        return model.compute_elbo(
            X, S, mask, chain_M, residue_idx, chain_encoding_all,
            return_debug=True,
            lambda_entropy=lam,
        )

    if run_elbo_flag:
        print("======== compute_elbo ========")
        if run_hooks_flag:
            handles, first_hit = register_nan_hooks(model, verbose=args.verbose_hooks)
        try:
            if args.mixed_precision and device.type == "cuda":
                with autocast("cuda"):
                    loss, info = run_elbo_inner()
            else:
                loss, info = run_elbo_inner()
        finally:
            if handles:
                remove_hooks(handles)

        print(f"loss={loss.item()}  finite={bool(torch.isfinite(loss).item())}")
        for k in sorted(info.keys()):
            if k.startswith("dbg_") or k in (
                "entropy_q", "entropy_q_weighted", "elbo_no_penalty", "elbo",
            ):
                v = info[k]
                if torch.is_tensor(v):
                    print(f"  {k}={v.detach().cpu().item()}")
        if run_hooks_flag and first_hit[0] is not None:
            print(f"First submodule with non-finite output: {first_hit[0]}")
        elif run_hooks_flag:
            print("Hooks: no non-finite tensor observed in module outputs.")

    print("Done.")


if __name__ == "__main__":
    main()
