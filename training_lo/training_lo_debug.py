from __future__ import annotations

import argparse
import os.path


def main(args: argparse.Namespace) -> None:
    """Debug variant of LO training with extra logging.

    This script is a near-copy of `training_lo.py`, but additionally writes:
    - step-level grad norms to the main epoch log (same columns)
    - an optional debug log file that records the same columns as `log.txt`
      every N steps (controlled by --debug_log_interval).
    """

    import copy
    import json
    import os
    import queue
    import random
    import sys
    import time
    from concurrent.futures import ProcessPoolExecutor
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any, Dict, Iterable, List, Optional, Tuple

    import numpy as np
    import torch
    from torch.cuda.amp import GradScaler, autocast

    # ------------------------------------------------------------------
    # Import training utilities from the existing `training/` directory.
    # ------------------------------------------------------------------
    repo_root = Path(__file__).resolve().parents[1]
    training_dir = repo_root / "training"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))

    from utils import (  # type: ignore[import-not-found]
        worker_init_fn,
        get_pdbs,
        loader_pdb,
        build_training_clusters,
        PDB_dataset,
        StructureDataset,
        StructureLoader,
    )
    from model_utils import (  # type: ignore[import-not-found]
        featurize,
        loss_nll,
        get_std_opt,
    )

    from protein_mpnn_lo_utils import ProteinMPNN_LO

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @dataclass(frozen=True)
    class PartialLoadReport:
        """Report for partial checkpoint loading."""

        loaded: List[str]
        skipped_shape_mismatch: List[str]
        skipped_missing_in_ckpt: List[str]

    def _partial_load_state_dict(
        model: torch.nn.Module,
        ckpt_state: Dict[str, torch.Tensor],
        *,
        exclude_prefixes: Tuple[str, ...] = (
            "W_order_p",
            "W_order_q",
            "q_decoder_layers",
            "W_order_q_sep",
        ),
    ) -> PartialLoadReport:
        """Load parameters by key+shape match, skipping excluded prefixes."""
        model_state = model.state_dict()
        loaded: List[str] = []
        skipped_shape_mismatch: List[str] = []
        skipped_missing_in_ckpt: List[str] = []

        for k, v in model_state.items():
            if k.startswith(exclude_prefixes):
                continue
            if k not in ckpt_state:
                skipped_missing_in_ckpt.append(k)
                continue
            src = ckpt_state[k]
            if tuple(src.shape) != tuple(v.shape):
                skipped_shape_mismatch.append(k)
                continue
            v.copy_(src)
            loaded.append(k)

        model.load_state_dict(model_state, strict=True)
        return PartialLoadReport(
            loaded=loaded,
            skipped_shape_mismatch=skipped_shape_mismatch,
            skipped_missing_in_ckpt=skipped_missing_in_ckpt,
        )

    def _compute_nll_metrics(
        model: ProteinMPNN_LO,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """Compute NLL/perplexity/accuracy as diagnostics."""
        device = X.device
        randn = torch.randn(chain_M.shape, device=device)
        log_probs = model.forward(
            X,
            S,
            mask,
            chain_M,
            residue_idx,
            chain_encoding_all,
            randn=randn,
        )
        mask_for_loss = mask * chain_M
        loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
        nll = float(
            (loss * mask_for_loss).sum().detach().cpu().item()
            / (mask_for_loss.sum().detach().cpu().item() + 1e-8)
        )
        perplexity = float(np.exp(nll))
        acc = float(
            (true_false * mask_for_loss).sum().detach().cpu().item()
            / (mask_for_loss.sum().detach().cpu().item() + 1e-8)
        )
        return nll, perplexity, acc

    def _compute_isq_nll(
        model: ProteinMPNN_LO,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        *,
        num_samples_eval: int,
    ) -> Tuple[float, float]:
        """Compute IS(q) NLL/perplexity for evaluation (teacher-forced)."""
        loglik_per_res = model.compute_loglik_is_q(
            X,
            S,
            mask,
            chain_M,
            residue_idx,
            chain_encoding_all,
            num_samples_eval=num_samples_eval,
        )  # [B]
        L_design = (mask * chain_M).sum(dim=-1).clamp(min=1.0)  # [B]
        nll = float(
            (
                -(loglik_per_res * L_design).sum()
                / (L_design.sum() + 1e-8)
            ).detach().cpu().item()
        )
        ppl = float(np.exp(nll))
        return nll, ppl

    # ------------------------------------------------------------------
    # RNG & device
    # ------------------------------------------------------------------
    if args.seed == 0:
        seed = int(np.random.randint(0, high=999, size=1, dtype=int)[0])
    else:
        seed = int(args.seed)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    scaler = GradScaler(enabled=bool(args.mixed_precision and torch.cuda.is_available()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Output folders
    # ------------------------------------------------------------------
    base_folder = time.strftime(args.path_for_outputs, time.localtime())
    if base_folder[-1] != "/":
        base_folder += "/"
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(base_folder + "model_weights", exist_ok=True)

    logfile = base_folder + "log.txt"
    if not args.previous_checkpoint:
        with open(logfile, "w") as f:
            f.write(
                "epoch\tstep\ttime_s\t"
                "train_elbo_loss\ttrain_nll\ttrain_ppl\ttrain_acc\t"
                "valid_elbo_loss\tvalid_nll_proxy\tvalid_ppl_proxy\tvalid_acc\t"
                "valid_nll_isq\tvalid_ppl_isq\t"
                "i_mean\tdelta_F_abs\tgrad_norm\n"
            )

    # Debug log: same columns as logfile, but written every debug_log_interval steps.
    debug_logfile = base_folder + "log_debug.txt"
    if args.debug_log_interval > 0 and (not args.previous_checkpoint):
        with open(debug_logfile, "w") as f:
            f.write(
                "epoch\tstep\ttime_s\t"
                "train_elbo_loss\ttrain_nll\ttrain_ppl\ttrain_acc\t"
                "valid_elbo_loss\tvalid_nll_proxy\tvalid_ppl_proxy\tvalid_acc\t"
                "valid_nll_isq\tvalid_ppl_isq\t"
                "i_mean\tdelta_F_abs\tgrad_norm\n"
            )

    # Nonfinite debug log: event/statistics log to diagnose NaN/inf root causes.
    nonfinite_logfile = base_folder + "log_nonfinite_debug.txt"
    nonfinite_header = (
        "epoch\tstep\ttime_s\t"
        "scaler_scale_before\tscaler_scale_after\tstep_skipped\t"
        "loss_isfinite\tgrad_nonfinite_count\t"
        "design_sum_min\tdesign_sum_max\tdesign_zero_count\t"
        "remaining_sum_min\tremaining_sum_max\tremaining_zero_count\t"
        "all_neg_inf_rows_count\t"
        "any_nonfinite_F\tany_nonfinite_log_q\tany_nonfinite_q_logits\tany_nonfinite_log_probs\tany_nonfinite_p_order_logits\n"
    )
    if not args.previous_checkpoint:
        with open(nonfinite_logfile, "w") as f:
            f.write(nonfinite_header)
    first_nonfinite_dumped = False

    # ------------------------------------------------------------------
    # Data pipeline (same as training/training.py)
    # ------------------------------------------------------------------
    data_path = args.path_for_training_data
    params: Dict[str, Any] = {
        "LIST": f"{data_path}/list.csv",
        "VAL": f"{data_path}/valid_clusters.txt",
        "TEST": f"{data_path}/test_clusters.txt",
        "DIR": f"{data_path}",
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut,
        "HOMO": 0.70,
    }

    load_param = {
        "batch_size": 1,
        "shuffle": True,
        "pin_memory": False,
        "num_workers": 4,
    }

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, _test = build_training_clusters(params, args.debug)
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(
        train_set, worker_init_fn=worker_init_fn, **load_param,
    )
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, worker_init_fn=worker_init_fn, **load_param,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ProteinMPNN_LO(
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

    # Resume from LO checkpoint (full state)
    total_step = 0
    start_epoch = 0
    if args.previous_checkpoint:
        checkpoint = torch.load(args.previous_checkpoint, map_location=device)
        total_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        # Initialize from a non-LO ProteinMPNN checkpoint (partial match load).
        if args.init_from_checkpoint:
            init_ckpt = torch.load(args.init_from_checkpoint, map_location=device)
            init_state = init_ckpt.get("model_state_dict", init_ckpt)
            report = _partial_load_state_dict(model, init_state)
            init_log = {
                "loaded": len(report.loaded),
                "skipped_shape_mismatch": len(report.skipped_shape_mismatch),
                "skipped_missing_in_ckpt": len(report.skipped_missing_in_ckpt),
            }
            with open(base_folder + "init_from_checkpoint_report.json", "w") as f:
                json.dump(init_log, f, indent=2)

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    if args.previous_checkpoint:
        optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # ------------------------------------------------------------------
    # Training loop with background data prefetching (same pattern)
    # ------------------------------------------------------------------
    with ProcessPoolExecutor(max_workers=12) as executor:
        q_train: "queue.Queue[Any]" = queue.Queue(maxsize=3)
        q_valid: "queue.Queue[Any]" = queue.Queue(maxsize=3)

        for _ in range(3):
            q_train.put_nowait(
                executor.submit(
                    get_pdbs,
                    train_loader,
                    1,
                    args.max_protein_length,
                    args.num_examples_per_epoch,
                )
            )
            q_valid.put_nowait(
                executor.submit(
                    get_pdbs,
                    valid_loader,
                    1,
                    args.max_protein_length,
                    args.num_examples_per_epoch,
                )
            )

        pdb_dict_train = q_train.get().result()
        pdb_dict_valid = q_valid.get().result()

        dataset_train = StructureDataset(
            pdb_dict_train, truncate=None, max_length=args.max_protein_length,
        )
        dataset_valid = StructureDataset(
            pdb_dict_valid, truncate=None, max_length=args.max_protein_length,
        )

        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)

        reload_c = 0
        best_valid_ppl_proxy = float("inf")
        for e0 in range(args.num_epochs):
            t0 = time.time()
            epoch_idx = start_epoch + e0

            model.train()
            train_elbo_sum = 0.0
            train_elbo_w = 0.0
            train_nll_sum = 0.0
            train_acc_sum = 0.0
            train_w = 0.0
            i_mean_sum = 0.0
            delta_f_sum = 0.0
            info_w = 0.0
            grad_norm_sum = 0.0
            grad_norm_w = 0.0

            if epoch_idx % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q_train.get().result()
                    dataset_train = StructureDataset(
                        pdb_dict_train,
                        truncate=None,
                        max_length=args.max_protein_length,
                    )
                    loader_train = StructureLoader(
                        dataset_train, batch_size=args.batch_size,
                    )

                    pdb_dict_valid = q_valid.get().result()
                    dataset_valid = StructureDataset(
                        pdb_dict_valid,
                        truncate=None,
                        max_length=args.max_protein_length,
                    )
                    loader_valid = StructureLoader(
                        dataset_valid, batch_size=args.batch_size,
                    )

                    q_train.put_nowait(
                        executor.submit(
                            get_pdbs,
                            train_loader,
                            1,
                            args.max_protein_length,
                            args.num_examples_per_epoch,
                        )
                    )
                    q_valid.put_nowait(
                        executor.submit(
                            get_pdbs,
                            valid_loader,
                            1,
                            args.max_protein_length,
                            args.num_examples_per_epoch,
                        )
                    )
                reload_c += 1

            for _batch_idx, batch in enumerate(loader_train):
                X, S, mask, lengths, chain_M, residue_idx, _mask_self, chain_encoding_all = featurize(batch, device)

                optimizer.zero_grad()
                if scaler.is_enabled():
                    scaler_scale_before = float(scaler.get_scale())
                    with autocast():
                        loss_elbo, info = model.compute_elbo(
                            X, S, mask, chain_M, residue_idx, chain_encoding_all,
                            return_debug=True,
                        )
                    scaler.scale(loss_elbo).backward()
                    scaler.unscale_(optimizer)
                    # Detect nonfinite gradients after unscale (indicates backward overflow or NaN propagation).
                    grad_nonfinite_count = 0
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        if not torch.isfinite(p.grad).all():
                            grad_nonfinite_count += 1
                    grad_norm_value_cur: float | None = None
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.gradient_norm,
                        )
                        grad_norm_value_cur = float(total_norm.detach().cpu().item())
                    scaler.step(optimizer)
                    scaler.update()
                    scaler_scale_after = float(scaler.get_scale())
                    step_skipped = int(scaler_scale_after < scaler_scale_before)
                else:
                    scaler_scale_before = float("nan")
                    loss_elbo, info = model.compute_elbo(
                        X, S, mask, chain_M, residue_idx, chain_encoding_all,
                        return_debug=True,
                    )
                    loss_elbo.backward()
                    grad_nonfinite_count = 0
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        if not torch.isfinite(p.grad).all():
                            grad_nonfinite_count += 1
                    grad_norm_value_cur = None
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.gradient_norm,
                        )
                        grad_norm_value_cur = float(total_norm.detach().cpu().item())
                    optimizer.step()
                    scaler_scale_after = float("nan")
                    step_skipped = 0

                # Mask non-finite steps out of all epoch-level statistics updates
                # (skip_both: don't add to numerator nor denominator).
                # Pull debug stats from info (present when return_debug=True).
                # We compute them once and reuse the same scalars for:
                #   1) valid_step (aggregation mask)
                #   2) nonfinite_logfile / FIRST_EVENT logging.
                loss_isfinite = float(
                    info.get("dbg_loss_isfinite", torch.tensor(0.0))
                    .detach().cpu().item(),
                )
                design_sum_min = float(
                    info.get("dbg_design_sum_min", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )
                design_sum_max = float(
                    info.get("dbg_design_sum_max", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )
                design_zero_count = float(
                    info.get("dbg_design_zero_count", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )
                remaining_sum_min = float(
                    info.get("dbg_remaining_sum_min", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )
                remaining_sum_max = float(
                    info.get("dbg_remaining_sum_max", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )
                remaining_zero_count = float(
                    info.get("dbg_remaining_zero_count", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )
                all_neg_inf_rows_count = float(
                    info.get("dbg_all_neg_inf_rows_count", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )
                any_nonfinite_F = float(
                    info.get("dbg_any_nonfinite_F", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )
                any_nonfinite_log_q = float(
                    info.get("dbg_any_nonfinite_log_q", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )
                any_nonfinite_q_logits = float(
                    info.get("dbg_any_nonfinite_q_logits", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )
                any_nonfinite_log_probs = float(
                    info.get("dbg_any_nonfinite_log_probs", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )
                any_nonfinite_p_order_logits = float(
                    info.get("dbg_any_nonfinite_p_order_logits", torch.tensor(float("nan")))
                    .detach().cpu().item(),
                )

                nonfinite_trigger_mask = (
                    (loss_isfinite == 0.0)
                    or (grad_nonfinite_count > 0)
                    or (design_zero_count > 0)
                    or (remaining_zero_count > 0)
                    or (any_nonfinite_F > 0)
                    or (any_nonfinite_log_q > 0)
                    or (any_nonfinite_q_logits > 0)
                    or (any_nonfinite_log_probs > 0)
                    or (any_nonfinite_p_order_logits > 0)
                    or (step_skipped > 0)
                )
                valid_step = not nonfinite_trigger_mask

                # Diagnostic NLL metrics (no grad)
                with torch.no_grad():
                    nll, ppl, acc = _compute_nll_metrics(
                        model, X, S, mask, chain_M, residue_idx, chain_encoding_all,
                    )
                    weight = float(
                        (mask * chain_M).sum().detach().cpu().item(),
                    )
                    if valid_step:
                        train_elbo_sum += float(loss_elbo.detach().cpu().item()) * weight
                        train_elbo_w += weight
                        train_nll_sum += nll * weight
                        train_acc_sum += acc * weight
                        train_w += weight

                        if "i_mean" in info:
                            i_mean_sum += float(
                                info["i_mean"].detach().cpu().item(),
                            )
                            info_w += 1.0
                        if "delta_F_abs" in info:
                            delta_f_sum += float(
                                info["delta_F_abs"].detach().cpu().item(),
                            )

                        if grad_norm_value_cur is not None and np.isfinite(grad_norm_value_cur):
                            grad_norm_sum += grad_norm_value_cur
                            grad_norm_w += 1.0

                total_step += 1

                # Nonfinite debug statistics log (event-based but lightweight enough to write each step).
                dt_cur = float(time.time() - t0)

                with open(nonfinite_logfile, "a") as nf:
                    nf.write(
                        f"{epoch_idx + 1}\t{total_step}\t{dt_cur:.1f}\t"
                        f"{scaler_scale_before:.1f}\t{scaler_scale_after:.1f}\t{step_skipped}\t"
                        f"{loss_isfinite:.0f}\t{grad_nonfinite_count}\t"
                        f"{design_sum_min:.0f}\t{design_sum_max:.0f}\t{design_zero_count:.0f}\t"
                        f"{remaining_sum_min:.0f}\t{remaining_sum_max:.0f}\t{remaining_zero_count:.0f}\t"
                        f"{all_neg_inf_rows_count:.0f}\t"
                        f"{any_nonfinite_F:.0f}\t{any_nonfinite_log_q:.0f}\t{any_nonfinite_q_logits:.0f}\t{any_nonfinite_log_probs:.0f}\t{any_nonfinite_p_order_logits:.0f}\n"
                    )

                # One-time detailed dump on first sign of nonfinite behavior.
                if (not first_nonfinite_dumped) and nonfinite_trigger_mask:
                    first_nonfinite_dumped = True
                    with open(nonfinite_logfile, "a") as nf:
                        nf.write(
                            "FIRST_EVENT\t"
                            f"epoch={epoch_idx + 1}\tstep={total_step}\t"
                            f"loss_isfinite={loss_isfinite:.0f}\t"
                            f"grad_nonfinite_count={grad_nonfinite_count}\t"
                            f"design_zero_count={design_zero_count:.0f}\t"
                            f"remaining_zero_count={remaining_zero_count:.0f}\t"
                            f"all_neg_inf_rows_count={all_neg_inf_rows_count:.0f}\t"
                            f"any_nonfinite_q_logits={any_nonfinite_q_logits:.0f}\t"
                            f"step_skipped={step_skipped}\n"
                        )

                # Optional step-level debug log (training side only; val fields set to nan).
                if args.debug_log_interval > 0 and (total_step % args.debug_log_interval == 0):
                    train_elbo_cur = train_elbo_sum / max(train_elbo_w, 1e-8)
                    train_nll_cur = train_nll_sum / max(train_w, 1e-8)
                    train_ppl_cur = float(np.exp(train_nll_cur))
                    train_acc_cur = train_acc_sum / max(train_w, 1e-8)
                    i_mean_cur = i_mean_sum / max(info_w, 1e-8)
                    delta_f_cur = delta_f_sum / max(info_w, 1e-8)
                    grad_norm_avg_cur = grad_norm_sum / max(grad_norm_w, 1.0)

                    with open(debug_logfile, "a") as df:
                        df.write(
                            f"{epoch_idx + 1}\t{total_step}\t{dt_cur:.1f}\t"
                            f"{train_elbo_cur:.6f}\t{train_nll_cur:.6f}\t{train_ppl_cur:.3f}\t{train_acc_cur:.4f}\t"
                            f"nan\tnan\tnan\tnan\tnan\tnan\t"
                            f"{i_mean_cur:.3f}\t{delta_f_cur:.6f}\t{grad_norm_avg_cur:.6f}\n"
                        )

            # Validation: always proxy; full IS-q every interval (and epoch 1)
            epoch_num = epoch_idx + 1
            run_full_isq = (epoch_num % int(args.eval_full_interval) == 0)
            model.eval()
            valid_elbo_sum = 0.0
            valid_elbo_w = 0.0
            valid_nll_proxy_sum = 0.0
            valid_nll_isq_sum = 0.0
            valid_acc_sum = 0.0
            valid_w = 0.0
            valid_isq_w = 0.0

            with torch.no_grad():
                for _batch_idx, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, _mask_self, chain_encoding_all = featurize(batch, device)
                    loss_elbo, _info = model.compute_elbo(
                        X, S, mask, chain_M, residue_idx, chain_encoding_all,
                    )
                    # Always compute accuracy from a forward pass (diagnostic).
                    nll_diag, _ppl_diag, acc = _compute_nll_metrics(
                        model, X, S, mask, chain_M, residue_idx, chain_encoding_all,
                    )
                    proxy_loglik_per_res = model.compute_loglik_proxy_q_px(
                        X,
                        S,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_encoding_all,
                        num_samples_eval=int(args.proxy_num_samples),
                    )  # [B]
                    L_design = (mask * chain_M).sum(dim=-1).clamp(min=1.0)  # [B]
                    proxy_nll = float(
                        (
                            -(proxy_loglik_per_res * L_design).sum()
                            / (L_design.sum() + 1e-8)
                        ).detach().cpu().item()
                    )
                    if run_full_isq:
                        nll_isq, _ppl_isq = _compute_isq_nll(
                            model,
                            X,
                            S,
                            mask,
                            chain_M,
                            residue_idx,
                            chain_encoding_all,
                            num_samples_eval=int(args.eval_num_samples),
                        )
                    weight = float(
                        (mask * chain_M).sum().detach().cpu().item(),
                    )
                    valid_elbo_sum += float(loss_elbo.detach().cpu().item()) * weight
                    valid_elbo_w += weight
                    valid_nll_proxy_sum += proxy_nll * weight
                    valid_acc_sum += acc * weight
                    valid_w += weight
                    if run_full_isq:
                        valid_nll_isq_sum += nll_isq * weight
                        valid_isq_w += weight

            train_elbo = train_elbo_sum / max(train_elbo_w, 1e-8)
            train_nll = train_nll_sum / max(train_w, 1e-8)
            train_ppl = float(np.exp(train_nll))
            train_acc = train_acc_sum / max(train_w, 1e-8)

            valid_elbo = valid_elbo_sum / max(valid_elbo_w, 1e-8)
            valid_nll_proxy = valid_nll_proxy_sum / max(valid_w, 1e-8)
            valid_ppl_proxy = float(np.exp(valid_nll_proxy))
            if run_full_isq:
                valid_nll_isq = valid_nll_isq_sum / max(valid_isq_w, 1e-8)
                valid_ppl_isq = float(np.exp(valid_nll_isq))
            else:
                valid_nll_isq = float("nan")
                valid_ppl_isq = float("nan")
            valid_acc = valid_acc_sum / max(valid_w, 1e-8)

            grad_norm_avg = grad_norm_sum / max(grad_norm_w, 1.0)

            i_mean = i_mean_sum / max(info_w, 1e-8)
            delta_f = delta_f_sum / max(info_w, 1e-8)

            t1 = time.time()
            dt = float(t1 - t0)

            with open(logfile, "a") as f:
                f.write(
                    f"{epoch_idx + 1}\t{total_step}\t{dt:.1f}\t"
                    f"{train_elbo:.6f}\t{train_nll:.6f}\t{train_ppl:.3f}\t{train_acc:.4f}\t"
                    f"{valid_elbo:.6f}\t{valid_nll_proxy:.6f}\t{valid_ppl_proxy:.3f}\t{valid_acc:.4f}\t"
                    f"{valid_nll_isq:.6f}\t{valid_ppl_isq:.3f}\t"
                    f"{i_mean:.3f}\t{delta_f:.6f}\t{grad_norm_avg:.6f}\n"
                )
            print(
                f"epoch: {epoch_idx + 1}, step: {total_step}, time: {dt:.1f}s, "
                f"train_elbo: {train_elbo:.4f}, valid_elbo: {valid_elbo:.4f}, "
                f"train_ppl: {train_ppl:.3f}, valid_ppl_proxy: {valid_ppl_proxy:.3f}, "
                f"valid_ppl_isq: {valid_ppl_isq:.3f}, "
                f"grad_norm: {grad_norm_avg:.3f}, "
                f"train_acc: {train_acc:.3f}, valid_acc: {valid_acc:.3f}"
            )

            if valid_ppl_proxy < best_valid_ppl_proxy:
                best_valid_ppl_proxy = valid_ppl_proxy
                ckpt_best = base_folder + "model_weights/epoch_best_proxy.pt"
                torch.save(
                    {
                        "epoch": epoch_idx + 1,
                        "step": total_step,
                        "num_edges": args.num_neighbors,
                        "noise_level": args.backbone_noise,
                        "num_samples": args.num_lo_samples,
                        "separate_q_decoder": bool(args.separate_q_decoder),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.optimizer.state_dict(),
                    },
                    ckpt_best,
                )

            ckpt_last = base_folder + "model_weights/epoch_last.pt"
            torch.save(
                {
                    "epoch": epoch_idx + 1,
                    "step": total_step,
                    "num_edges": args.num_neighbors,
                    "noise_level": args.backbone_noise,
                    "num_samples": args.num_lo_samples,
                    "separate_q_decoder": bool(args.separate_q_decoder),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.optimizer.state_dict(),
                },
                ckpt_last,
            )

            if (epoch_idx + 1) % args.save_model_every_n_epochs == 0:
                ckpt_path = base_folder + f"model_weights/epoch{epoch_idx + 1}_step{total_step}.pt"
                torch.save(
                    {
                        "epoch": epoch_idx + 1,
                        "step": total_step,
                        "num_edges": args.num_neighbors,
                        "noise_level": args.backbone_noise,
                        "num_samples": args.num_lo_samples,
                        "separate_q_decoder": bool(args.separate_q_decoder),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.optimizer.state_dict(),
                    },
                    ckpt_path,
                )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    argparser.add_argument(
        "--path_for_training_data",
        type=str,
        default="my_path/pdb_2021aug02",
        help="Path for loading training data (same layout as training/).",
    )
    argparser.add_argument(
        "--path_for_outputs",
        type=str,
        default="./exp_lo",
        help="Path for logs and model weights.",
    )
    argparser.add_argument(
        "--previous_checkpoint",
        type=str,
        default="",
        help="Path to a previous LO checkpoint to resume from.",
    )
    argparser.add_argument(
        "--init_from_checkpoint",
        type=str,
        default="",
        help="Optional path to a non-LO ProteinMPNN checkpoint to initialize from.",
    )

    argparser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train for.")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="Save model weights every n epochs.")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="Reload training data every n epochs.")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="Number of training examples per epoch.")
    argparser.add_argument("--batch_size", type=int, default=10000, help="Number of tokens per batch.")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="Maximum protein length.")

    argparser.add_argument("--hidden_dim", type=int, default=128, help="Hidden model dimension.")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="Number of encoder layers.")
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="Number of decoder layers.")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="Number of neighbors for the sparse graph.")
    argparser.add_argument("--dropout", type=float, default=0.1, help="Dropout level.")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="Backbone noise (augment_eps).")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff.")
    argparser.add_argument("--debug", type=bool, default=False, help="Minimal data loading for debugging.")
    argparser.add_argument("--gradient_norm", type=float, default=1.0, help="Clip gradient norm, negative to disable.")
    argparser.add_argument("--mixed_precision", action="store_true", help="Train with mixed precision.")

    # LO-ARM specific
    argparser.add_argument("--num_lo_samples", type=int, default=2, help="Number of RLOO samples K (>=2).")
    argparser.add_argument("--separate_q_decoder", type=int, default=0, help="0/1: use a separate q decoder.")
    argparser.add_argument("--ca_only", type=int, default=0, help="0/1: CA-only features/model.")
    argparser.add_argument("--seed", type=int, default=0, help="If 0, a random seed is picked.")
    argparser.add_argument(
        "--eval_mode",
        type=str,
        default="nll",
        choices=["nll", "is_q", "mc_p"],
        help="Validation NLL evaluation mode. 'is_q' uses importance sampling with q; 'mc_p' is reserved for future p-order Monte Carlo.",
    )
    argparser.add_argument(
        "--eval_num_samples",
        type=int,
        default=8,
        help="Number of samples for eval_mode 'is_q' (and future 'mc_p').",
    )
    argparser.add_argument(
        "--proxy_num_samples",
        type=int,
        default=8,
        help="Number of q(z) samples for the fast proxy validation NLL/PPL.",
    )
    argparser.add_argument(
        "--eval_full_interval",
        type=int,
        default=100,
        help="Run full IS-q evaluation every N epochs (epoch 1 is always run).",
    )
    argparser.add_argument(
        "--debug_log_interval",
        type=int,
        default=0,
        help="If >0, write step-level debug log every N steps to log_debug.txt.",
    )

    parsed = argparser.parse_args()

    parsed.separate_q_decoder = int(parsed.separate_q_decoder)
    parsed.ca_only = int(parsed.ca_only)
    if parsed.init_from_checkpoint == "":
        parsed.init_from_checkpoint = ""
    if parsed.previous_checkpoint == "":
        parsed.previous_checkpoint = ""

    main(parsed)

