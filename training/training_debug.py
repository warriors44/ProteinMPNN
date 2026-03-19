from __future__ import annotations

import argparse
import os.path
from typing import Tuple


def main(args: argparse.Namespace) -> None:
    import glob
    import json
    import os
    import sys
    import time
    import shutil
    import warnings
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import subprocess

    from utils import (
        worker_init_fn,
        get_pdbs,
        loader_pdb,
        build_training_clusters,
        PDB_dataset,
        StructureDataset,
        StructureLoader,
    )
    from model_utils import (
        featurize,
        loss_smoothed,
        loss_nll,
        get_std_opt,
        ProteinMPNN,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.mixed_precision and torch.cuda.is_available()))

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != "/":
        base_folder += "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ["model_weights"]
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    path_checkpoint = args.previous_checkpoint

    logfile = base_folder + "log.txt"
    if not path_checkpoint:
        # Epoch-level log: LO デバッグ版と同じカラム構成（一部は nan 固定）。
        with open(logfile, "w") as f:
            f.write(
                "epoch\tstep\ttime_s\t"
                "train_elbo_loss\ttrain_nll\ttrain_ppl\ttrain_acc\t"
                "valid_elbo_loss\tvalid_nll_proxy\tvalid_ppl_proxy\tvalid_acc\t"
                "valid_nll_isq\tvalid_ppl_isq\t"
                "i_mean\tdelta_F_abs\tgrad_norm\n"
            )

    # Debug step-level log: LO デバッグ版と同じカラム構成。
    debug_logfile = base_folder + "log_debug.txt"
    if args.debug_log_interval > 0 and (not path_checkpoint):
        with open(debug_logfile, "w") as f:
            f.write(
                "epoch\tstep\ttime_s\t"
                "train_elbo_loss\ttrain_nll\ttrain_ppl\ttrain_acc\t"
                "valid_elbo_loss\tvalid_nll_proxy\tvalid_ppl_proxy\tvalid_acc\t"
                "valid_nll_isq\tvalid_ppl_isq\t"
                "i_mean\tdelta_F_abs\tgrad_norm\n"
            )

    # Nonfinite debug log: AMP overflow / gradient NaN の簡易トレース。
    nonfinite_logfile = base_folder + "log_nonfinite_debug.txt"
    if not path_checkpoint:
        with open(nonfinite_logfile, "w") as f:
            f.write(
                "epoch\tstep\ttime_s\t"
                "scaler_scale_before\tscaler_scale_after\tstep_skipped\t"
                "loss_isfinite\tgrad_nonfinite_count\n"
            )

    data_path = args.path_for_training_data
    params = {
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

    train, valid, test = build_training_clusters(params, args.debug)

    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        worker_init_fn=worker_init_fn,
        **load_param,
    )
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        worker_init_fn=worker_init_fn,
        **load_param,
    )

    model = ProteinMPNN(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_encoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
    ).to(device)

    if path_checkpoint:
        checkpoint = torch.load(path_checkpoint)
        total_step = int(checkpoint["step"])
        start_epoch = int(checkpoint["epoch"])
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        total_step = 0
        start_epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)

    if path_checkpoint:
        optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    with ProcessPoolExecutor(max_workers=12) as executor:
        q_train: "queue.Queue[object]" = queue.Queue(maxsize=3)
        q_valid: "queue.Queue[object]" = queue.Queue(maxsize=3)
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
            pdb_dict_train,
            truncate=None,
            max_length=args.max_protein_length,
        )
        dataset_valid = StructureDataset(
            pdb_dict_valid,
            truncate=None,
            max_length=args.max_protein_length,
        )

        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)

        reload_c = 0
        for epoch_offset in range(args.num_epochs):
            t0 = time.time()
            epoch_idx = start_epoch + epoch_offset
            model.train()

            train_sum = 0.0
            train_weights = 0.0
            train_acc = 0.0
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
                        dataset_train,
                        batch_size=args.batch_size,
                    )
                    pdb_dict_valid = q_valid.get().result()
                    dataset_valid = StructureDataset(
                        pdb_dict_valid,
                        truncate=None,
                        max_length=args.max_protein_length,
                    )
                    loader_valid = StructureLoader(
                        dataset_valid,
                        batch_size=args.batch_size,
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

            for _, batch in enumerate(loader_train):
                optimizer.zero_grad()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                mask_for_loss = mask * chain_M

                if args.mixed_precision:
                    scaler_scale_before = float(scaler.get_scale())
                    with torch.cuda.amp.autocast():
                        log_probs = model(
                            X,
                            S,
                            mask,
                            chain_M,
                            residue_idx,
                            chain_encoding_all,
                        )
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

                    scaler.scale(loss_av_smoothed).backward()
                    scaler.unscale_(optimizer)

                    grad_nonfinite_count = 0
                    for param in model.parameters():
                        if param.grad is None:
                            continue
                        if not torch.isfinite(param.grad).all():
                            grad_nonfinite_count += 1

                    total_norm: float | None = None
                    if args.gradient_norm > 0.0:
                        total_norm_tensor = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.gradient_norm,
                        )
                        total_norm = float(total_norm_tensor.detach().cpu().item())
                        grad_norm_sum += total_norm
                        grad_norm_w += 1.0

                    scaler.step(optimizer)
                    scaler.update()
                    scaler_scale_after = float(scaler.get_scale())
                    step_skipped = int(scaler_scale_after < scaler_scale_before)
                else:
                    scaler_scale_before = float("nan")
                    log_probs = model(
                        X,
                        S,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_encoding_all,
                    )
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()

                    grad_nonfinite_count = 0
                    for param in model.parameters():
                        if param.grad is None:
                            continue
                        if not torch.isfinite(param.grad).all():
                            grad_nonfinite_count += 1

                    total_norm: float | None = None
                    if args.gradient_norm > 0.0:
                        total_norm_tensor = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.gradient_norm,
                        )
                        total_norm = float(total_norm_tensor.detach().cpu().item())
                        grad_norm_sum += total_norm
                        grad_norm_w += 1.0

                    optimizer.step()
                    scaler_scale_after = float("nan")
                    step_skipped = 0

                loss, _, true_false = loss_nll(S, log_probs, mask_for_loss)

                train_sum += float(torch.sum(loss * mask_for_loss).detach().cpu().item())
                train_acc += float(torch.sum(true_false * mask_for_loss).detach().cpu().item())
                train_weights += float(torch.sum(mask_for_loss).detach().cpu().item())

                total_step += 1

                # Nonfinite debug: 毎 step で簡易統計を記録。
                loss_isfinite = float(torch.isfinite(loss_av_smoothed).all().detach().cpu().item())
                dt_cur = float(time.time() - t0)
                with open(nonfinite_logfile, "a") as nf:
                    nf.write(
                        f"{epoch_idx + 1}\t{total_step}\t{dt_cur:.1f}\t"
                        f"{scaler_scale_before:.1f}\t{scaler_scale_after:.1f}\t{step_skipped}\t"
                        f"{int(loss_isfinite)}\t{grad_nonfinite_count}\n"
                    )

                if args.debug_log_interval > 0 and (total_step % args.debug_log_interval == 0):
                    dt_cur = float(time.time() - t0)
                    train_nll = train_sum / max(train_weights, 1e-8)
                    train_ppl = float(np.exp(train_nll))
                    train_accuracy = train_acc / max(train_weights, 1e-8)
                    grad_norm_avg = grad_norm_sum / max(grad_norm_w, 1.0)

                    with open(debug_logfile, "a") as df:
                        df.write(
                            f"{epoch_idx + 1}\t{total_step}\t{dt_cur:.1f}\t"
                            f"{float(loss_av_smoothed.detach().cpu().item()):.6f}\t"
                            f"{train_nll:.6f}\t{train_ppl:.3f}\t{train_accuracy:.4f}\t"
                            f"nan\tnan\tnan\tnan\tnan\tnan\t"
                            f"nan\tnan\t{grad_norm_avg:.6f}\n"
                        )

            model.eval()
            with torch.no_grad():
                validation_sum = 0.0
                validation_weights = 0.0
                validation_acc = 0.0
                for _, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    log_probs = model(
                        X,
                        S,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_encoding_all,
                    )
                    mask_for_loss = mask * chain_M
                    loss, _, true_false = loss_nll(S, log_probs, mask_for_loss)

                    validation_sum += float(torch.sum(loss * mask_for_loss).detach().cpu().item())
                    validation_acc += float(torch.sum(true_false * mask_for_loss).detach().cpu().item())
                    validation_weights += float(torch.sum(mask_for_loss).detach().cpu().item())

            train_nll_epoch = train_sum / max(train_weights, 1e-8)
            train_accuracy = train_acc / max(train_weights, 1e-8)
            train_ppl_epoch = float(np.exp(train_nll_epoch))
            validation_nll_epoch = validation_sum / max(validation_weights, 1e-8)
            validation_accuracy = validation_acc / max(validation_weights, 1e-8)
            validation_ppl_epoch = float(np.exp(validation_nll_epoch))

            grad_norm_avg_epoch = grad_norm_sum / max(grad_norm_w, 1.0)

            # LO 版に合わせた数値フォーマット（ただし ELBO / ISQ / i_mean / delta_F_abs は nan）。
            t1 = time.time()
            dt = float(t1 - t0)

            with open(logfile, "a") as f:
                f.write(
                    f"{epoch_idx + 1}\t{total_step}\t{dt:.1f}\t"
                    f"nan\t{train_nll_epoch:.6f}\t{train_ppl_epoch:.3f}\t{train_accuracy:.4f}\t"
                    f"nan\t{validation_nll_epoch:.6f}\t{validation_ppl_epoch:.3f}\t{validation_accuracy:.4f}\t"
                    f"nan\tnan\t"
                    f"nan\tnan\t{grad_norm_avg_epoch:.6f}\n"
                )
            print(
                f"epoch: {epoch_idx + 1}, step: {total_step}, time: {dt:.1f}s, "
                f"train_nll: {train_nll_epoch:.4f}, valid_nll: {validation_nll_epoch:.4f}, "
                f"train_ppl: {train_ppl_epoch:.3f}, valid_ppl: {validation_ppl_epoch:.3f}, "
                f"grad_norm: {grad_norm_avg_epoch:.3f}, "
                f"train_acc: {train_accuracy:.3f}, valid_acc: {validation_accuracy:.3f}"
            )

            checkpoint_filename_last = base_folder + "model_weights/epoch_last.pt"
            torch.save(
                {
                    "epoch": epoch_idx + 1,
                    "step": total_step,
                    "num_edges": args.num_neighbors,
                    "noise_level": args.backbone_noise,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.optimizer.state_dict(),
                },
                checkpoint_filename_last,
            )

            if (epoch_idx + 1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = (
                    base_folder + f"model_weights/epoch{epoch_idx + 1}_step{total_step}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch_idx + 1,
                        "step": total_step,
                        "num_edges": args.num_neighbors,
                        "noise_level": args.backbone_noise,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.optimizer.state_dict(),
                    },
                    checkpoint_filename,
                )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    argparser.add_argument(
        "--path_for_training_data",
        type=str,
        default="my_path/pdb_2021aug02",
        help="path for loading training data",
    )
    argparser.add_argument(
        "--path_for_outputs",
        type=str,
        default="./exp_020",
        help="path for logs and model weights",
    )
    argparser.add_argument(
        "--previous_checkpoint",
        type=str,
        default="",
        help="path for previous model weights, e.g. file.pt",
    )
    argparser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="number of epochs to train for",
    )
    argparser.add_argument(
        "--save_model_every_n_epochs",
        type=int,
        default=10,
        help="save model weights every n epochs",
    )
    argparser.add_argument(
        "--reload_data_every_n_epochs",
        type=int,
        default=2,
        help="reload training data every n epochs",
    )
    argparser.add_argument(
        "--num_examples_per_epoch",
        type=int,
        default=1000000,
        help="number of training example to load for one epoch",
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="number of tokens for one batch",
    )
    argparser.add_argument(
        "--max_protein_length",
        type=int,
        default=10000,
        help="maximum length of the protein complext",
    )
    argparser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="hidden model dimension",
    )
    argparser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=3,
        help="number of encoder layers",
    )
    argparser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=3,
        help="number of decoder layers",
    )
    argparser.add_argument(
        "--num_neighbors",
        type=int,
        default=48,
        help="number of neighbors for the sparse graph",
    )
    argparser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout level; 0.0 means no dropout",
    )
    argparser.add_argument(
        "--backbone_noise",
        type=float,
        default=0.2,
        help="amount of noise added to backbone during training",
    )
    argparser.add_argument(
        "--rescut",
        type=float,
        default=3.5,
        help="PDB resolution cutoff",
    )
    argparser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="minimal data loading for debugging",
    )
    argparser.add_argument(
        "--gradient_norm",
        type=float,
        default=-1.0,
        help="clip gradient norm, set to negative to omit clipping",
    )
    argparser.add_argument(
        "--mixed_precision",    
        action="store_true",
        help="train with mixed precision",
    )
    argparser.add_argument(
        "--debug_log_interval",
        type=int,
        default=0,
        help="If >0, write step-level debug log every N steps to log_debug.txt.",
    )

    args = argparser.parse_args()
    main(args)
