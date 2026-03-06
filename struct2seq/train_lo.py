"""Training script for Struct2SeqLO (Learning-Order autoregressive model)."""
from __future__ import print_function

import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data.dataset import Subset

sys.path.insert(0, '..')
from struct2seq import data, noam_opt, struct2seq_lo

from utils import featurize, loss_nll, load_checkpoint, setup_device_rng, get_args




def main() -> None:
    args = get_args()

    if args.model_type != 'structure_lo':
        raise ValueError(
            "train_lo.py requires --model_type structure_lo, "
            f"got {args.model_type!r}.",
        )

    device = setup_device_rng(args)

    # Build model
    model = struct2seq_lo.Struct2SeqLO(
        num_letters=args.vocab_size,
        node_features=args.hidden,
        edge_features=args.hidden,
        hidden_dim=args.hidden,
        k_neighbors=args.k_neighbors,
        protein_features=args.features,
        dropout=args.dropout,
        num_samples=args.num_samples,
        p_encoder_arch=args.p_encoder_arch,
        p_decoder_arch=args.p_decoder_arch,
        q_encoder_arch=args.q_encoder_arch,
        q_decoder_arch=args.q_decoder_arch,
        separate_encoder=args.separate_encoder,
        separate_decoder=args.separate_decoder,
    ).to(device)
    print('Number of parameters: {}'.format(
        sum(p.numel() for p in model.parameters())
    ))

    if args.restore != '':
        load_checkpoint(args.restore, model)

    optimizer = noam_opt.get_std_opt(model.parameters(), args.hidden)

    # Dataset
    dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)
    dataset_indices = {d['name']: i for i, d in enumerate(dataset)}
    with open(args.file_splits) as f:
        dataset_splits = json.load(f)

    train_set, validation_set, test_set = [
        Subset(dataset, [
            dataset_indices[name]
            for name in dataset_splits[key]
            if name in dataset_indices
        ])
        for key in ['train', 'validation', 'test']
    ]
    loader_train, loader_validation, loader_test = [
        data.StructureLoader(d, batch_size=args.batch_tokens)
        for d in [train_set, validation_set, test_set]
    ]
    print('Training:{}, Validation:{}, Test:{}'.format(
        len(train_set), len(validation_set), len(test_set),
    ))

    # Logging
    if args.name != '':
        base_folder = 'log/' + args.name + '/'
    else:
        base_folder = time.strftime('log/lo_%y%b%d_%I%M%p/', time.localtime())
    os.makedirs(base_folder, exist_ok=True)
    for sub in ['checkpoints']:
        os.makedirs(base_folder + sub, exist_ok=True)

    logfile = base_folder + 'log.txt'
    with open(logfile, 'w') as f:
        f.write('Epoch\tTrain_PPL\tVal_PPL\tTrain_ELBO\tTrain_NLL\tVal_NLL\n')
    with open(base_folder + 'args.json', 'w') as f:
        json.dump(vars(args), f)

    # Training loop
    start_train = time.time()
    epoch_losses_valid: list[float] = []
    epoch_checkpoints: list[str] = []
    total_step = 0

    for e in range(args.epochs):
        model.train()
        train_elbo_sum, train_nll_sum, train_weights = 0.0, 0.0, 0.0

        for train_i, batch in enumerate(loader_train):
            X, S, mask, lengths = featurize(
                batch, device, shuffle_fraction=args.shuffle,
            )

            optimizer.zero_grad()
            loss, info = model.compute_elbo_paper(X, S, lengths, mask)

            loss.backward()
            optimizer.step()

            # Decoder-based NLL for logging (matches validation metric)
            with torch.no_grad():
                log_probs = model(X, S, lengths, mask)
                _, loss_avg = loss_nll(S, log_probs, mask)

            total_step += 1
            n_tokens = mask.sum().item()
            train_elbo_sum += info["elbo"].item() * n_tokens
            train_nll_sum += loss_avg.item() * n_tokens
            train_weights += n_tokens

            if total_step % 100 == 0:
                elapsed = time.time() - start_train
                print(
                    "Step {} | {:.0f}s | ELBO {:.4f} | NLL {:.4f}"
                    " | PPL {:.2f} | dF {:.4f}".format(
                        total_step,
                        elapsed,
                        info["elbo"].item(),
                        loss_avg.item(),
                        float(np.exp(loss_avg.item())),
                        info.get("delta_F_abs", torch.tensor(0.0)).item(),
                    )
                )

            if total_step % 5000 == 0:
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                }, base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e + 1, total_step))

        # Validation
        model.eval()
        with torch.no_grad():
            val_sum, val_weights = 0., 0.
            for _, batch in enumerate(loader_validation):
                X, S, mask, lengths = featurize(batch, device)

                if args.eval_mode == 'is_q':
                    loglik_per_res = model.compute_loglik_is_q(
                        X,
                        S,
                        lengths,
                        mask,
                        num_samples_eval=args.eval_num_samples,
                    )
                    L_tensor = torch.tensor(
                        lengths,
                        dtype=torch.float32,
                        device=loglik_per_res.device,
                    )
                    nll_batch = -(loglik_per_res * L_tensor).sum().cpu().item()
                    val_sum += nll_batch
                    val_weights += L_tensor.sum().cpu().item()
                elif args.eval_mode == 'mc_p':
                    loglik_per_res = model.compute_loglik_mc_p(
                        X,
                        S,
                        lengths,
                        mask,
                        num_samples_eval=args.eval_num_samples,
                    )
                    L_tensor = torch.tensor(
                        lengths,
                        dtype=torch.float32,
                        device=loglik_per_res.device,
                    )
                    nll_batch = -(loglik_per_res * L_tensor).sum().cpu().item()
                    val_sum += nll_batch
                    val_weights += L_tensor.sum().cpu().item()
                else:
                    log_probs = model(X, S, lengths, mask)
                    loss_per_res, loss_avg = loss_nll(S, log_probs, mask)
                    val_sum += torch.sum(loss_per_res * mask).cpu().item()
                    val_weights += torch.sum(mask).cpu().item()

        train_elbo = train_elbo_sum / train_weights
        train_nll = train_nll_sum / train_weights
        val_nll = val_sum / val_weights
        val_ppl = np.exp(val_nll)
        train_ppl = np.exp(train_nll)

        print('Epoch {} | Train ELBO {:.4f} | Train PPL {:.2f} | Val PPL {:.2f}'.format(
            e + 1, train_elbo, train_ppl, val_ppl,
        ))

        with open(logfile, 'a') as f:
            f.write('{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                e + 1, train_ppl, val_ppl, train_elbo, train_nll, val_nll,
            ))

        ckpt = base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e + 1, total_step)
        torch.save({
            'epoch': e,
            'hyperparams': vars(args),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
        }, ckpt)

        epoch_losses_valid.append(val_ppl)
        epoch_checkpoints.append(ckpt)

    # Best model
    best_idx = int(np.argmin(epoch_losses_valid))
    best_ckpt = epoch_checkpoints[best_idx]
    best_copy = base_folder + 'best_checkpoint_epoch{}.pt'.format(best_idx + 1)
    best_validation_ppl = epoch_losses_valid[best_idx]
    best_train_ppl = np.exp(train_nll_sum / train_weights)
    shutil.copy(best_ckpt, best_copy)
    load_checkpoint(best_copy, model)

    # Test
    model.eval()
    with torch.no_grad():
        test_sum, test_weights = 0., 0.
        for _, batch in enumerate(loader_test):
            X, S, mask, lengths = featurize(batch, device)

            if args.eval_mode == 'is_q':
                loglik_per_res = model.compute_loglik_is_q(
                    X,
                    S,
                    lengths,
                    mask,
                    num_samples_eval=args.eval_num_samples,
                )
                L_tensor = torch.tensor(
                    lengths,
                    dtype=torch.float32,
                    device=loglik_per_res.device,
                )
                nll_batch = -(loglik_per_res * L_tensor).sum().cpu().item()
                test_sum += nll_batch
                test_weights += L_tensor.sum().cpu().item()
            elif args.eval_mode == 'mc_p':
                loglik_per_res = model.compute_loglik_mc_p(
                    X,
                    S,
                    lengths,
                    mask,
                    num_samples_eval=args.eval_num_samples,
                )
                L_tensor = torch.tensor(
                    lengths,
                    dtype=torch.float32,
                    device=loglik_per_res.device,
                )
                nll_batch = -(loglik_per_res * L_tensor).sum().cpu().item()
                test_sum += nll_batch
                test_weights += L_tensor.sum().cpu().item()
            else:
                log_probs = model(X, S, lengths, mask)
                loss_per_res, _ = loss_nll(S, log_probs, mask)
                test_sum += torch.sum(loss_per_res * mask).cpu().item()
                test_weights += torch.sum(mask).cpu().item()

    test_ppl = np.exp(test_sum / test_weights)
    print('Perplexity\tTest:{}'.format(test_ppl))

    with open(base_folder + 'results.txt', 'w') as f:
        f.write('Best epoch: {}\nPerplexities:\n\tTrain: {}\n\tValidation: {}\n\tTest: {}'.format(
            best_idx + 1, best_train_ppl, best_validation_ppl, test_ppl,
        ))


if __name__ == '__main__':
    main()
