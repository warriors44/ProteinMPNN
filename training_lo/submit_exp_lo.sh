#!/bin/bash
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -q week
#PBS -l walltime=168:00:00


cd $PBS_O_WORKDIR
source ../.venv/bin/activate

python3 ./training_lo_debug.py \
  --path_for_outputs ./exp_lo \
  --path_for_training_data ../data/pdb_2021aug02 \
  --save_model_every_n_epochs 10 \
  --eval_num_samples 8 \
  --separate_q_decoder 0 \
  --batch_size 4000 \
  --num_epochs 1000 \
  --num_lo_samples 8 \
  --num_neighbors 48 \
  --max_protein_length 4000 \
  --debug_log_interval 1 \
    # --previous_checkpoint ./exp_lo/model_weights/epoch_best_proxy.pt \
    # --num_examples_per_epoch 1000 \