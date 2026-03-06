#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q week
#PBS -l walltime=168:00:00


cd $PBS_O_WORKDIR
source ../.venv/bin/activate

python3 ./training_lo.py \
  --path_for_outputs ./exp_lo/ \
  --path_for_training_data ../data/pdb_2021aug02_sample \
  --num_examples_per_epoch 1000 \
  --save_model_every_n_epochs 50 \
  --num_lo_samples 2 \
  --eval_mode is_q \
  --eval_num_samples 8 \
  --separate_q_decoder 0

