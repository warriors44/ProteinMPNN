#!/bin/bash
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -q week
#PBS -l walltime=168:00:00

cd $PBS_O_WORKDIR
source ../.venv/bin/activate

python ./training.py \
           --path_for_outputs "./exp_020_sample" \
           --path_for_training_data ../data/pdb_2021aug02_sample \
           --num_examples_per_epoch 1000 \
           --save_model_every_n_epochs 50 \
           --max_protein_length 4000 \
           --batch_size 4000 \
           --num_epochs 1000