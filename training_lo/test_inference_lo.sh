#!/bin/bash

# Example inference script for a trained LO-ARM model checkpoint.

set -euo pipefail

path_to_PDB="../inputs/PDB_monomers/pdbs/5L33.pdb"
chains_to_design="A"

output_dir="../outputs/training_lo_test_output"
mkdir -p "${output_dir}"

python ../protein_mpnn_lo_run.py \
  --path_to_model_weights "./training_lo/exp_lo/model_weights" \
  --model_name "epoch_last" \
  --pdb_path "${path_to_PDB}" \
  --pdb_path_chains "${chains_to_design}" \
  --out_folder "${output_dir}" \
  --num_seq_per_target 8 \
  --sampling_temp "0.1" \
  --seed 37 \
  --batch_size 1 \
  --order_temperature 1.0

