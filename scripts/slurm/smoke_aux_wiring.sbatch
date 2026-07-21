#!/bin/bash
#SBATCH --job-name=smg_smoke_aux
#SBATCH --partition=gpu-short
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Lean smoke test for new aux head wiring. Skips Lightning, dataset
# loading, and wandb. Just builds the model with the release
# flags, runs forward + backward on synthetic tensors, asserts losses
# are finite and aux head gradients flow.

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"
mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh
export PYTHONUNBUFFERED=1
python scripts/smoke_aux_wiring.py
