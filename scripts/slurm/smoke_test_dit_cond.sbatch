#!/bin/bash
#SBATCH --job-name=smoke_dit
#SBATCH --partition=gpu-short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Smoke test for the DiT-style per-layer beat-phase conditioning. Runs
# scripts/smoke_test_dit_cond.py on a single A100. Verifies the model
# instantiates with the new flags, a forward pass produces finite logits,
# and a tiny generate() runs end-to-end. Cheap (<5 min).

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

export PYTHONUNBUFFERED=1

echo "============================================================"
echo "DiT beat-phase conditioning — smoke test"
echo "  node:    $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started: $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

python scripts/smoke_test_dit_cond.py

echo "[smoke] finished at $(date)"
