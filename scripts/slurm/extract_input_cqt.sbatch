#!/bin/bash
#SBATCH --job-name=smg_cqt_i
#SBATCH --partition=cpu-short,cpu-zen4
#SBATCH --time=03:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

# CPU array job: extract per-window input-mix CQT from audio (librosa cqt
# on 84 bins from C1). Same shape as ``target_cqt.pt`` ([T, 84] float16),
# computed from the sum of all input-stem FLAC slices.
#
# Usage:
#   sbatch --array=0-63 scripts/slurm/extract_input_cqt.sbatch
# Re-run only missing shards:
#   sbatch --array=4-63 scripts/slurm/extract_input_cqt.sbatch
# Overwrite existing input_cqt.pt files:
#   OVERWRITE=1 sbatch --array=0-63 scripts/slurm/extract_input_cqt.sbatch

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

NUM_SHARDS=64
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" = "1" ]; then OVERWRITE_FLAG="--overwrite"; fi

echo "============================================================"
echo "Input-CQT extraction (audio CQT -> per-frame log-magnitude)"
echo "  shard:        ${SHARD_IDX}/${NUM_SHARDS}"
echo "  partition:    ${SLURM_JOB_PARTITION:-?}"
echo "  node:         $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started:      $(date)"
echo "============================================================"

export PYTHONUNBUFFERED=1

python scripts/extract_input_cqt.py \
  --shard_idx "${SHARD_IDX}" \
  --num_shards "${NUM_SHARDS}" \
  --splits train valid test \
  ${OVERWRITE_FLAG}

echo "[done] $(date)"
