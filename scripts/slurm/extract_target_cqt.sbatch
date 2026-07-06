#!/bin/bash
#SBATCH --job-name=smg_cqt_t
#SBATCH --partition=cpu-zen4
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-63
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

# CPU array job: extract per-window target-stem CQT from audio (librosa.cqt
# at 84 bins / 7 octaves / C1-C8). Idempotent.
# Mirrors extract_input_chroma.sbatch's array breadth, since CQT cost is
# similar to chroma_cqt cost (both use librosa CQT internals).
#
# Usage:
#   sbatch scripts/slurm/extract_target_cqt.sbatch
# Override array breadth:
#   sbatch --array=0-63 scripts/slurm/extract_target_cqt.sbatch

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" = "1" ]; then OVERWRITE_FLAG="--overwrite"; fi

echo "============================================================"
echo "Target-CQT extraction (audio CQT -> per-frame [T, 84] log-magnitude)"
echo "  shard:        ${SHARD_IDX}/${NUM_SHARDS}"
echo "  node:         $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started:      $(date)"
echo "============================================================"

export PYTHONUNBUFFERED=1

python scripts/extract_target_cqt.py \
  --shard_idx "${SHARD_IDX}" \
  --num_shards "${NUM_SHARDS}" \
  --splits train valid test \
  ${OVERWRITE_FLAG}

echo "[done] $(date)"
