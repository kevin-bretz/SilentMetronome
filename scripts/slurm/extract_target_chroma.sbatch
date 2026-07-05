#!/bin/bash
#SBATCH --job-name=smg_chroma_t
#SBATCH --partition=cpu-short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-7
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

# CPU array job: extract per-window TARGET-stem chroma from MIDI.
# Each array task processes 1/NUM_SHARDS of the windows. Idempotent (safe
# to re-run after a partial completion — already-extracted windows skipped).
#
# Usage:
#   sbatch scripts/slurm/extract_target_chroma.sbatch
# Override array breadth:
#   sbatch --array=0-15 scripts/slurm/extract_target_chroma.sbatch

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

NUM_SHARDS="${NUM_SHARDS_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"

echo "============================================================"
echo "Target-chroma extraction (MIDI -> per-frame pitch class)"
echo "  shard:        ${SHARD_IDX}/${NUM_SHARDS}"
echo "  node:         $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started:      $(date)"
echo "============================================================"

export PYTHONUNBUFFERED=1

python scripts/extract_target_chroma.py \
  --shard_idx "${SHARD_IDX}" \
  --num_shards "${NUM_SHARDS}" \
  --splits train valid test

echo "[done] $(date)"
