#!/bin/bash
#SBATCH --job-name=smg_mp_t
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

# CPU array job: extract per-window target-stem multipitch + velocity from MIDI.
# Each array task processes 1/NUM_SHARDS of the windows. Idempotent.
# Same sharding pattern as extract_target_cqt.sbatch.
#
# Usage:
#   sbatch scripts/slurm/extract_target_multipitch.sbatch
# Override array breadth:
#   sbatch --array=0-15 scripts/slurm/extract_target_multipitch.sbatch

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

NUM_SHARDS="${NUM_SHARDS_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" = "1" ]; then OVERWRITE_FLAG="--overwrite"; fi

echo "============================================================"
echo "Target-multipitch extraction (MIDI -> per-frame [T, 128] piano roll)"
echo "  shard:        ${SHARD_IDX}/${NUM_SHARDS}"
echo "  node:         $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started:      $(date)"
echo "============================================================"

export PYTHONUNBUFFERED=1

python scripts/extract_target_multipitch.py \
  --shard_idx "${SHARD_IDX}" \
  --num_shards "${NUM_SHARDS}" \
  --splits train valid test \
  ${OVERWRITE_FLAG}

echo "[done] $(date)"
