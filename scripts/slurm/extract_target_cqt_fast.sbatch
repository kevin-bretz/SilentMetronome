#!/bin/bash
#SBATCH --job-name=smg_cqt_t
#SBATCH --partition=cpu-short,cpu-zen4
#SBATCH --time=03:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

# Fan-out resubmission: same target_cqt extraction as the original sbatch
# but allowed to land on cpu-short (~700 idle CPUs RIGHT NOW) or cpu-zen4
# (~80 idle), whichever Slurm picks. Each shard takes ~50-60 min on a
# completed run (proved by shards 0-3) so 4h limit is comfortable.
#
# IMPORTANT: this sbatch hardcodes NUM_SHARDS=64 so the stride pattern
# matches the original 64-shard partition (shards 0-3 already complete).
# Submit ONLY the missing range, e.g.:
#   sbatch --array=4-63 scripts/slurm/extract_target_cqt_fast.sbatch
# Each task uses SLURM_ARRAY_TASK_ID as its shard_idx, so shards align
# with the original numbering.

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

NUM_SHARDS=64                                  # hardcoded — preserves stride alignment
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" = "1" ]; then OVERWRITE_FLAG="--overwrite"; fi

echo "============================================================"
echo "Target-CQT extraction (fan-out: cpu-short / cpu-zen4)"
echo "  shard:        ${SHARD_IDX}/${NUM_SHARDS}"
echo "  partition:    ${SLURM_JOB_PARTITION:-?}"
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
