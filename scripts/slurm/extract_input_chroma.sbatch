#!/bin/bash
#SBATCH --job-name=smg_chroma_i
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

# CPU array job: extract per-window INPUT-mix chroma from audio (librosa
# chroma_cqt). HPSS off by default — full HPSS is too slow at this scale.
# Idempotent (safe to re-run; already-extracted windows skipped).
#
# Usage:
#   sbatch scripts/slurm/extract_input_chroma.sbatch
# Toggle HPSS:
#   HPSS=1 sbatch scripts/slurm/extract_input_chroma.sbatch
# Override array breadth:
#   sbatch --array=0-63 scripts/slurm/extract_input_chroma.sbatch

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
HPSS_FLAG=""
if [ "${HPSS:-0}" = "1" ]; then HPSS_FLAG="--hpss"; fi
OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" = "1" ]; then OVERWRITE_FLAG="--overwrite"; fi

echo "============================================================"
echo "Input-chroma extraction (audio chroma_cqt -> per-frame pitch class)"
echo "  shard:        ${SHARD_IDX}/${NUM_SHARDS}"
echo "  hpss:         ${HPSS:-0}"
echo "  node:         $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started:      $(date)"
echo "============================================================"

export PYTHONUNBUFFERED=1

python scripts/extract_input_chroma.py \
  --shard_idx "${SHARD_IDX}" \
  --num_shards "${NUM_SHARDS}" \
  --splits train valid test \
  ${HPSS_FLAG} ${OVERWRITE_FLAG}

echo "[done] $(date)"
