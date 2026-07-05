#!/bin/bash
#SBATCH --job-name=smg_extract_rms
#SBATCH --partition=gpu-short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Optional: export VENV_ACTIVATE before sbatch, e.g.
#   sbatch --export=ALL,VENV_ACTIVATE=/path/to/.venv/bin/activate scripts/slurm/preprocess_extract_rms.sbatch
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

mkdir -p logs

DATASETS="${DATASETS:-slakh2100}"
DATASET_ROOT="${DATASET_ROOT:-stream_music_gen_data}"
OUTPUT_DIR="${OUTPUT_DIR:-stream_music_gen_data/rms_50hz}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Single-process extraction does not benefit from requesting multiple GPUs.
python stream_music_gen/dataset/extract_rms.py \
  --datasets ${DATASETS} \
  --dataset_root "${DATASET_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_workers "${NUM_WORKERS}"
