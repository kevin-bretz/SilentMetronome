#!/bin/bash
#SBATCH --job-name=smg_pc_equiv
#SBATCH --partition=gpu-short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Equivalence test: naive per-step DiT conditioning vs chunk-ahead
# modulation precompute. Exits nonzero on FAIL (so downstream jobs can
# depend on it with afterok).
#
# Required env vars (pass via sbatch --export):
#   MODEL_PATH : absolute path to the .ckpt (must be a DiT-conditioned model)
# Optional:
#   SEQ_CHUNKS    : chunks per generation run (default 5)
#   DATA_BASE_DIR : dataloader data_base_dir override
#   OUT_JSON      : results path (default logs/precompute_equiv/<exp>.json)

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be an absolute .ckpt path}"
SEQ_CHUNKS="${SEQ_CHUNKS:-5}"

EXP_NAME="$(basename "$(dirname "${MODEL_PATH}")")"
OUT_JSON="${OUT_JSON:-logs/precompute_equiv/${EXP_NAME}.json}"

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs logs/precompute_equiv
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

export PYTHONUNBUFFERED=1

[ -f "${MODEL_PATH}" ] || { echo "ERROR: ckpt not found: ${MODEL_PATH}"; exit 1; }

echo "============================================================"
echo "Precompute equivalence test"
echo "  model_path: ${MODEL_PATH}"
echo "  seq_chunks: ${SEQ_CHUNKS}"
echo "  out_json:   ${OUT_JSON}"
echo "  node:       $(hostname)    job: ${SLURM_JOB_ID}"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

EXTRA_ARGS=""
if [ -n "${DATA_BASE_DIR:-}" ]; then
  EXTRA_ARGS="--data_base_dir ${DATA_BASE_DIR}"
fi

python scripts/gen_pred/test_precompute_equivalence.py \
  --model_path "${MODEL_PATH}" \
  --seq_chunks "${SEQ_CHUNKS}" \
  ${EXTRA_ARGS} \
  --out_json "${OUT_JSON}"

echo "[pc-equiv] finished at $(date)"
