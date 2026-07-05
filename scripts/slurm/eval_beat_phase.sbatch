#!/bin/bash
#SBATCH --job-name=smg_eval_bp
#SBATCH --partition=gpu-short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --exclude=node870
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Run the FAD / beat-F / COCOLA eval pipeline for a single trained checkpoint.
#
# Required env vars (pass via sbatch --export):
#   MODEL_PATH : absolute path to the .ckpt to evaluate
# Optional:
#   SPLIT       : "valid" | "test" | "train" (default: test, matches baseline)
#   NUM_SAMPLES : default 1024 (README recommendation for FAD/COCOLA)
#   BATCH_SIZE  : default 64
#   MODEL_TYPE  : default prefix_decoder_online

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be an absolute .ckpt path}"
SPLIT="${SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_TYPE="${MODEL_TYPE:-prefix_decoder_online}"
GEN_KWARGS="${GEN_KWARGS:-{\}}"

# Match the existing project convention: results JSON at
# logs/eval_results/<exp>_eval.json and per-eval audio under
# models/<exp>/<exp>_eval/. <exp> is the model directory's basename.
EXP_NAME="$(basename "$(dirname "${MODEL_PATH}")")"
RESULTS_TAG="${RESULTS_TAG:-${EXP_NAME}_eval}"

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

# Force unbuffered stdout — same reasoning as in the training sbatch.
export PYTHONUNBUFFERED=1

[ -f "${MODEL_PATH}" ] || { echo "ERROR: ckpt not found: ${MODEL_PATH}"; exit 1; }

echo "============================================================"
echo "Beat-phase eval"
echo "  model_type:    ${MODEL_TYPE}"
echo "  model_path:    ${MODEL_PATH}"
echo "  split:         ${SPLIT}"
echo "  num_samples:   ${NUM_SAMPLES}"
echo "  batch_size:    ${BATCH_SIZE}"
echo "  results_tag:   ${RESULTS_TAG}"
echo "  gen_kwargs:    ${GEN_KWARGS}"
echo "  node:          $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started:       $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

python scripts/gen_pred/gen_and_evaluate.py \
  --model_type "${MODEL_TYPE}" \
  --model_path "${MODEL_PATH}" \
  --split "${SPLIT}" \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --save_dir_name "${RESULTS_TAG}" \
  --results_save_dir "logs/eval_results" \
  --gen_kwargs "${GEN_KWARGS}"

echo "[eval] finished at $(date)"
