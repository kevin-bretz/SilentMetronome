#!/bin/bash
#SBATCH --job-name=smg_eval_lb
#SBATCH --partition=gpu-short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# LiveBand-protocol eval (arXiv:2606.03803): 20 s generations on the Slakh
# test split, scored on two non-overlapping 10 s segments + drift.
# See scripts/eval_liveband_protocol.py for the protocol details.
#
# Required env vars (pass via sbatch --export=ALL,VAR=...):
#   MODEL_PATH  : absolute path to the .ckpt to evaluate
# Optional:
#   SEED            : sampling seed (default 42)
#   RESULTS_TAG     : default <exp>_eval_lb20_s<SEED>
#   NUM_SAMPLES     : default 1024
#   BATCH_SIZE      : default 32 (20 s sequences ~2x the 10 s memory)
#   DATA_BASE_DIR   : override the ckpt args.yml data_base_dir
#   FAD_BACKGROUND  : paired (default) | pooled
#   SKIP_CLAP=1     : skip the LAION-CLAP FAD
#   EXTRA_ARGS      : anything else for eval_liveband_protocol.py

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be an absolute .ckpt path}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DATA_BASE_DIR="${DATA_BASE_DIR:-}"
FAD_BACKGROUND="${FAD_BACKGROUND:-paired}"
SKIP_CLAP="${SKIP_CLAP:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

EXP_NAME="$(basename "$(dirname "${MODEL_PATH}")")"
RESULTS_TAG="${RESULTS_TAG:-${EXP_NAME}_eval_lb20_s${SEED}}"

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh
export PYTHONUNBUFFERED=1

[ -f "${MODEL_PATH}" ] || { echo "ERROR: ckpt not found: ${MODEL_PATH}"; exit 1; }

CLI_ARGS=(
  --model_path "${MODEL_PATH}"
  --results_tag "${RESULTS_TAG}"
  --results_save_dir "logs/eval_results"
  --split test
  --num_samples "${NUM_SAMPLES}"
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --fad_background "${FAD_BACKGROUND}"
)
if [ -n "${DATA_BASE_DIR}" ]; then
  CLI_ARGS+=(--data_base_dir "${DATA_BASE_DIR}")
fi
if [ "${SKIP_CLAP}" = "1" ]; then
  CLI_ARGS+=(--skip_fad_clap)
fi

echo "============================================================"
echo "LiveBand-protocol eval"
echo "  model_path:   ${MODEL_PATH}"
echo "  results_tag:  ${RESULTS_TAG}"
echo "  num_samples:  ${NUM_SAMPLES}   batch: ${BATCH_SIZE}   seed: ${SEED}"
echo "  fad_bg:       ${FAD_BACKGROUND}   skip_clap: ${SKIP_CLAP}"
echo "  node:         $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started:      $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# shellcheck disable=SC2086
python scripts/eval_liveband_protocol.py "${CLI_ARGS[@]}" ${EXTRA_ARGS}

echo "[eval_lb] finished at $(date)"
