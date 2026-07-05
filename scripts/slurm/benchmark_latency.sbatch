#!/bin/bash
#SBATCH --job-name=smg_latency
#SBATCH --partition=gpu-short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --exclude=node870,node874
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Streaming-inference latency benchmark for the SOTA checkpoint.
#
# Required env vars (pass via sbatch --export):
#   MODEL_PATH : absolute path to the .ckpt
# Optional:
#   CHUNK_FRAMES   : frames per chunk (default 50 = 1s of audio)
#   N_CHUNKS_LONG  : chunks for steady-state measurement (default 10)
#   N_RUNS         : timed repeats (default 20)
#   OUT_JSON       : results path (default logs/latency/<exp>.json)

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be an absolute .ckpt path}"
N_CHUNKS_PER_SESSION="${N_CHUNKS_PER_SESSION:-10}"
N_SESSIONS="${N_SESSIONS:-30}"
N_WARMUP_SESSIONS="${N_WARMUP_SESSIONS:-3}"
N_DECODE_RUNS="${N_DECODE_RUNS:-100}"

EXP_NAME="$(basename "$(dirname "${MODEL_PATH}")")"
DIT_PRECOMPUTE="${DIT_PRECOMPUTE:-0}"
PC_SUFFIX=""
if [ "${DIT_PRECOMPUTE}" = "1" ]; then
  PC_SUFFIX="_pc"
fi
OUT_JSON="${OUT_JSON:-logs/latency/${EXP_NAME}${PC_SUFFIX}.json}"

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs logs/latency
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

export PYTHONUNBUFFERED=1

[ -f "${MODEL_PATH}" ] || { echo "ERROR: ckpt not found: ${MODEL_PATH}"; exit 1; }

echo "============================================================"
echo "Latency benchmark"
echo "  model_path:           ${MODEL_PATH}"
echo "  n_chunks_per_session: ${N_CHUNKS_PER_SESSION}"
echo "  n_sessions:           ${N_SESSIONS}"
echo "  n_warmup_sessions:    ${N_WARMUP_SESSIONS}"
echo "  n_decode_runs:        ${N_DECODE_RUNS}"
echo "  out_json:             ${OUT_JSON}"
echo "  node:                 $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started:              $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

EXTRA_ARGS=""
if [ -n "${DATA_BASE_DIR:-}" ]; then
  EXTRA_ARGS="--data_base_dir ${DATA_BASE_DIR}"
fi
if [ "${DIT_PRECOMPUTE}" = "1" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --dit_precompute"
fi

python scripts/gen_pred/benchmark_latency.py \
  --model_path "${MODEL_PATH}" \
  --n_chunks_per_session "${N_CHUNKS_PER_SESSION}" \
  --n_sessions "${N_SESSIONS}" \
  --n_warmup_sessions "${N_WARMUP_SESSIONS}" \
  --n_decode_runs "${N_DECODE_RUNS}" \
  ${EXTRA_ARGS} \
  --out_json "${OUT_JSON}"

echo "[latency] finished at $(date)"
