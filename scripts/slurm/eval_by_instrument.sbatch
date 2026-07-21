#!/bin/bash
#SBATCH --job-name=smg_eval_by_inst
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

# Per-instrument-class re-scoring of an EXISTING eval output tree
# (metrics only, no generation — the eval root must still contain the
# per-sample audio incl. the *_16000.wav resamples).
#
# Required env vars (pass via --export=ALL after exporting them, so the
# JSON's commas survive; --export=ALL,GROUPS_JSON=... would split on them):
#   EVAL_ROOT    : eval dir with per-sample subdirs (00000/...)
#   RESULTS_TAG  : output name -> logs/eval_results/<RESULTS_TAG>.json
#   GROUPS_JSON  : e.g. '{"drums": [14], "bass": [4], "guitar": [3]}'
#     ... or GROUPS_FILE : path to a JSON file with the same content.
# Optional:
#   SKIP_FAD=1   : skip the (set-level, small-n-noisy) FAD per group.

EVAL_ROOT="${EVAL_ROOT:?EVAL_ROOT must point at an existing eval dir}"
RESULTS_TAG="${RESULTS_TAG:?RESULTS_TAG required}"
GROUPS_JSON="${GROUPS_JSON:-}"
GROUPS_FILE="${GROUPS_FILE:-}"
SKIP_FAD="${SKIP_FAD:-0}"

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh
export PYTHONUNBUFFERED=1

[ -d "${EVAL_ROOT}" ] || { echo "ERROR: eval root not found: ${EVAL_ROOT}"; exit 1; }

GROUP_ARGS=()
if [ -n "${GROUPS_FILE}" ]; then
  GROUP_ARGS+=(--groups_file "${GROUPS_FILE}")
elif [ -n "${GROUPS_JSON}" ]; then
  GROUP_ARGS+=(--groups "${GROUPS_JSON}")
else
  echo "ERROR: set GROUPS_JSON or GROUPS_FILE"; exit 1
fi

EXTRA_ARGS=()
if [ "${SKIP_FAD}" = "1" ]; then
  EXTRA_ARGS+=(--skip_fad)
fi

echo "============================================================"
echo "Per-instrument eval"
echo "  eval_root:    ${EVAL_ROOT}"
echo "  results_tag:  ${RESULTS_TAG}"
echo "  groups:       ${GROUPS_JSON:-file:${GROUPS_FILE}}"
echo "  node:         $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started:      $(date)"
echo "============================================================"

python scripts/eval_by_instrument.py \
  --eval_root "${EVAL_ROOT}" \
  --results_tag "${RESULTS_TAG}" \
  --results_save_dir "logs/eval_results" \
  "${GROUP_ARGS[@]}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[eval_by_inst] finished at $(date)"
