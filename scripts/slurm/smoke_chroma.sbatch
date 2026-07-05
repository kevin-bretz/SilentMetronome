#!/bin/bash
#SBATCH --job-name=smg_chroma_smoke
#SBATCH --partition=gpu-short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Smoke-test BOTH Phase A (aux only) and Phase B (aux + DiT cond) chroma
# configs end-to-end on a single gpu-short slot. ~20 train steps + a val
# pass + a checkpoint write each. If both pass, the long gpu-a100-80g
# slots are safe to wait out.

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"
mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

run_smoke() {
  local label="$1"
  local cfg="$2"
  local save="${PROJECT_DIR}/models/_smoke_${label}_$$"
  echo "============================================================"
  echo "SMOKE [$label]: $cfg"
  echo "  save_dir: $save"
  echo "  started:  $(date)"
  echo "============================================================"
  python scripts/train_prefix_dec_online.py \
    --args.load "${cfg}" \
    --save_dir "${save}"
  local rc=$?
  echo "[smoke:$label] python exit=${rc}"
  rm -rf "${save}" || true
  return ${rc}
}

run_smoke "phaseA_aux"      configs/online/online_prefix_decoder/online_prefix_decoder_fv0_k50_chroma_smoke.yml
run_smoke "phaseB_aux_cond" configs/online/online_prefix_decoder/online_prefix_decoder_fv0_k50_chroma_cond_smoke.yml

echo "[smoke] BOTH PHASES PASSED"
