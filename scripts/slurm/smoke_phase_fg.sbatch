#!/bin/bash
#SBATCH --job-name=smg_smoke_fg
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

# Smoke test for Phase F (cond + mp + cqt aux) and Phase G (mp + cqt + beat
# aux full, no cond). Runs ~30 train steps each with compile=false /
# batch_size=2 / num_workers=2 to surface wiring bugs (shape mismatch,
# missing tensor in batch, NaN loss, etc.) before committing to 200K x2.

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"
mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/${USER}_inductor_${SLURM_JOB_ID}"
export WANDB_MODE=disabled
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

run_one () {
  local label="$1"
  local cfg="$2"
  local save_dir="/tmp/smoke_${label}_${SLURM_JOB_ID}"
  echo "=========================================================="
  echo "SMOKE — ${label}: ${cfg}"
  echo "  save_dir: ${save_dir}"
  echo "  started:  $(date)"
  echo "=========================================================="
  rm -rf "${save_dir}"
  mkdir -p "${save_dir}"
  python scripts/train_prefix_dec_online.py \
    --args.load "${cfg}" \
    --save_dir "${save_dir}" \
    --train_steps 30 \
    --val_interval 25 \
    --limit_val_batches 2 \
    --batch_size 2 \
    --num_workers 2 \
    --LitOnlinePrefixDecoderMultiOut.compile false \
    --checkpoint_interval 100000 \
    --sample_interval 100000
  echo "[smoke ${label}] OK at $(date)"
  rm -rf "${save_dir}"
}

run_one "phase_f" "configs/online/online_prefix_decoder/online_prefix_decoder_fv0_k50_beat_phase_dit_mp_cqt_aux.yml"
run_one "phase_g" "configs/online/online_prefix_decoder/online_prefix_decoder_fv0_k50_beat_phase_aux_mp_cqt_aux.yml"

echo "[smoke] both passed at $(date)"
rm -rf "${TORCHINDUCTOR_CACHE_DIR}" 2>/dev/null || true
