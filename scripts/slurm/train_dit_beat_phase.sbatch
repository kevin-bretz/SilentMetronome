#!/bin/bash
#SBATCH --job-name=smg_dit_bp
#SBATCH --partition=gpu-short
#SBATCH --time=04:00:00
#SBATCH --signal=B:SIGTERM@180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Self-chaining training for DiT-style per-layer beat-phase conditioning.
# Same chain mechanics as train_ft_beat_phase.sbatch (signal forwarding,
# wandb watchdog, per-job inductor cache, self-resubmit) but points at
# the DiT config family.
#
# Required env vars (pass via sbatch --export):
#   FV       : future_visibility, one of "-50" | "0" | "50"
# Optional:
#   INIT_CKPT    : warm-start ckpt (default: train from random init).
#   EXP_NAME     : directory under models/ (default: pref_dec_online_fv${FV}_k50_beat_phase_dit)
#   TARGET_STEPS : training step target (default: 200000)

FV="${FV:?FV must be set to -50, 0, or 50}"

case "${FV}" in
  -50|0|50) : ;;
  *) echo "ERROR: FV='${FV}' must be one of -50, 0, 50"; exit 1 ;;
esac

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

mkdir -p slurm_logs

# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

CONFIG="configs/online/online_prefix_decoder/online_prefix_decoder_fv${FV}_k50_beat_phase_dit.yml"
INIT_CKPT="${INIT_CKPT:-}"   # empty => train from random init
EXP_NAME="${EXP_NAME:-pref_dec_online_fv${FV}_k50_beat_phase_dit}"
SAVE_DIR="${PROJECT_DIR}/models/${EXP_NAME}"
TARGET_STEPS="${TARGET_STEPS:-200000}"

echo "============================================================"
echo "DiT beat-phase training: FV=${FV}, k=50"
echo "  config:      ${CONFIG}"
echo "  init_ckpt:   ${INIT_CKPT:-<none: training from scratch>}"
echo "  save_dir:    ${SAVE_DIR}"
echo "  target:      ${TARGET_STEPS} steps"
echo "  node:        $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started:     $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# Fail fast if any of the critical inputs are missing
[ -f "${CONFIG}" ] || { echo "ERROR: config not found: ${CONFIG}"; exit 1; }
if [ -n "${INIT_CKPT}" ] && [ ! -f "${INIT_CKPT}" ]; then
  echo "ERROR: INIT_CKPT set but not found: ${INIT_CKPT}"; exit 1
fi
[ -d "${PROJECT_DIR}/stream_music_gen_data/precompute_audio_mixdown_20s_beat/slakh2100/train" ] \
  || { echo "ERROR: beat-phase data root missing"; exit 1; }

# Per-job inductor cache to avoid cross-slot Triton lock contention.
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/${USER}_inductor_${SLURM_JOB_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
echo "[chain] TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"

export PYTHONUNBUFFERED=1

# Run training in background so the shell can receive SIGTERM (from --signal=B)
# and forward it to python. Lightning catches SIGTERM, saves a checkpoint,
# and exits fit() cleanly.
forward_term() {
  echo "[chain] Received SIGTERM at $(date); forwarding to python pid=${PY_PID}"
  if [ -n "${PY_PID:-}" ]; then
    kill -TERM "${PY_PID}" 2>/dev/null || true
  fi
}
trap forward_term SIGTERM

TRAIN_CMD=(
  python scripts/train_prefix_dec_online.py
  --args.load "${CONFIG}"
  --save_dir "${SAVE_DIR}"
)
if [ -n "${INIT_CKPT}" ]; then
  TRAIN_CMD+=(--init_from_checkpoint "${INIT_CKPT}")
fi
"${TRAIN_CMD[@]}" &
PY_PID=$!

# Mid-slot pruner: every 60s after a 10-min grace, keep only the
# highest-step step=*.ckpt. Race-safe with Lightning's atomic ckpt write
# (tmp + rename — *.tmp files don't match step=*.ckpt). Caps mid-slot
# disk usage at one ckpt (~6GB) instead of letting the per-20k-step
# saves accumulate to 5-7 ckpts before the post-slot prune runs.
midslot_pruner() {
  sleep 600
  while kill -0 "${PY_PID}" 2>/dev/null; do
    if compgen -G "${SAVE_DIR}/step=*.ckpt" > /dev/null; then
      local keep
      keep=$(find "${SAVE_DIR}" -maxdepth 1 -name 'step=*.ckpt' -printf '%f\n' \
        | sed -E 's/^step=([0-9]+)\.ckpt$/\1\t&/' \
        | sort -n | tail -1 | cut -f2)
      if [ -n "${keep}" ]; then
        find "${SAVE_DIR}" -maxdepth 1 -name 'step=*.ckpt' \
          ! -name "${keep}" -print -delete 2>/dev/null \
          | sed 's/^/[midprune] removed /' || true
      fi
    fi
    sleep 60
  done
}
midslot_pruner &
MP_PID=$!

# Watchdog: same as train_ft_beat_phase.sbatch — wandb .wandb file size
# growth as health signal. SIGTERMs python after 30 min of no growth.
WD_GRACE=1800
WD_STALE=1800
watchdog() {
  sleep "${WD_GRACE}"
  local prev_size=0
  local stale_for=0
  while kill -0 "${PY_PID}" 2>/dev/null; do
    local wandb_file
    wandb_file=$(ls -t "${SAVE_DIR}/wandb/latest-run/"*.wandb 2>/dev/null | head -1)
    if [ -n "${wandb_file}" ] && [ -f "${wandb_file}" ]; then
      local cur_size
      cur_size=$(stat -c %s "${wandb_file}" 2>/dev/null || echo 0)
      if (( cur_size > prev_size + 1024 )); then
        prev_size="${cur_size}"
        stale_for=0
      else
        stale_for=$(( stale_for + 60 ))
      fi
      if (( stale_for > WD_STALE )); then
        echo "[watchdog] ${wandb_file} size stagnant for ${stale_for}s (size=${cur_size}); SIGTERM ${PY_PID}"
        kill -TERM "${PY_PID}" 2>/dev/null || true
        return
      fi
    fi
    sleep 60
  done
}
watchdog &
WD_PID=$!

set +e
wait "${PY_PID}"
PY_EXIT=$?
set -e
kill "${WD_PID}" 2>/dev/null || true
wait "${WD_PID}" 2>/dev/null || true
kill "${MP_PID}" 2>/dev/null || true
wait "${MP_PID}" 2>/dev/null || true
echo "[chain] python exited with code ${PY_EXIT}"

rm -rf "${TORCHINDUCTOR_CACHE_DIR}" 2>/dev/null || true

# Prune step=*.ckpt to keep only the highest-step file. base config has
# checkpoint_top_k=-1 so each slot adds ~5-7 ckpts (~6GB each). The full
# Lightning ckpt carries state_dict + optimizer + LR scheduler + AMP scaler
# + global_step, so the highest-step ckpt is sufficient to resume; auto-
# resume (load_from_latest_checkpoint=true, base_trainer.py) picks it by
# mtime, and wandb continuity comes from the wandb/ run dir, not the ckpt.
if compgen -G "${SAVE_DIR}/step=*.ckpt" > /dev/null; then
  KEEP_CKPT="$(find "${SAVE_DIR}" -maxdepth 1 -name 'step=*.ckpt' -printf '%f\n' \
    | sed -E 's/^step=([0-9]+)\.ckpt$/\1\t&/' \
    | sort -n | tail -1 | cut -f2)"
  if [ -n "${KEEP_CKPT}" ]; then
    echo "[prune] keeping ${KEEP_CKPT} as resume source for next slot"
    find "${SAVE_DIR}" -maxdepth 1 -name 'step=*.ckpt' \
      ! -name "${KEEP_CKPT}" -print -delete | sed 's/^/[prune] removed /' || true
  fi
fi

LATEST_STEP=$(find "${SAVE_DIR}" -maxdepth 1 -name 'step=*.ckpt' -printf '%f\n' 2>/dev/null \
  | sed -E 's/^step=([0-9]+)\.ckpt$/\1/' | sort -n | tail -1)
LATEST_STEP="${LATEST_STEP:-0}"

echo "[chain] latest step on disk: ${LATEST_STEP}  (target=${TARGET_STEPS})"
if (( LATEST_STEP < TARGET_STEPS )); then
  echo "[chain] not done yet; self-submitting next slot"
  NEXT_JOB=$(sbatch --parsable \
    --job-name="${SLURM_JOB_NAME}" \
    --export=ALL,FV="${FV}",INIT_CKPT="${INIT_CKPT}",EXP_NAME="${EXP_NAME}",TARGET_STEPS="${TARGET_STEPS}" \
    scripts/slurm/train_dit_beat_phase.sbatch)
  echo "[chain] queued next slot: job ${NEXT_JOB}"
else
  echo "[chain] training complete — no further slots needed"
fi
