#!/bin/bash
#SBATCH --job-name=smg_phase_k_fvm50
#SBATCH --partition=gpu-short
#SBATCH --time=04:00:00
#SBATCH --signal=B:SIGTERM@180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --exclude=node870,node874
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# PHASE K @ fv=-50: same Phase K target-token future-aux head (K=3,
# [10,25,40]) at the hardest deployability setting — encoder sees inputs
# only up to position p-50 (lagging by 50 frames). Phase K must
# extrapolate ~|fv|+δ frames forward; this is the most adverse fv span
# for the mechanism.
#
# Self-chains until TARGET_STEPS reached, then queues eval (cond-on default).

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"
mkdir -p slurm_logs
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

CONFIG="configs/online/online_prefix_decoder/online_prefix_decoder_fv-50_k50_beat_phase_dit_mp_cqt_aux_tt_future.yml"
INIT_CKPT="${INIT_CKPT:-}"
EXP_NAME="${EXP_NAME:-pref_dec_online_fv-50_k50_beat_phase_dit_mp_cqt_aux_tt_future}"
SAVE_DIR="${PROJECT_DIR}/models/${EXP_NAME}"
TARGET_STEPS="${TARGET_STEPS:-200000}"
CHAIN_NEXT="${CHAIN_NEXT:-1}"

mkdir -p "${SAVE_DIR}"
LATEST_BEFORE=$(find "${SAVE_DIR}" -maxdepth 1 -name 'step=*.ckpt' -printf '%f\n' 2>/dev/null \
  | sed -E 's/^step=([0-9]+)\.ckpt$/\1/' | sort -n | tail -1 || true)
LATEST_BEFORE="${LATEST_BEFORE:-0}"

CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"

SLOT_TARGET=$(( LATEST_BEFORE + CHECKPOINT_INTERVAL ))
if (( SLOT_TARGET > TARGET_STEPS )); then SLOT_TARGET=${TARGET_STEPS}; fi

echo "============================================================"
echo "PHASE K @ fv=-50 — Phase K (K=3) at the most-adverse causal setting"
echo "  config:      ${CONFIG}"
echo "  init_ckpt:   ${INIT_CKPT:-<none: training from scratch>}"
echo "  save_dir:    ${SAVE_DIR}"
echo "  target:      ${TARGET_STEPS} steps  (this slot stops at ${SLOT_TARGET})"
echo "  ckpt_iv:     ${CHECKPOINT_INTERVAL}  (latest_before=${LATEST_BEFORE})"
echo "  node:        $(hostname)    job: ${SLURM_JOB_ID}"
echo "  started:     $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

[ -f "${CONFIG}" ] || { echo "ERROR: config not found: ${CONFIG}"; exit 1; }
if [ -n "${INIT_CKPT}" ] && [ ! -f "${INIT_CKPT}" ]; then
  echo "ERROR: INIT_CKPT set but not found: ${INIT_CKPT}"; exit 1
fi
[ -d "${PROJECT_DIR}/stream_music_gen_data/precompute_audio_mixdown_20s_beat/slakh2100/train" ] \
  || { echo "ERROR: beat-phase data root missing"; exit 1; }

export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/${USER}_inductor_${SLURM_JOB_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
echo "[chain] TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
export PYTHONUNBUFFERED=1

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
  --train_steps "${SLOT_TARGET}"
  --checkpoint_interval "${CHECKPOINT_INTERVAL}"
)
if [ -n "${INIT_CKPT}" ]; then
  TRAIN_CMD+=(--init_from_checkpoint "${INIT_CKPT}")
fi
"${TRAIN_CMD[@]}" &
PY_PID=$!

WD_GRACE=1800
WD_STALE=5400
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
        echo "[watchdog] ${wandb_file} stagnant for ${stale_for}s; SIGTERM ${PY_PID}"
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
echo "[chain] python exited with code ${PY_EXIT}"
rm -rf "${TORCHINDUCTOR_CACHE_DIR}" 2>/dev/null || true

LATEST_STEP=$(find "${SAVE_DIR}" -maxdepth 1 -name 'step=*.ckpt' -printf '%f\n' 2>/dev/null \
  | sed -E 's/^step=([0-9]+)\.ckpt$/\1/' | sort -n | tail -1)
LATEST_STEP="${LATEST_STEP:-0}"
echo "[chain] latest step on disk: ${LATEST_STEP}  (target=${TARGET_STEPS})"

# Auto-prune intermediate ckpts to cap storage growth during long chained
# trainings. Keeps only step=${LATEST_STEP}.ckpt (next-slot resume target).
# Skipped when LATEST_STEP==0 (nothing trained) or when LATEST_STEP <=
# LATEST_BEFORE (no progress: leave files alone for manual debug).
if (( LATEST_STEP > LATEST_BEFORE )); then
  pruned=0
  for f in "${SAVE_DIR}"/step=*.ckpt; do
    [ -f "$f" ] || continue
    [[ "$f" == *"step=${LATEST_STEP}.ckpt" ]] && continue
    rm -f "$f"
    pruned=$(( pruned + 1 ))
  done
  echo "[chain] pruned ${pruned} intermediate ckpt(s); kept step=${LATEST_STEP}.ckpt"
fi

if (( LATEST_STEP <= LATEST_BEFORE )); then
  echo "[chain] no progress this slot (latest=${LATEST_STEP} <= before=${LATEST_BEFORE}); NOT chaining"
elif (( LATEST_STEP < TARGET_STEPS )); then
  echo "[chain] not done yet; self-submitting next slot"
  NEXT_JOB=$(sbatch --parsable \
    --job-name="${SLURM_JOB_NAME}" \
    --exclude=node870,node874 \
    --export=ALL,INIT_CKPT="${INIT_CKPT}",EXP_NAME="${EXP_NAME}",TARGET_STEPS="${TARGET_STEPS}",CHAIN_NEXT="${CHAIN_NEXT}" \
    scripts/slurm/train_phase_k_fv-50.sbatch)
  echo "[chain] queued next slot: job ${NEXT_JOB}"
else
  echo "[chain] PHASE K fv=-50 training complete"
  if [ "${CHAIN_NEXT}" = "1" ]; then
    CKPT_PATH="${SAVE_DIR}/step=${TARGET_STEPS}.ckpt"
    EVAL_J=$(sbatch --parsable \
      --exclude=node870,node874 \
      --export=ALL,MODEL_PATH="${CKPT_PATH}",SPLIT=test \
      scripts/slurm/eval_beat_phase.sbatch)
    echo "[chain] PHASE K fv=-50 eval (cond-on) job: ${EVAL_J}"
  fi
fi
