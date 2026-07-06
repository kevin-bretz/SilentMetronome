#!/bin/bash
###############################################################################
# Reads pipeline.conf, computes defaults, and submits run_pipeline.sh.
#
# Usage:  ./submit.sh
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${SCRIPT_DIR}/pipeline.conf"

if [ ! -f "${CONF}" ]; then
    echo "ERROR: ${CONF} not found."
    exit 1
fi
source "${CONF}"

# Auto-set defaults per model type
if [ "${MODEL_TYPE}" = "dec_online" ]; then
    : "${TRAIN_STEPS:=100000}"
    : "${BATCH_SIZE:=$(( 64 / NUM_GPUS ))}"
    : "${CHECKPOINT_INTERVAL:=10000}"
    : "${SAMPLE_INTERVAL:=10000}"
elif [ "${MODEL_TYPE}" = "prefix_decoder_online" ]; then
    : "${TRAIN_STEPS:=200000}"
    : "${BATCH_SIZE:=$(( 16 / NUM_GPUS ))}"
    : "${CHECKPOINT_INTERVAL:=20000}"
    : "${SAMPLE_INTERVAL:=20000}"
else
    echo "ERROR: Unknown MODEL_TYPE '${MODEL_TYPE}'."
    exit 1
fi

# Auto-generate experiment name
if [ -z "${EXP_NAME}" ]; then
    if [ "${SKIP_TRAINING}" = "true" ]; then
        CKPT_BASENAME=$(basename "$(dirname "${EVAL_CKPT_PATH}")")
        EXP_NAME="eval_${CKPT_BASENAME}"
    elif [ "${MODEL_TYPE}" = "prefix_decoder_online" ]; then
        EXP_NAME="${MODEL_TYPE}_fv${FUTURE_VISIBILITY}_k${CHUNK_SIZE}_steps${TRAIN_STEPS}_bs${BATCH_SIZE}"
    else
        EXP_NAME="${MODEL_TYPE}_fv${FUTURE_VISIBILITY}_steps${TRAIN_STEPS}_bs${BATCH_SIZE}"
    fi
fi

SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"
mkdir -p "${SLURM_LOG_DIR}"

echo "============================================================"
echo "Submitting: ${EXP_NAME}"
echo "  MODEL_TYPE=${MODEL_TYPE}  FV=${FUTURE_VISIBILITY}"
[ "${MODEL_TYPE}" = "prefix_decoder_online" ] && echo "  CHUNK_SIZE=${CHUNK_SIZE}"
echo "  TRAIN_STEPS=${TRAIN_STEPS}  BATCH_SIZE=${BATCH_SIZE}/GPU (×${NUM_GPUS}=$(( BATCH_SIZE * NUM_GPUS )) eff.)"
echo "  CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}  SAMPLE_INTERVAL=${SAMPLE_INTERVAL}"
echo "  LR=${LEARNING_RATE}  WARMUP=${WARMUP_STEPS}  PRECISION=${PRECISION}"
echo "  SKIP_TRAINING=${SKIP_TRAINING}"
[ "${SKIP_TRAINING}" = "true" ] && echo "  EVAL_CKPT_PATH=${EVAL_CKPT_PATH}"
echo "  PARTITION=${PARTITION}  GPUs=${NUM_GPUS}  GPU_TYPE=${GPU_TYPE:-any}  TIME=${TIME_LIMIT}"
echo "  COMPILE=${COMPILE}"
echo "============================================================"

# Build gres string: "gpu:TYPE:N" if GPU_TYPE is set, otherwise "gpu:N"
if [ -n "${GPU_TYPE:-}" ]; then
    GRES="gpu:${GPU_TYPE}:${NUM_GPUS}"
else
    GRES="gpu:${NUM_GPUS}"
fi

# Export all config as env vars so run_pipeline.sh uses submit-time values
# (not whatever pipeline.conf contains when the job eventually starts)
JOB_ID=$(sbatch \
    --job-name="smg-${EXP_NAME}" \
    --partition="${PARTITION}" \
    --gres="${GRES}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${TIME_LIMIT}" \
    --output="${SLURM_LOG_DIR}/${EXP_NAME}_%j.out" \
    --error="${SLURM_LOG_DIR}/${EXP_NAME}_%j.err" \
    --export="ALL,\
SMG_MODEL_TYPE=${MODEL_TYPE},\
SMG_FUTURE_VISIBILITY=${FUTURE_VISIBILITY},\
SMG_CHUNK_SIZE=${CHUNK_SIZE:-50},\
SMG_TRAIN_STEPS=${TRAIN_STEPS},\
SMG_BATCH_SIZE=${BATCH_SIZE},\
SMG_CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL},\
SMG_SAMPLE_INTERVAL=${SAMPLE_INTERVAL},\
SMG_LEARNING_RATE=${LEARNING_RATE},\
SMG_WARMUP_STEPS=${WARMUP_STEPS},\
SMG_PRECISION=${PRECISION},\
SMG_COMPILE=${COMPILE},\
SMG_NUM_EVAL_SAMPLES=${NUM_EVAL_SAMPLES},\
SMG_EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},\
SMG_EVAL_SPLIT=${EVAL_SPLIT},\
SMG_SKIP_TRAINING=${SKIP_TRAINING},\
SMG_EVAL_CKPT_PATH=${EVAL_CKPT_PATH:-},\
SMG_EXP_NAME=${EXP_NAME},\
SMG_PARTITION=${PARTITION},\
SMG_NUM_GPUS=${NUM_GPUS}" \
    "${SCRIPT_DIR}/run_pipeline.sh")

echo ""
echo "Submitted job ${JOB_ID}"
echo "  stdout: ${SLURM_LOG_DIR}/${EXP_NAME}_${JOB_ID}.out"
echo "  stderr: ${SLURM_LOG_DIR}/${EXP_NAME}_${JOB_ID}.err"
