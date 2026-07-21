#!/bin/bash
###############################################################################
# stream-music-gen training + evaluation pipeline
#
# Usage:  ./submit.sh           (computes defaults, submits this script via sbatch)
#         sbatch [flags] run_pipeline.sh   (if you prefer manual sbatch flags)
#
# Configure via pipeline.conf rather than editing this script.
###############################################################################
#SBATCH --parsable

set -euo pipefail

###############################################################################
# LOAD CONFIGURATION
# If SMG_* env vars are set (from submit.sh --export), use those.
# Otherwise fall back to pipeline.conf (for manual sbatch or local runs).
###############################################################################
PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"

if [ -n "${SMG_MODEL_TYPE:-}" ]; then
    echo "(Using submit-time configuration from environment)"
    MODEL_TYPE="${SMG_MODEL_TYPE}"
    FUTURE_VISIBILITY="${SMG_FUTURE_VISIBILITY}"
    CHUNK_SIZE="${SMG_CHUNK_SIZE}"
    TRAIN_STEPS="${SMG_TRAIN_STEPS}"
    BATCH_SIZE="${SMG_BATCH_SIZE}"
    CHECKPOINT_INTERVAL="${SMG_CHECKPOINT_INTERVAL}"
    SAMPLE_INTERVAL="${SMG_SAMPLE_INTERVAL}"
    LEARNING_RATE="${SMG_LEARNING_RATE}"
    WARMUP_STEPS="${SMG_WARMUP_STEPS}"
    PRECISION="${SMG_PRECISION}"
    COMPILE="${SMG_COMPILE}"
    NUM_EVAL_SAMPLES="${SMG_NUM_EVAL_SAMPLES}"
    EVAL_BATCH_SIZE="${SMG_EVAL_BATCH_SIZE}"
    EVAL_SPLIT="${SMG_EVAL_SPLIT}"
    SKIP_TRAINING="${SMG_SKIP_TRAINING}"
    EVAL_CKPT_PATH="${SMG_EVAL_CKPT_PATH:-}"
    EXP_NAME="${SMG_EXP_NAME}"
    PARTITION="${SMG_PARTITION}"
    NUM_GPUS="${SMG_NUM_GPUS}"
else
    CONF="${PROJECT_DIR}/pipeline.conf"
    if [ ! -f "${CONF}" ]; then
        echo "ERROR: ${CONF} not found and no SMG_* env vars set."
        exit 1
    fi
    echo "(Using pipeline.conf. Values are read at job start, not submit time)"
    source "${CONF}"
fi

###############################################################################
# AUTO-SET DEFAULTS (same logic as submit.sh, in case of manual sbatch)
###############################################################################
if [ "${MODEL_TYPE}" = "prefix_decoder_online" ]; then
    : "${TRAIN_STEPS:=200000}"
    : "${BATCH_SIZE:=$(( 16 / NUM_GPUS ))}"
    : "${CHECKPOINT_INTERVAL:=20000}"
    : "${SAMPLE_INTERVAL:=20000}"
fi

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

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
SAVE_DIR="${PROJECT_DIR}/logs/${EXP_NAME}"

###############################################################################
# ENVIRONMENT
###############################################################################
echo "============================================================"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname): ${EXP_NAME}"
echo "GPUs: ${NUM_GPUS}  Partition: ${PARTITION}"
echo "Started: $(date)"
echo "============================================================"
echo ""
echo "Resolved hyperparameters:"
echo "  MODEL_TYPE=${MODEL_TYPE}  FUTURE_VISIBILITY=${FUTURE_VISIBILITY}"
[ "${MODEL_TYPE}" = "prefix_decoder_online" ] && echo "  CHUNK_SIZE=${CHUNK_SIZE}"
echo "  TRAIN_STEPS=${TRAIN_STEPS}  BATCH_SIZE=${BATCH_SIZE}/GPU (×${NUM_GPUS}=$(( BATCH_SIZE * NUM_GPUS )) eff.)"
echo "  CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}  SAMPLE_INTERVAL=${SAMPLE_INTERVAL}"
echo "  LR=${LEARNING_RATE}  WARMUP=${WARMUP_STEPS}  PRECISION=${PRECISION}"
echo ""

source "${PROJECT_DIR}/scripts/slurm/activate_env.sh"

cd "${PROJECT_DIR}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

###############################################################################
# STEP 1: RESOLVE MODEL TYPE TO SCRIPTS / CONFIGS
###############################################################################

if [ "${MODEL_TYPE}" = "prefix_decoder_online" ]; then
    TRAIN_SCRIPT="scripts/train_prefix_dec_online.py"
    BASE_CONFIG="configs/online/online_prefix_decoder/online_prefix_decoder_base.yml"
    MODEL_CLASS="OnlinePrefixDecoderTransformerMultiOut"
    EVAL_MODEL_TYPE="prefix_decoder_online"
else
    echo "ERROR: Unknown MODEL_TYPE '${MODEL_TYPE}'. Use 'prefix_decoder_online'."
    exit 1
fi

###############################################################################
# STEP 2: TRAINING (skipped when SKIP_TRAINING=true)
###############################################################################

if [ "${SKIP_TRAINING}" = "true" ]; then
    echo "============================================================"
    echo "SKIPPING TRAINING (using pre-selected checkpoint)"
    echo "============================================================"

    if [[ "${EVAL_CKPT_PATH}" != /* ]]; then
        CKPT_PATH="${PROJECT_DIR}/${EVAL_CKPT_PATH}"
    else
        CKPT_PATH="${EVAL_CKPT_PATH}"
    fi

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "ERROR: Checkpoint not found: ${CKPT_PATH}"
        exit 1
    fi

    echo "Checkpoint: ${CKPT_PATH}"
else
    echo "============================================================"
    echo "TRAINING  (${MODEL_TYPE}, fv=${FUTURE_VISIBILITY}, steps=${TRAIN_STEPS})"
    echo "============================================================"

    # Use find_unused_parameters for DDP when some params may not contribute
    # to loss (e.g. input_delay_emb is unused when future_visibility <= 0)
    if [ "${NUM_GPUS}" -gt 1 ]; then
        STRATEGY="ddp_find_unused_parameters_true"
    else
        STRATEGY="auto"
    fi

    TRAIN_CMD=(
        python "${TRAIN_SCRIPT}"
        --args.load "${BASE_CONFIG}"
        --save_dir "${SAVE_DIR}"
        --train_steps "${TRAIN_STEPS}"
        --batch_size "${BATCH_SIZE}"
        --checkpoint_interval "${CHECKPOINT_INTERVAL}"
        --sample_interval "${SAMPLE_INTERVAL}"
        --strategy "${STRATEGY}"
        --"${MODEL_CLASS}.future_visibility" "${FUTURE_VISIBILITY}"
        --"${MODEL_CLASS}.cond_method" "add"
        --AdamW.lr "${LEARNING_RATE}"
        --LinearWarmupCosineDecay.warmup_iters "${WARMUP_STEPS}"
        --LinearWarmupCosineDecay.total_iters "${TRAIN_STEPS}"
        --precision "${PRECISION}"
        --compile "${COMPILE}"
    )

    if [ "${MODEL_TYPE}" = "prefix_decoder_online" ]; then
        TRAIN_CMD+=(--"${MODEL_CLASS}.chunk_length" "${CHUNK_SIZE}")
    fi

    echo "Command: ${TRAIN_CMD[*]}"
    echo ""
    TRAIN_START=$(date +%s)
    "${TRAIN_CMD[@]}"
    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$(( TRAIN_END - TRAIN_START ))
    TRAIN_HOURS=$(( TRAIN_DURATION / 3600 ))
    TRAIN_MINS=$(( (TRAIN_DURATION % 3600) / 60 ))
    echo ""
    echo "Training complete: $(date)"
    echo "Training duration: ${TRAIN_HOURS}h ${TRAIN_MINS}m (${TRAIN_DURATION}s total, ${TRAIN_STEPS} steps)"
    echo "Throughput: ~$(( TRAIN_STEPS / (TRAIN_DURATION > 0 ? TRAIN_DURATION : 1) )) steps/s, ~$(( TRAIN_DURATION / (TRAIN_STEPS > 0 ? TRAIN_STEPS : 1) * 1000 ))ms/step"

    echo "============================================================"
    echo "LOCATING CHECKPOINT"
    echo "============================================================"

    CKPT_PATH=$(find "${SAVE_DIR}" -name "*.ckpt" -printf '%T@ %p\n' | sort -n | tail -1 | awk '{print $2}')

    if [ -z "${CKPT_PATH}" ]; then
        echo "ERROR: No checkpoint found in ${SAVE_DIR}"
        exit 1
    fi

    echo "Using checkpoint: ${CKPT_PATH}"
fi

###############################################################################
# STEP 3: EVALUATION
###############################################################################
echo "============================================================"
echo "EVALUATION  (${NUM_EVAL_SAMPLES} samples, split=${EVAL_SPLIT})"
echo "============================================================"

RESULTS_DIR="${PROJECT_DIR}/logs/eval_results"
EVAL_SAVE_NAME="${EXP_NAME}_eval"

EVAL_CMD=(
    python scripts/gen_pred/gen_and_evaluate.py
    --model_type "${EVAL_MODEL_TYPE}"
    --model_path "${CKPT_PATH}"
    --batch_size "${EVAL_BATCH_SIZE}"
    --num_samples "${NUM_EVAL_SAMPLES}"
    --split "${EVAL_SPLIT}"
    --results_save_dir "${RESULTS_DIR}"
    --save_dir_name "${EVAL_SAVE_NAME}"
)

echo "Command: ${EVAL_CMD[*]}"
echo ""
"${EVAL_CMD[@]}"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Finished: $(date)"
echo "Training logs:     ${SAVE_DIR}/"
echo "Checkpoint:        ${CKPT_PATH}"
echo "Eval results:      ${RESULTS_DIR}/${EVAL_SAVE_NAME}.json"
echo "SLURM logs:        ${SLURM_LOG_DIR}/${EXP_NAME}_${SLURM_JOB_ID}.{out,err}"
