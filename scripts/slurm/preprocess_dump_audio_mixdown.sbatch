#!/bin/bash
#SBATCH --job-name=smg_dump_mix
#SBATCH --partition=gpu-short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Optional: export VENV_ACTIVATE before sbatch, e.g.
#   sbatch --export=ALL,VENV_ACTIVATE=/path/to/.venv/bin/activate,SPLIT=train scripts/slurm/preprocess_dump_audio_mixdown.sbatch
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

mkdir -p logs

DATASET="${DATASET:-slakh2100}"
SPLIT="${SPLIT:-train}"               # train | valid | test
MAX_EXAMPLES="${MAX_EXAMPLES:-1000000}" # Use 10000 for valid/test
OUTPUT_DIR="${OUTPUT_DIR:-stream_music_gen_data/precompute_audio_mixdown_20s}"
AUDIO_DURATION="${AUDIO_DURATION:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_BASE_DIR="${DATA_BASE_DIR:-stream_music_gen_data/causal_dac_codes_32khz}"
RMS_BASE_DIR="${RMS_BASE_DIR:-stream_music_gen_data/rms_50hz}"
AUDIO_BASE_DIR="${AUDIO_BASE_DIR:-stream_music_gen_data}"
SAVE_AUDIO="${SAVE_AUDIO:-0}"          # Set to 1 for valid/test if desired

SPLIT_DIR="${OUTPUT_DIR}/${DATASET}/${SPLIT}"
mkdir -p "${SPLIT_DIR}"

# When submitted as a SLURM job array, each task processes a disjoint slice of
# the [0, MAX_EXAMPLES) output range so tasks can run in parallel without
# racing into the same output directory. Within a slice, the task resumes by
# counting the already-written subdirs that fall inside its own range.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  PER_TASK=$(( (MAX_EXAMPLES + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
  TASK_START=$(( (SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN) * PER_TASK ))
  TASK_END=$(( TASK_START + PER_TASK ))
  if (( TASK_END > MAX_EXAMPLES )); then
    TASK_END=$MAX_EXAMPLES
  fi
  # Count completed subdirs already within [TASK_START, TASK_END)
  COMPLETED_IN_RANGE="$(find "${SPLIT_DIR}" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' \
    | awk -v s="${TASK_START}" -v e="${TASK_END}" '{v=$1+0; if (v>=s && v<e) c++} END {print c+0}')"
  START_INDEX=$(( TASK_START + COMPLETED_IN_RANGE ))
  EFFECTIVE_MAX=$TASK_END
  echo "[info] array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} "
  echo "[info]   range=[${TASK_START}, ${TASK_END}) per_task=${PER_TASK}"
  echo "[info]   completed_in_range=${COMPLETED_IN_RANGE} start_index=${START_INDEX}"
else
  # Non-array (single-job) mode: resume from global dir count.
  START_INDEX="$(find "${SPLIT_DIR}" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]' | wc -l)"
  EFFECTIVE_MAX=$MAX_EXAMPLES
fi

echo "[info] split=${SPLIT} start_index=${START_INDEX} effective_max=${EFFECTIVE_MAX}"
if (( START_INDEX >= EFFECTIVE_MAX )); then
  echo "[info] Nothing to do: slice already complete."
  exit 0
fi

EXTRA_ARGS=()
if [[ "${SAVE_AUDIO}" == "1" ]]; then
  EXTRA_ARGS+=(--save_audio)
fi

# Single-process extraction does not benefit from requesting multiple GPUs.
python stream_music_gen/dataset/dump_audio_mixdown.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --max_examples "${EFFECTIVE_MAX}" \
  --start_index "${START_INDEX}" \
  --output_dir "${OUTPUT_DIR}" \
  --audio_duration "${AUDIO_DURATION}" \
  --num_workers "${NUM_WORKERS}" \
  --data_base_dir "${DATA_BASE_DIR}" \
  --rms_base_dir "${RMS_BASE_DIR}" \
  --audio_base_dir "${AUDIO_BASE_DIR}" \
  "${EXTRA_ARGS[@]}"
