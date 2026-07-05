#!/bin/bash
# Orchestrator: launches the Phase A (chroma aux) → Phase B (chroma aux +
# cond) sequence on gpu-a100-80g. Each train sbatch is self-chaining, so
# Phase A submits Phase B + Phase A eval automatically when it reaches
# TARGET_STEPS. Phase B submits Phase B eval when done.
#
# Usage:
#   ./scripts/slurm/run_chroma_sequence.sh           # default FV=0
#   FV=0 ./scripts/slurm/run_chroma_sequence.sh
#
# To skip auto-chaining (run only Phase A, no Phase B / no eval):
#   CHAIN_NEXT=0 ./scripts/slurm/run_chroma_sequence.sh

set -euo pipefail

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

FV="${FV:-0}"
CHAIN_NEXT="${CHAIN_NEXT:-1}"

case "${FV}" in
  -50|0|50) : ;;
  *) echo "ERROR: FV='${FV}' must be one of -50, 0, 50"; exit 1 ;;
esac

# Sanity: chroma data must be ready (target_chroma + input_chroma in train).
# Use a fixed known window rather than `ls -d */` over a 1M-entry dir, which
# takes minutes/hangs in D-state.
TRAIN_DIR="${PROJECT_DIR}/stream_music_gen_data/precompute_audio_mixdown_20s_beat/slakh2100/train"
SAMPLE_WIN="${TRAIN_DIR}/0000999"
if [ ! -d "${SAMPLE_WIN}" ]; then
  echo "ERROR: sample window ${SAMPLE_WIN} missing"; exit 1
fi
[ -f "${SAMPLE_WIN}/target_chroma.pt" ] || \
  echo "WARNING: target_chroma.pt not found in sample window — chroma extraction may not be complete yet"
[ -f "${SAMPLE_WIN}/input_chroma.pt" ] || \
  echo "WARNING: input_chroma.pt not found in sample window — chroma extraction may not be complete yet"

echo "============================================================"
echo "Launching chroma training sequence: FV=${FV} chain_next=${CHAIN_NEXT}"
echo "============================================================"
JOB_A=$(sbatch --parsable \
  --export=ALL,FV="${FV}",CHAIN_NEXT="${CHAIN_NEXT}" \
  scripts/slurm/train_chroma_aux.sbatch)
echo "Submitted PHASE A train: job ${JOB_A}"
echo
echo "Phase A self-chains until step=200000, then auto-submits:"
echo "  - PHASE A eval on step=200000 ckpt"
echo "  - PHASE B train (which self-chains and auto-submits PHASE B eval)"
echo
echo "Watch with: squeue -u \$USER | grep -E 'smg_chroma|smg_eval'"
