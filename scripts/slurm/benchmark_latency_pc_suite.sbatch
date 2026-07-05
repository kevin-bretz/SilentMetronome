#!/bin/bash
#SBATCH --job-name=smg_rtf_suite
#SBATCH --partition=gpu-short
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
# exclusive: co-tenant contention inflated latency ~33% (Jun 10 node875 quiet vs Jul 3 node871 busy)
#SBATCH --exclusive
#SBATCH --exclude=node870,node874
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Same-node RTF benchmark suite for the paper's inference-overhead numbers:
#   1. unconditioned baseline          (fv0_k50, 218M)
#   2. conditioned, naive per-step path (fv50 Phase K, 510M)
#   3. conditioned, chunk-ahead modulation precompute
# Running all three inside one allocation removes node-to-node variance.
#
# Optional env vars: BASELINE_CKPT, COND_CKPT, N_* (see benchmark_latency.py),
# DATA_BASE_DIR, OUT_DIR.

PROJECT_DIR="/zfsstore/user/s4483480/JAM/stream-music-gen"
cd "${PROJECT_DIR}"

BASELINE_CKPT="${BASELINE_CKPT:-${PROJECT_DIR}/models/pref_dec_online_fv0_k50/step=200000.ckpt}"
COND_CKPT="${COND_CKPT:-${PROJECT_DIR}/models/pref_dec_online_fv50_k50_beat_phase_dit_mp_cqt_aux_tt_future/step=200000.ckpt}"
N_CHUNKS_PER_SESSION="${N_CHUNKS_PER_SESSION:-10}"
N_SESSIONS="${N_SESSIONS:-30}"
N_WARMUP_SESSIONS="${N_WARMUP_SESSIONS:-3}"
N_DECODE_RUNS="${N_DECODE_RUNS:-100}"
OUT_DIR="${OUT_DIR:-logs/latency/pc_suite_$(date +%Y%m%d)}"

mkdir -p slurm_logs "${OUT_DIR}"
# shellcheck disable=SC1091
source scripts/slurm/activate_env.sh

export PYTHONUNBUFFERED=1

[ -f "${BASELINE_CKPT}" ] || { echo "ERROR: ckpt not found: ${BASELINE_CKPT}"; exit 1; }
[ -f "${COND_CKPT}" ] || { echo "ERROR: ckpt not found: ${COND_CKPT}"; exit 1; }

echo "============================================================"
echo "RTF suite (same node: $(hostname), job ${SLURM_JOB_ID})"
echo "  baseline: ${BASELINE_CKPT}"
echo "  cond:     ${COND_CKPT}"
echo "  out_dir:  ${OUT_DIR}"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# The baseline ckpt's args.yml records the pre-beat-grid dataset dir, which
# no longer exists; the _beat superset works for all models (Jun 10 runs).
DATA_BASE_DIR="${DATA_BASE_DIR:-stream_music_gen_data/precompute_audio_mixdown_20s_beat}"
EXTRA_ARGS="--data_base_dir ${DATA_BASE_DIR}"
echo "[suite] data_base_dir: ${DATA_BASE_DIR}"

run_one () {
  local ckpt="$1" out="$2"
  shift 2
  echo
  echo "------------------------------------------------------------"
  echo "[suite] ${out}  ($*)"
  echo "------------------------------------------------------------"
  python scripts/gen_pred/benchmark_latency.py \
    --model_path "${ckpt}" \
    --n_chunks_per_session "${N_CHUNKS_PER_SESSION}" \
    --n_sessions "${N_SESSIONS}" \
    --n_warmup_sessions "${N_WARMUP_SESSIONS}" \
    --n_decode_runs "${N_DECODE_RUNS}" \
    ${EXTRA_ARGS} "$@" \
    --out_json "${out}"
}

run_one "${BASELINE_CKPT}" "${OUT_DIR}/baseline_fv0_k50.json"
run_one "${COND_CKPT}" "${OUT_DIR}/cond_fv50_naive.json"
run_one "${COND_CKPT}" "${OUT_DIR}/cond_fv50_precompute.json" --dit_precompute

echo
echo "[suite] all three runs done at $(date)"
echo "[suite] warm gen-only RTFs:"
for j in "${OUT_DIR}"/*.json; do
  python - "$j" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  {sys.argv[1]}: rtf_warm_gen_only={d['rtf_warm_gen_only']:.4f} "
      f"(+decode {d['rtf_warm_gen_plus_decode']:.4f}, "
      f"params {d['param_count_M']:.1f}M, pc={d.get('dit_modulation_precompute')})")
EOF
done
