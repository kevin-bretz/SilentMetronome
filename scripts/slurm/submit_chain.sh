#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <sbatch_script> <num_runs> [extra sbatch args ...]"
  echo "Example: $0 scripts/slurm/preprocess_extract_causal_dac_32k.sbatch 6 --export=ALL,VENV_ACTIVATE=/path/.venv/bin/activate"
  exit 1
fi

SBATCH_SCRIPT="$1"
NUM_RUNS="$2"
shift 2

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "Error: script not found: ${SBATCH_SCRIPT}"
  exit 1
fi

if ! [[ "${NUM_RUNS}" =~ ^[0-9]+$ ]] || [[ "${NUM_RUNS}" -lt 1 ]]; then
  echo "Error: num_runs must be a positive integer"
  exit 1
fi

prev_jobid=""
for ((i=1; i<=NUM_RUNS; i++)); do
  if [[ -n "${prev_jobid}" ]]; then
    out="$(sbatch --dependency=afterany:${prev_jobid} "$@" "${SBATCH_SCRIPT}")"
  else
    out="$(sbatch "$@" "${SBATCH_SCRIPT}")"
  fi

  jobid="$(awk '{print $4}' <<< "${out}")"
  echo "[chain ${i}/${NUM_RUNS}] ${out}"
  prev_jobid="${jobid}"
done

echo "[done] Last queued job id: ${prev_jobid}"
