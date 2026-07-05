#!/bin/bash
#SBATCH --job-name=smg_dl_slakh
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
#   sbatch --export=ALL,VENV_ACTIVATE=/path/to/.venv/bin/activate scripts/slurm/preprocess_download_slakh2100.sbatch
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

mkdir -p logs

DATASET_ROOT="${DATASET_ROOT:-stream_music_gen_data}"

# Trigger download/extract once up front so chained preprocessing jobs spend time on compute.
python - <<'PY'
import os
from pathlib import Path
from stream_music_gen.dataset.slakh2100 import Slakh2100

root = Path(os.getenv("DATASET_ROOT", "stream_music_gen_data")) / "slakh2100"

for split in ["train", "validation", "test"]:
    ds = Slakh2100(
        root_dir=str(root),
        download=True,
        split=split,
        regenerate_metadata=False,
    )
    print(f"[ok] split={split} files={len(ds)}")
PY
