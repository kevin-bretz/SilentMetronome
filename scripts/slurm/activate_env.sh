#!/bin/bash
# Activate the stream-music-gen conda env. Source this from SLURM sbatch
# scripts via `VENV_ACTIVATE=scripts/slurm/activate_env.sh`.
# shellcheck disable=SC1091
source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate stream-music-gen
