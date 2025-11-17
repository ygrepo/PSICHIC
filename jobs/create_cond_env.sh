#!/bin/bash
set -eo pipefail

# --- Modules / shell setup ---
module purge
module load anaconda3/latest
# For GPU: only load CUDA module if your cluster requires it.
# pytorch-cuda=12.1 already ships its own CUDA runtime.
# module load cuda/12.1

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Paths (edit if needed) ---
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/psichic"
ENV_YML="config/environment_gpu.yml"   

PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"

# Keep installs off $HOME and avoid user-site leakage
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR
export CONDA_PKGS_DIRS
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# Create env if missing
if [ ! -d "${ENV_PREFIX}" ] || [ ! -f "${ENV_PREFIX}/conda-meta/history" ]; then
  echo "Creating conda env at ${ENV_PREFIX} from ${ENV_YML}..."
  conda env create --prefix "${ENV_PREFIX}" --file "${ENV_YML}"
else
  echo "Using existing env at ${ENV_PREFIX}"
fi

# Activate env
conda activate "${ENV_PREFIX}"
