#!/bin/bash
set -euo pipefail

# --- Modules / shell setup ---
module purge
module load anaconda3/latest
module load cuda/12.4.0    

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Paths (edit if needed) ---
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/psichic"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"  # conda cache (optional but recommended)

# --- Caches & hygiene ---
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# --- Activate env (must exist already) ---
conda activate "${ENV_PREFIX}"
