#!/bin/bash
# ML_gene_prediction_analysis.sh —

# ------- LSF resources ------
#BSUB -J ML_gene_prediction_analysis
#BSUB -P acc_DiseaseGeneCell
#BSUB -q premium
#BSUB -n 1
#BSUB -R "rusage[mem=128G]"
#BSUB -W 6:00
#BSUB -o logs/ML_gene_prediction_analysis.%J.out
#BSUB -e logs/ML_gene_prediction_analysis.%J.err


# --------------------------------

set -Eeuo pipefail
trap 'ec=$?; echo "[ERROR] line ${LINENO} status ${ec}" >&2' ERR

# --- Make sure a logs dir exists in the SUBMISSION directory ---
mkdir -p ./logs

# --- Resolve repo root to the SUBMISSION directory, not the script folder ---
SUBMIT_DIR="$(pwd)"     # because -cwd is set by LSF to the submission dir (or %J_workdir)
echo "Submit dir: ${SUBMIT_DIR}"

# ---- Logging (mirror to ./logs) ----
ts="$(date +"%Y%m%d_%H%M%S")"
log_file="./logs/${ts}_ML_gene_benchmark_prediction_analysis.log"

echo "------------------------------------------------------------"
echo "JOB START: $(date)"
echo "JOBID     : ${LSB_JOBID:-local}  IDX=${LSB_JOBINDEX:-}"
echo "HOST      : $(hostname)"
echo "PWD       : $(pwd)"
echo "LOG FILE  : ${log_file}"
echo "------------------------------------------------------------"

# ---- Modules / shell setup ----
module purge || true
module load anaconda3/latest || true
module load cuda/12.4.0 || true

# ---- Conda bootstrap ----
if ! base_dir="$(conda info --base 2>/dev/null)"; then
  base_dir="$HOME/miniconda3"
fi
# shellcheck disable=SC1091
source "${base_dir}/etc/profile.d/conda.sh"

# ---- Paths / env ----
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/dti"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"

export PIP_CACHE_DIR CONDA_PKGS_DIRS PYTHONNOUSERSITE=1 TERM=xterm PYTHONUNBUFFERED=1
unset PYTHONPATH || true

echo "Activating conda env: ${ENV_PREFIX}"
conda activate "${ENV_PREFIX}" || { echo "[ERROR] conda activate failed"; exit 1; }
PYTHON="${ENV_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="python"

# ---- Project paths ----
LOG_LEVEL="INFO"
DATA_FN="output/metrics/20251031_all_binding_db_genes.parquet"
OUTPUT_DIR="output/metrics"; mkdir -p "${OUTPUT_DIR}"
PREFIX="All_BindingDB_prediction_analysis"
MAIN="src/ML_gene_prediction_analysis.py"

[[ -f "${MAIN}" ]] || { echo "[ERROR] MAIN not found: ${MAIN} (PWD=$(pwd))"; exit 2; }

TOP_K=0
MIN_N=0
N=0

echo "Python     : $(command -v "${PYTHON}")"
echo "Main script: ${MAIN}"
echo "Data file  : ${DATA_FN}"
echo "Output dir : ${OUTPUT_DIR}"
echo "Prefix     : ${PREFIX}"
echo "Top K      : ${TOP_K}"
echo "Min N      : ${MIN_N}"
echo "N rows     : ${N}"
echo "------------------------------------------------------------"

set +e
"${PYTHON}" "${MAIN}" \
  --log_fn "${log_file}" \
  --log_level "${LOG_LEVEL}" \
  --data_fn "${DATA_FN}" \
  --top_k "${TOP_K}" \
  --output_dir "${OUTPUT_DIR}" \
  --prefix "${PREFIX}" \
  --N "${N}" \
  --min_n "${MIN_N}"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "[OK] finished at $(date)"
else
  echo "[ERROR] exit code ${exit_code} at $(date)"
  exit ${exit_code}
fi

echo "JOB END: $(date)"
