#!/bin/bash
# train.sh

# ------- LSF resources ------
#BSUB -J train
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu                  # queue
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=128G]"
#BSUB -W 6:00
#BSUB -o logs/train.%J.out
#BSUB -e logs/train.%J.err


# --------------------------------

set -Eeo pipefail
trap 'ec=$?; echo "[ERROR] line ${LINENO} status ${ec}" >&2' ERR

# --- Make sure a logs dir exists in the SUBMISSION directory ---
mkdir -p ./logs

# --- Resolve repo root to the SUBMISSION directory, not the script folder ---
SUBMIT_DIR="$(pwd)"     # because -cwd is set by LSF to the submission dir (or %J_workdir)
echo "Submit dir: ${SUBMIT_DIR}"

# ---- Logging (mirror to ./logs) ----
LOG_LEVEL="INFO"

mkdir -p ./logs
ts="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="./logs/${ts}_train.log"

echo "------------------------------------------------------------"
echo "JOB START: $(date)"
echo "JOBID     : ${LSB_JOBID:-local}  IDX=${LSB_JOBINDEX:-}"
echo "HOST      : $(hostname)"
echo "PWD       : $(pwd)"
echo "LOG FILE  : ${LOG_FILE}"
echo "------------------------------------------------------------"

# ---- Modules / shell setup ----
module purge || true
module load anaconda3/latest || true
module load cuda/12.4.0 || true
ml proxies/1 || true

# ---- Conda bootstrap ----
if ! base_dir="$(conda info --base 2>/dev/null)"; then
  base_dir="$HOME/miniconda3"
fi
# shellcheck disable=SC1091
source "${base_dir}/etc/profile.d/conda.sh"

# ---- Paths / env ----
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/psichic"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"

export PIP_CACHE_DIR CONDA_PKGS_DIRS PYTHONNOUSERSITE=1 TERM=xterm PYTHONUNBUFFERED=1
unset PYTHONPATH || true

echo "Activating conda env: ${ENV_PREFIX}"
conda activate "${ENV_PREFIX}" || { echo "[ERROR] conda activate failed"; exit 1; }
PYTHON="${ENV_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="python"

export HF_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models"
export HF_TOKEN_PATH="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/tokens/hf_token.csv"

# ---- Project paths ----
CONFIG_PATH="config"
RESULT_PATH="./output/models/PSICHIC/results/PDB2020_BENCHMARK/"
mkdir -p "${RESULT_PATH}"
MODEL_PATH="${RESULT_PATH}/save_models"
mkdir -p "${MODEL_PATH}"
INTERPRET_PATH="${RESULT_PATH}/interpretation_result"
REGRESSION_TASK=True
CLASSIFICATION_TASK=False
MCLASSIFICATION_TASK=0
EPOCHS=30
EVALUATE_EPOCH=1
TOTAL_ITERS=30000
EVALUATE_STEP=500
LRATE=1e-4
EPS=1e-8
BETAS="(0.9,0.999)"
BATCH_SIZE=16
N=10

DATAFOLDER="./dataset/pdb2020"
TRAINED_MODEL_PATH=""

MAIN="src/train.py"

[[ -f "${MAIN}" ]] || { echo "[ERROR] MAIN not found: ${MAIN} (PWD=$(pwd))"; exit 2; }

echo "Python     : $(command -v "${PYTHON}")"
echo "Main script: ${MAIN}"
echo "Result path: ${RESULT_PATH}"
echo "Model path: ${MODEL_PATH}"
echo "Interpret path: ${INTERPRET_PATH}"
echo "Regression task: ${REGRESSION_TASK}"
echo "Classification task: ${CLASSIFICATION_TASK}"
echo "Multiclassification task: ${MCLASSIFICATION_TASK}"
echo "EPOCHS: ${EPOCHS}"
echo "N: ${N}"
echo "Evaluate epoch: ${EVALUATE_EPOCH}"
echo "Total iters: ${TOTAL_ITERS}"
echo "Evaluate step: ${EVALUATE_STEP}"
echo "LRATE: ${LRATE}"
echo "EPS: ${EPS}"
echo "BETAS: ${BETAS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Datafolder: ${DATAFOLDER}"
echo "Trained model path: ${TRAINED_MODEL_PATH}"
echo "------------------------------------------------------------"

set +e
"${PYTHON}" "${MAIN}" \
  --log_fn "${LOG_FILE}" \
  --log_level "${LOG_LEVEL}" \
  --config_path "${CONFIG_PATH}" \
  --result_path "${RESULT_PATH}" \
  --model_path "${MODEL_PATH}" \
  --interpret_path "${INTERPRET_PATH}" \
  --save_interpret "${SAVE_INTERPRET}" \
  --regression_task "${REGRESSION_TASK}" \
  --classification_task "${CLASSIFICATION_TASK}" \
  --mclassification_task "${MCLASSIFICATION_TASK}" \
  --epochs "${EPOCHS}" \
  --n "${N}" \
  --evaluate_epoch "${EVALUATE_EPOCH}" \
  --total_iters "${TOTAL_ITERS}" \
  --evaluate_step "${EVALUATE_STEP}" \
  --lrate "${LRATE}" \
  --eps "${EPS}" \
  --betas "${BETAS}" \
  --batch_size "${BATCH_SIZE}" \
  --datafolder "${DATAFOLDER}" \
  --trained_model_path "${TRAINED_MODEL_PATH}"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "[OK] finished at $(date)"
else
  echo "[ERROR] exit code ${exit_code} at $(date)"
  exit ${exit_code}
fi

echo "JOB END: $(date)"
