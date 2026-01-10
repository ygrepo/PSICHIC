#!/bin/bash
# train_phase1_short.sh - Train on SHORT proteins (≤500 residues)
# Phase 1 of curriculum learning: Full training on easy examples

# ------- LSF resources ------
#BSUB -J train_p1
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R a10080g
#BSUB -n 2
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
#BSUB -o logs/train_p1.%J.out
#BSUB -e logs/train_p1.%J.err
# --------------------------------
set -Eeo pipefail
trap 'ec=$?; echo "[ERROR] line ${LINENO} status ${ec}" >&2' ERR

mkdir -p ./logs

SUBMIT_DIR="$(pwd)"
echo "Submit dir: ${SUBMIT_DIR}"

LOG_LEVEL="INFO"
mkdir -p ./logs
ts="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="./logs/${ts}_train_phase1.log"

echo "------------------------------------------------------------"
echo "PHASE 1: Training on SHORT proteins (≤500 residues)"
echo "JOB START: $(date)"
echo "JOBID     : ${LSB_JOBID:-local}"
echo "HOST      : $(hostname)"
echo "LOG FILE  : ${LOG_FILE}"
echo "------------------------------------------------------------"

# ---- Modules / shell setup ----
module purge || true
module load anaconda3/latest || true
module load cuda/12.4.0 || true
ml proxies/1 || true

if ! base_dir="$(conda info --base 2>/dev/null)"; then
  base_dir="$HOME/miniconda3"
fi
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
export TORCH_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.torch_hub/checkpoints/"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ---- PHASE 1 SETTINGS ----
MODEL_PLM_TYPE="ESMv1"
MODEL_PLM_FN="$TORCH_HOME/esm1v_t33_650M_UR90S_5.pt"
DATASET="catpred"
LABEL="kcat"
SPLITMODE="drug"

CONFIG_PATH="config"
RESULT_PATH="./output/models/PSICHIC/curriculum/${DATASET}_${LABEL}_${SPLITMODE}/phase1_short"
mkdir -p "${RESULT_PATH}"
MODEL_PATH="${RESULT_PATH}/save_models"
mkdir -p "${MODEL_PATH}"

# Phase 1: Full training on short proteins
REGRESSION_TASK=True
CLASSIFICATION_TASK=False
MCLASSIFICATION_TASK=0
SAVE_INTERPRET=False
EPOCHS=50
EVALUATE_EPOCH=5
TOTAL_ITERS=0
LRATE=1e-4
EPS=1e-8
BETAS="(0.9,0.999)"
SAVE_MODEL=True

# KEY SETTINGS FOR PHASE 1
BATCH_SIZE=16           # Can use larger batch with short proteins
MIN_PROT_LEN=0          # No minimum
MAX_PROT_LEN=500        # Only proteins ≤ 500 residues

DATAFOLDER="./data/${DATASET}_${LABEL}_${SPLITMODE}"
MAIN="src/train.py"

[[ -f "${MAIN}" ]] || { echo "[ERROR] MAIN not found: ${MAIN}"; exit 2; }

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=== GPU status ==="
nvidia-smi
echo "=================="

echo "Phase 1 Settings:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Protein length: ${MIN_PROT_LEN} - ${MAX_PROT_LEN}"
echo "  Result path: ${RESULT_PATH}"
echo "------------------------------------------------------------"

set +e
"${PYTHON}" "${MAIN}" \
  --log_fn "${LOG_FILE}" \
  --log_level "${LOG_LEVEL}" \
  --config_path "${CONFIG_PATH}" \
  --result_path "${RESULT_PATH}" \
  --model_path "${MODEL_PATH}" \
  --model_plm_type "${MODEL_PLM_TYPE}" \
  --model_plm_fn "${MODEL_PLM_FN}" \
  --save_interpret "${SAVE_INTERPRET}" \
  --regression_task "${REGRESSION_TASK}" \
  --classification_task "${CLASSIFICATION_TASK}" \
  --mclassification_task "${MCLASSIFICATION_TASK}" \
  --epochs "${EPOCHS}" \
  --evaluate_epoch "${EVALUATE_EPOCH}" \
  --total_iters "${TOTAL_ITERS}" \
  --lrate "${LRATE}" \
  --eps "${EPS}" \
  --betas "${BETAS}" \
  --batch_size "${BATCH_SIZE}" \
  --datafolder "${DATAFOLDER}" \
  --save_model "${SAVE_MODEL}" \
  --min_prot_len "${MIN_PROT_LEN}" \
  --max_prot_len "${MAX_PROT_LEN}" \
  --data_name "${DATASET}" \
  --label_name "${LABEL}" \
  --embedding_name "${MODEL_PLM_TYPE}" \
  --split_name "${SPLITMODE}"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "[OK] Phase 1 finished at $(date)"
  echo "Next: Run phase 2 with --trained_model_path ${RESULT_PATH}"
else
  echo "[ERROR] exit code ${exit_code} at $(date)"
  exit ${exit_code}
fi

