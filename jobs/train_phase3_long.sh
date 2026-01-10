#!/bin/bash
# train_phase3_long.sh - Fine-tune on LONG proteins (>1500 residues)
# Phase 3: Freeze most layers, fine-tune only output heads

# ------- LSF resources ------
#BSUB -J train_p3
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h10080g
#BSUB -n 16
#BSUB -R "rusage[mem=80G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/train_p3.%J.out
#BSUB -e logs/train_p3.%J.err
# --------------------------------
set -Eeo pipefail
trap 'ec=$?; echo "[ERROR] line ${LINENO} status ${ec}" >&2' ERR

mkdir -p ./logs

LOG_LEVEL="INFO"
ts="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="./logs/${ts}_train_phase3.log"

echo "------------------------------------------------------------"
echo "PHASE 3: Fine-tuning on LONG proteins (>1500 residues)"
echo "JOB START: $(date)"
echo "JOBID     : ${LSB_JOBID:-local}"
echo "HOST      : $(hostname)"
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
export PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
export CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"
export PYTHONNOUSERSITE=1 TERM=xterm PYTHONUNBUFFERED=1
unset PYTHONPATH || true

conda activate "${ENV_PREFIX}" || { echo "[ERROR] conda activate failed"; exit 1; }
PYTHON="${ENV_PREFIX}/bin/python"

export HF_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models"
export TORCH_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.torch_hub/checkpoints/"

# ---- PHASE 3 SETTINGS ----
DATASET="catpred"
LABEL="kcat"
SPLITMODE="drug"

CONFIG_PATH="config"
# Load from Phase 2
TRAINED_MODEL_PATH="./output/models/PSICHIC/curriculum/${DATASET}_${LABEL}_${SPLITMODE}/phase2_medium"
RESULT_PATH="./output/models/PSICHIC/curriculum/${DATASET}_${LABEL}_${SPLITMODE}/phase3_long"
mkdir -p "${RESULT_PATH}"
MODEL_PATH="${RESULT_PATH}/save_models"
mkdir -p "${MODEL_PATH}"

REGRESSION_TASK=True
CLASSIFICATION_TASK=False
MCLASSIFICATION_TASK=0
SAVE_INTERPRET=False
EPOCHS=20
EVALUATE_EPOCH=5
TOTAL_ITERS=0
LRATE=5e-6              # Even lower learning rate
EPS=1e-8
BETAS="(0.9,0.999)"
SAVE_MODEL=True

# KEY SETTINGS FOR PHASE 3
BATCH_SIZE=1            # Minimum batch for very long proteins
MIN_PROT_LEN=1501       # Proteins > 1500 only
MAX_PROT_LEN=0          # No upper limit (0 = disabled)

# Fine-tune only output heads (freeze everything else)
FINETUNE_MODULES="['reg_out','cls_out','mcls_out']"

DATAFOLDER="./data/${DATASET}_${LABEL}_${SPLITMODE}"
MAIN="src/train.py"

[[ -f "${MAIN}" ]] || { echo "[ERROR] MAIN not found"; exit 2; }
[[ -d "${TRAINED_MODEL_PATH}" ]] || { echo "[ERROR] Phase 2 model not found: ${TRAINED_MODEL_PATH}"; exit 2; }

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=== GPU status ===" && nvidia-smi && echo "=================="
echo "Phase 3 Settings:"
echo "  Loading from: ${TRAINED_MODEL_PATH}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Protein length: >${MIN_PROT_LEN}"
echo "  Fine-tune modules: ${FINETUNE_MODULES}"
echo "------------------------------------------------------------"

set +e
"${PYTHON}" "${MAIN}" \
  --log_fn "${LOG_FILE}" \
  --log_level "${LOG_LEVEL}" \
  --config_path "${CONFIG_PATH}" \
  --result_path "${RESULT_PATH}" \
  --model_path "${MODEL_PATH}" \
  --trained_model_path "${TRAINED_MODEL_PATH}" \
  --finetune_modules "${FINETUNE_MODULES}" \
  --save_interpret "${SAVE_INTERPRET}" \
  --regression_task "${REGRESSION_TASK}" \
  --classification_task "${CLASSIFICATION_TASK}" \
  --mclassification_task "${MCLASSIFICATION_TASK}" \
  --epochs "${EPOCHS}" \
  --evaluate_epoch "${EVALUATE_EPOCH}" \
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
  --embedding_name "ESMv1" \
  --split_name "${SPLITMODE}"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "[OK] Phase 3 finished at $(date)"
  echo "CURRICULUM TRAINING COMPLETE!"
  echo "Final model: ${RESULT_PATH}"
else
  echo "[ERROR] exit code ${exit_code}" && exit ${exit_code}
fi

