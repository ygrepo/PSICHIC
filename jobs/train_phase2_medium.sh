#!/bin/bash
# train_phase2_medium.sh - Fine-tune on MEDIUM proteins (500-1500 residues)
# Phase 2: Freeze backbone, fine-tune interaction & output layers

# ------- LSF resources ------
#BSUB -J train_p2
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h10080g
#BSUB -n 16
#BSUB -R "rusage[mem=80G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
#BSUB -o logs/train_p2.%J.out
#BSUB -e logs/train_p2.%J.err
# --------------------------------
set -Eeo pipefail
trap 'ec=$?; echo "[ERROR] line ${LINENO} status ${ec}" >&2' ERR

mkdir -p ./logs

LOG_LEVEL="INFO"
ts="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="./logs/${ts}_train_phase2.log"

echo "------------------------------------------------------------"
echo "PHASE 2: Fine-tuning on MEDIUM proteins (500-1500 residues)"
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

# ---- PHASE 2 SETTINGS ----
DATASET="catpred"
LABEL="kcat"
SPLITMODE="drug"

CONFIG_PATH="config"
# Load from Phase 1
TRAINED_MODEL_PATH="./output/models/PSICHIC/curriculum/${DATASET}_${LABEL}_${SPLITMODE}/phase1_short/save_models"
RESULT_PATH="./output/models/PSICHIC/curriculum/${DATASET}_${LABEL}_${SPLITMODE}/phase2_medium"
mkdir -p "${RESULT_PATH}"
MODEL_PATH="${RESULT_PATH}/save_models"
mkdir -p "${MODEL_PATH}"

REGRESSION_TASK=True
CLASSIFICATION_TASK=False
MCLASSIFICATION_TASK=0
SAVE_INTERPRET=False
EPOCHS=100
EVALUATE_EPOCH=5
TOTAL_ITERS=0
LRATE=1e-5              # Lower learning rate for fine-tuning
EPS=1e-8
BETAS="(0.9,0.999)"
SAVE_MODEL=True

# KEY SETTINGS FOR PHASE 2
BATCH_SIZE=4            # Smaller batch for medium proteins
MIN_PROT_LEN=501        # Proteins > 500
MAX_PROT_LEN=1500       # Proteins ≤ 1500

# Fine-tune only interaction and output layers (freeze encoders & convolutions)
FINETUNE_MODULES="['inter_convs','reg_out','cls_out','mcls_out','mol_out','prot_out']"

DATAFOLDER="./data/${DATASET}_${LABEL}_${SPLITMODE}"
MAIN="src/train.py"

[[ -f "${MAIN}" ]] || { echo "[ERROR] MAIN not found"; exit 2; }
[[ -d "${TRAINED_MODEL_PATH}" ]] || { echo "[ERROR] Phase 1 model not found: ${TRAINED_MODEL_PATH}"; exit 2; }

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=== GPU status ===" && nvidia-smi && echo "=================="
echo "Phase 2 Settings:"
echo "  Loading from: ${TRAINED_MODEL_PATH}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Protein length: ${MIN_PROT_LEN} - ${MAX_PROT_LEN}"
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
  echo "[OK] Phase 2 finished at $(date)"
  echo "Next: Run phase 3 with --trained_model_path ${RESULT_PATH}"
else
  echo "[ERROR] exit code ${exit_code}" && exit ${exit_code}
fi

