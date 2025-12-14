


set -eo pipefail

# --- Clean environment to avoid ~/.local issues ---
module purge
module load cuda/11.8 cudnn
module load anaconda3/latest
source $(conda info --base)/etc/profile.d/conda.sh

export PROJ=/sc/arion/projects/DiseaseGeneCell/Huang_lab_data
export CONDARC="$PROJ/conda/condarc"
conda activate /sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/psichic

ml proxies/1 || true
export PYTHONUNBUFFERED=1 TERM=xterm
export RAYON_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Project paths ---
LOG_DIR="logs"; mkdir -p "${LOG_DIR}"

OUTPUT_DIR="output/data"; mkdir -p "${OUTPUT_DIR}"
LOG_LEVEL="INFO"

BASE_DATA_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/EnzymaticReactionPrediction/Regression_Data/"
BASE_DATA_DIR="${BASE_DATA_DIR}/exp_of_catpred_MPEK_EITLEM_inhouse_dataset/experiments/"
DATASET="catpred"
LABEL="kcat"
SPLITMODE="drug"
MAIN="src/create_train_val_test_data.py"
N=0
combo="${DATASET}_${SPLITMODE}"
ts=$(date +"%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/${ts}_create_train_val_test_data_${combo}.log"

echo "JOBID=${LSB_JOBID:-local}  IDX=${LSB_JOBINDEX:-}  HOST=$(hostname)"
echo "=== Running ${combo} ==="
echo "  data_dir : ${BASE_DATA_DIR}"
echo "  log_file : ${log_file}"
echo "  dataset  : ${DATASET}"
echo "  label    : ${LABEL}"
echo "  splitmode: ${SPLITMODE}"

set +e
"${PYTHON}" "${MAIN}" \
  --log_fn "${log_file}" \
  --log_level "${LOG_LEVEL}" \
  --data_dir "${BASE_DATA_DIR}" \
  --dataset_name "${DATASET}" \
  --label "${LABEL}" \
  --split_mode "${SPLITMODE}" \
  --N "${N}" \
  --output_dir "${OUTPUT_DIR}"
exit_code=$?
set -e