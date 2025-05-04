#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------
# train-mdm.sh — wrapper to submit one SLURM job that runs train + eval
# Usage: bash train-mdm.sh <TRAIN_DATASET> <TEST_DATASET> <NUM_GPUS> [EXCLUDE_NODES]
# ---------------------------------------------------------------------

TRAIN_DS=${1:-}
TEST_DS=${2:-}
NGPU=${3:-}
EXCLUDE_NODES=${4:-}
if [[ -z $TRAIN_DS || -z $TEST_DS || -z $NGPU ]]; then
  echo "Usage: $0 <TRAIN_DATASET> <TEST_DATASET> <NUM_GPUS> [EXCLUDE_NODES]"
  exit 1
fi

# central storage & run name
ETH_USERNAME=adiego
PROJECT_NAME=diffusion-vs-ar
CENTRAL_DIR=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
RUN_NAME="${TRAIN_DS}_${NGPU}GPU_$(date '+%Y%m%d-%H%M%S')"
mkdir -p "${CENTRAL_DIR}/output/${RUN_NAME}"

# optional exclude directive
EXC_SBATCH=
if [[ -n $EXCLUDE_NODES ]]; then
  EXC_SBATCH="#SBATCH --exclude=${EXCLUDE_NODES},tikgpu08,tikgpu10"
fi

# write one SBATCH script that does both train + eval
JOB_SCRIPT=${CENTRAL_DIR}/output/${RUN_NAME}/job-run.sh
cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=mdm-${RUN_NAME}
#SBATCH --output=${CENTRAL_DIR}/output/${RUN_NAME}/slurm-%j.out
#SBATCH --error=${CENTRAL_DIR}/output/${RUN_NAME}/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=${NGPU}
#SBATCH --cpus-per-task=8
#SBATCH --mem=$((15 * NGPU))G
#SBATCH --gres=gpu:${NGPU}
${EXC_SBATCH}
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL

# parameters baked in
TRAIN_DS="${TRAIN_DS}"
TEST_DS="${TEST_DS}"
NGPU="${NGPU}"
PROJECT_NAME="${PROJECT_NAME}"
CENTRAL_DIR="${CENTRAL_DIR}"
RUN_NAME="${RUN_NAME}"

# scratch paths (compute node only!)
NODE_SCRATCH_BASE="/scratch/${ETH_USERNAME}"
SCRATCH_PROJECT_DIR="\${NODE_SCRATCH_BASE}/\${PROJECT_NAME}"
EXP_DIR="\${SCRATCH_PROJECT_DIR}/output/\${RUN_NAME}"

# make dirs
mkdir -p "\${EXP_DIR}" "\${SCRATCH_PROJECT_DIR}"

# sync code & data to scratch
rsync -az --exclude 'output' "\${CENTRAL_DIR}/" "\${SCRATCH_PROJECT_DIR}/"

mkdir -p "\${EXP_DIR}/model_config"
cp -r "${SCRATCH_PROJECT_DIR}/model_config/." "${EXP_DIR}/model_config/"

# activate conda & set HF caches on scratch
if [[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]]; then
  eval "\$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
  conda activate diffusion
fi
export TRANSFORMERS_CACHE="\${SCRATCH_PROJECT_DIR}/hf_cache"
export HF_HOME="\${TRANSFORMERS_CACHE}"
export HF_DATASETS_CACHE="\${TRANSFORMERS_CACHE}/datasets"
mkdir -p "\${TRANSFORMERS_CACHE}"

# on-exit sync everything back to central
copy_back() {
  mkdir -p "${CENTRAL_DIR}/output/${RUN_NAME}"
  rsync -az "\${EXP_DIR}/" "${CENTRAL_DIR}/output/${RUN_NAME}/"
}
trap copy_back EXIT HUP INT TERM

# compute training flags
if (( NGPU > 1 )); then MP_FLAG="--multi_gpu"; else MP_FLAG=""; fi
NUM_PROC="--num_processes ${NGPU}"
GA_STEPS=$((8 / NGPU))

# ----- Training -----
cd "\${SCRATCH_PROJECT_DIR}"
accelerate launch \\
  --num_machines 1 --mixed_precision fp16 \\
  \${NUM_PROC} \${MP_FLAG} --main_process_port 20099 src/train_bash.py \\
  --report_to tensorboard --logging_dir "\${EXP_DIR}/tb_logs" \\
  --stage mdm --overwrite_output_dir --cache_dir ./cache \\
  --model_name_or_path model_config --do_train --dataset "\${TRAIN_DS}" \\
  --finetuning_type full --cutoff_len 64 --output_dir "\${EXP_DIR}" \\
  --overwrite_cache --per_device_train_batch_size 128 \\
  --gradient_accumulation_steps "\${GA_STEPS}" \\
  --lr_scheduler_type cosine --logging_steps 100 \\
  --val_size 448 --per_device_eval_batch_size 32 \\
  --evaluation_strategy steps --eval_steps 5000 --save_steps 10000 \\
  --learning_rate 3e-4 --num_train_epochs 600.0 --plot_loss --run_name "\${RUN_NAME}_prefix" \\
  --preprocessing_num_workers 8 --fp16 --save_total_limit 1 \\
  --remove_unused_columns False --diffusion_steps 20 \\
  --save_safetensors False --token_reweighting True \\
  --time_reweighting linear --topk_decoding True \\
  --alpha 0.25 --gamma 2 > "\${EXP_DIR}/train.log"

echo "[INFO] Training completed at \$(date). Starting evaluation..."

# ----- Evaluation -----
mkdir -p "\${EXP_DIR}/\${TEST_DS}"
accelerate launch \\
  --num_machines 1 --mixed_precision fp16 --num_processes 1 --main_process_port 20100 src/train_bash.py \\
  --stage mdm --overwrite_output_dir --cache_dir ./cache \\
  --model_name_or_path model_config --do_predict --cutoff_len 64 --dataset "\${TEST_DS}" \\
  --finetuning_type full --diffusion_steps 20 \\
  --output_dir "\${EXP_DIR}/\${TEST_DS}" --checkpoint_dir "\${EXP_DIR}" \\
  --remove_unused_columns False --decoding_strategy stochastic0.5-linear \\
  --topk_decoding True > "\${EXP_DIR}/\${TEST_DS}/eval-TopKTrue.log"
EOF

chmod +x "${JOB_SCRIPT}"

# submit the single job
# sbatch "${JOB_SCRIPT}"
echo "Submitted combined train+eval job mdm-${RUN_NAME}"
