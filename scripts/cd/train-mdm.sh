#!/bin/bash
#SBATCH --job-name=mdm-alpha0.25
#SBATCH --output=/itet-stor/adiego/net_scratch/diffusion-vs-ar/jobs/%j.out
#SBATCH --error=/itet-stor/adiego/net_scratch/diffusion-vs-ar/jobs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4                     # one process per GPU
#SBATCH --cpus-per-task=8              # preprocessing workers
#SBATCH --mem=60G                      # < ½ node RAM
#SBATCH --gres=gpu:4
#SBATCH --exclude=tikgpu04,tikgpu06,tikgpu08,tikgpu10
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL

# ---------------------------------------------------------------------
# 0. Static paths and settings
# ---------------------------------------------------------------------
ETH_USERNAME=adiego
PROJECT_NAME=diffusion-vs-ar

CENTRAL_DIR=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}   # code + data + long-term storage
NODE_SCRATCH_BASE=/scratch/${USER}                                   # semi-permanent local disk
SCRATCH_PROJECT_DIR=${NODE_SCRATCH_BASE}/${PROJECT_NAME}             # job runs here

CONDA_ENVIRONMENT=diffusion
DATASETS=(cd6_train cd6_test)                                        # add more here if needed

# ---------------------------------------------------------------------
# 1. Utility: copy results back no matter how the job ends
# ---------------------------------------------------------------------
copy_back () {
  echo "[INFO] Syncing results back to central storage ..."
  rsync -az --delete "${SCRATCH_PROJECT_DIR}/output" "${CENTRAL_DIR}/"
  echo "[INFO] Sync finished ($(date))."
}
trap copy_back EXIT HUP INT TERM

# ---------------------------------------------------------------------
# 2. Prepare local project copy
# ---------------------------------------------------------------------
mkdir -p "${SCRATCH_PROJECT_DIR}"
echo "[INFO] Rsyncing source code (without previous outputs) ..."
rsync -az --exclude 'output' "${CENTRAL_DIR}/" "${SCRATCH_PROJECT_DIR}/"

# 2a) Ensure datasets are on the node once
for ds in "${DATASETS[@]}"; do
  file="${ds}.jsonl"                             # every key maps to file_name in JSON
  if [[ ! -e "${SCRATCH_PROJECT_DIR}/data/${file}" ]]; then
    echo "[INFO] Copying dataset '${file}' to scratch ..."
    rsync -az "${CENTRAL_DIR}/data/${file}" "${SCRATCH_PROJECT_DIR}/data/"
  else
    echo "[INFO] Dataset '${file}' already on scratch – skip copy."
  fi
done

# 2b) Copy model_config once (avoids repeated NFS reads)
if [[ ! -d "${SCRATCH_PROJECT_DIR}/model_config" ]]; then
  rsync -az "${CENTRAL_DIR}/model_config" "${SCRATCH_PROJECT_DIR}/"
fi

# ---------------------------------------------------------------------
# 3. Activate environment
# ---------------------------------------------------------------------
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] \
  && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate "${CONDA_ENVIRONMENT}"
echo "[INFO] Conda '${CONDA_ENVIRONMENT}' activated."

# ---------------------------------------------------------------------
# 4. Use local caches to avoid network traffic
# ---------------------------------------------------------------------
export TRANSFORMERS_CACHE="${SCRATCH_PROJECT_DIR}/hf_cache"
export HF_HOME="${TRANSFORMERS_CACHE}"
export HF_DATASETS_CACHE="${TRANSFORMERS_CACHE}/datasets"
mkdir -p "${TRANSFORMERS_CACHE}"

# ---------------------------------------------------------------------
# 5. Run the experiment from scratch disk
# ---------------------------------------------------------------------
cd "${SCRATCH_PROJECT_DIR}"

EXP_DIR="${SCRATCH_PROJECT_DIR}/output/cd4/mdm-alpha0.25-gamma2-bs1024-lr3e-4-ep600-T20-$(date '+%Y%m%d-%H%M%S')"
mkdir -p "${EXP_DIR}"

echo "[INFO] Starting training at $(date) on $(hostname)"
accelerate launch \
  --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 4 \
  --main_process_port 20099 \
  src/train_bash.py \
    --report_to none \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config \
    --do_train \
    --dataset cd6_train \
    --finetuning_type full \
    --cutoff_len 64 \
    --output_dir "${EXP_DIR}" \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 3e-4 \
    --num_train_epochs 600.0 \
    --plot_loss \
    --run_name cd6_train_prefix \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting True \
    --time_reweighting linear \
    --topk_decoding True \
    --alpha 0.25 \
    --gamma 2 \
  > "${EXP_DIR}/train.log"

# ---------------------------------------------------------------------
# 6. Prediction runs
# ---------------------------------------------------------------------
for dataset in cd6_test; do
  mkdir -p "${EXP_DIR}/${dataset}"
  accelerate launch \
    --num_machines 1 --mixed_precision fp16 --num_processes 1 \
    --main_process_port 20100 \
    src/train_bash.py \
      --stage mdm --overwrite_output_dir \
      --cache_dir ./cache \
      --model_name_or_path model_config \
      --do_predict \
      --cutoff_len 64 \
      --dataset "${dataset}" \
      --finetuning_type full \
      --diffusion_steps 20 \
      --output_dir "${EXP_DIR}/${dataset}" \
      --checkpoint_dir "${EXP_DIR}" \
      --remove_unused_columns False \
      --decoding_strategy stochastic0.5-linear \
      --topk_decoding True \
    > "${EXP_DIR}/${dataset}/eval-TopKTrue.log"
done

echo "[INFO] Job finished at $(date)"
exit 0
