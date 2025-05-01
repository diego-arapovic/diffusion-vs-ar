#!/bin/bash
#SBATCH --job-name=mdm-alpha0.25
#SBATCH --output=/itet-stor/adiego/net_scratch/diffusion-vs-ar/jobs/%j.out
#SBATCH --error=/itet-stor/adiego/net_scratch/diffusion-vs-ar/jobs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4                     # one process per GPU
#SBATCH --cpus-per-task=4              # for preprocessing_num_workers
#SBATCH --mem=60G                     # stays under half of node memory
#SBATCH --gres=gpu:4                   # request 8 GPUs
#SBATCH --exclude=tikgpu02,tikgpu03,tikgpu04,tikgpu05,artongpu02,artongpu03,artongpu04,artongpu05,artongpu07,tikgpu08,tikgpu10
#SBATCH --time=6:00:00                # adjust as needed
#SBATCH --mail-type=END,FAIL

ETH_USERNAME=adiego
PROJECT_NAME=diffusion-vs-ar
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=diffusion
mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
echo 'Failed to create temp directory' >&2
exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"


[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

#— Experiment output dir —
exp=/itet-stor/${USER}/net_scratch/diffusion-vs-ar/output/cd4/mdm-alpha0.25-gamma2-bs1024-lr3e-4-ep600-T20-$(date "+%Y%m%d-%H%M%S")
mkdir -p "$exp"

#— Training —
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
    --output_dir "$exp" \
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
  > "$exp/train.log"

#— Prediction for test sets —
for dataset in cd6_test; do
  mkdir -p "$exp/$dataset"
  accelerate launch \
    --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 1 \
    --main_process_port 20100 \
    src/train_bash.py \
      --stage mdm --overwrite_output_dir \
      --cache_dir ./cache \
      --model_name_or_path model_config \
      --do_predict \
      --cutoff_len 64 \
      --dataset "$dataset" \
      --finetuning_type full \
      --diffusion_steps 20 \
      --output_dir "$exp/$dataset" \
      --checkpoint_dir "$exp" \
      --remove_unused_columns False \
      --decoding_strategy stochastic0.5-linear \
      --topk_decoding True \
    > "$exp/${dataset}/eval-TopKTrue.log"
done

echo "Training and evaluation completed. Check the logs in $exp for details."
exit 0
