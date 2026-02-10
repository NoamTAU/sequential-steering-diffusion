#!/bin/bash
#SBATCH --job-name=steer_dog2dog
#SBATCH --output=slurm_logs/steer_dog2dog_%j.out
#SBATCH --error=slurm_logs/steer_dog2dog_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

# Ensure logs directory exists in the working dir
mkdir -p slurm_logs

# GPU status (helps debug OOM)
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
nvidia-smi -L
nvidia-smi

# Config
IMG="/work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG"
OUT="/work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1"
TARGET_PROB_STOP=0.60

# Run with logging enabled (auto-select target dog class)
python scripts/steered_sequential_uturns_v4.py \
    --start_image_path "$IMG" \
    --output_dir "$OUT" \
    --orig_class_idx 249 \
    --target_class_idx 0 \
    --auto_target_dog True \
    --auto_target_topk 5 \
    --num_steps 50 \
    --noise_step 100 \
    --timestep_respacing "250" \
    --penalty 1.0 \
    --batch_size 64 \
    --max_retries 4 \
    --require_target_increase True \
    --target_prob_stop ${TARGET_PROB_STOP} \
    --fail_behavior skip \
    --classifier_use_fp16 True
