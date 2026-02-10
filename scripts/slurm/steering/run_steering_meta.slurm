#!/bin/bash
#SBATCH --job-name=steer_meta_cat
#SBATCH --output=slurm_logs/steer_meta_%j.out
#SBATCH --error=slurm_logs/steer_meta_%j.err
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
OUT="/work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2"
TARGET_PROB_STOP=0.90

# Run with logging enabled
python scripts/steered_sequential_uturns_meta.py \
    --start_image_path "$IMG" \
    --output_dir "$OUT" \
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
