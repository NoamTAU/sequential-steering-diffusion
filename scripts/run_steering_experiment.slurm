#!/bin/bash
#SBATCH --job-name=steer_husky_cat
#SBATCH --output=slurm_logs/steer_%j.out
#SBATCH --error=slurm_logs/steer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

# Config
IMG="/work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG"
OUT="/work/pcsl/Noam/sequential_diffusion/results/steering_force_analysis"

# Run with logging enabled
python steered_sequential_uturns_v4.py \
    --start_image_path "$IMG" \
    --output_dir "$OUT" \
    --orig_class_idx 249 \
    --target_class_idx 281 \
    --num_steps 30 \
    --noise_step 100 \
    --timestep_respacing "250" \
    --penalty 1.0 \
    --batch_size 64 \
    --max_retries 4