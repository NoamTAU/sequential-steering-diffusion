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

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

# Config
IMG="/work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG"
OUT="/work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2"

# Run with logging enabled
python scripts/steered_sequential_uturns_meta.py \
    --start_image_path "$IMG" \
    --output_dir "$OUT" \
    --num_steps 50 \
    --noise_step 100 \
    --timestep_respacing "250" \
    --penalty 1.0 \
    --batch_size 64 \
    --max_retries 4
