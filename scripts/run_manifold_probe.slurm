#!/bin/bash
#SBATCH --job-name=probe_husky
#SBATCH --output=slurm_logs/probe_%a.out
#SBATCH --error=slurm_logs/probe_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --array=0-49

# --- CONFIGURATION ---
# The starting point (Husky)
START_IMAGE="/work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG"

# Output location
OUTPUT_DIR="/work/pcsl/Noam/sequential_diffusion/results/manifold_probe"

# Parameters
NOISE_STEP=40
NUM_STEPS=200

# Setup Environment
source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"

# Run the Probe Script
# Note: We use the Array ID as the trajectory index
python probe_manifold.py \
    --start_image_path "$START_IMAGE" \
    --output_dir "$OUTPUT_DIR" \
    --trajectory_idx "$SLURM_ARRAY_TASK_ID" \
    --num_steps "$NUM_STEPS" \
    --noise_step "$NOISE_STEP"

echo "Job finished."