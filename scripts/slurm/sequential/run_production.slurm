#!/bin/bash

#----------------------------------------------------------------#
# Slurm Job Array for Mass Production (Images x Noise Levels)
#----------------------------------------------------------------#

#SBATCH --job-name=prod_uturns
#SBATCH --array=0-349
#SBATCH --output=slurm_logs/prod_%A_%a.out
#SBATCH --error=slurm_logs/prod_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# --- FIX: Increase CPUs to satisfy memory ratio ---
#SBATCH --cpus-per-task=8              # Increased from 4 to 8 to allow 32G RAM

#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1


#----------------------------------------------------------------#
# configuration
#----------------------------------------------------------------#

# 1. Define Noise Levels
NOISE_LEVELS=(25 50 75 100 150 200 250)
NUM_NOISE_LEVELS=${#NOISE_LEVELS[@]}

# 2. Define Image List File
IMAGE_LIST_FILE="image_list.txt"

# 3. Other Constants
NUM_UTURNS=100
NUM_TRAJECTORIES=50

#----------------------------------------------------------------#
# Logic to Map Task ID to (Image, Noise)
#----------------------------------------------------------------#

# Calculate indices using integer division and modulo
# Image Index changes slowly (every 7 tasks)
# Noise Index changes fast (every task)
IMAGE_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_NOISE_LEVELS))
NOISE_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_NOISE_LEVELS))

# Get the specific Noise Level
CURRENT_NOISE_STEP=${NOISE_LEVELS[$NOISE_INDEX]}

# Get the specific Image Path (sed uses 1-based indexing, so we add 1)
START_IMAGE=$(sed -n "$((IMAGE_INDEX + 1))p" $IMAGE_LIST_FILE)

#----------------------------------------------------------------#
# Execution
#----------------------------------------------------------------#

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "------------------------------------------------"
echo "Processing Image Index: $IMAGE_INDEX"
echo "Image Path: $START_IMAGE"
echo "Processing Noise Index: $NOISE_INDEX"
echo "Noise Level: $CURRENT_NOISE_STEP"
echo "------------------------------------------------"

# Safety check
if [ -z "$START_IMAGE" ]; then
    echo "Error: Could not find image for index $IMAGE_INDEX"
    exit 1
fi

# Load Environment
source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

# Run Script
echo "Starting Python script..."
python sequential_uturns.py \
    --start_image_path "$START_IMAGE" \
    --num_uturns "$NUM_UTURNS" \
    --noise_step "$CURRENT_NOISE_STEP" \
    --num_trajectories "$NUM_TRAJECTORIES"

echo "Job finished at $(date)"