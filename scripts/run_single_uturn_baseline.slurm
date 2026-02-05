#!/bin/bash

#----------------------------------------------------------------#
# Slurm Job Array for Single U-Turn Baseline (Images x Noise Levels)
#----------------------------------------------------------------#

#SBATCH --job-name=single_baseline
##SBATCH --array=0-349                  # 50 images * 7 noise levels
#SBATCH --array=0-149                  # 50 images * 7 noise levels
#SBATCH --output=slurm_logs/single_%A_%a.out
#SBATCH --error=slurm_logs/single_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8              # Keep 8 for memory ratio
#SBATCH --mem=32G
#SBATCH --time=01:00:00                # 1 U-turn * 50 trajectories is very fast (~15 mins)
#SBATCH --partition=h100
#SBATCH --gres=gpu:1

#----------------------------------------------------------------#
# Configuration
#----------------------------------------------------------------#

# 1. Same Noise Levels as your sequential run
## NOISE_LEVELS=(25 100 200 300 400 500 600 700 800 900 999)

NOISE_LEVELS=(25 100 200)


NUM_NOISE_LEVELS=${#NOISE_LEVELS[@]}

# 2. Same Image List
IMAGE_LIST_FILE="image_list.txt"

# 3. SETTINGS FOR SINGLE U-TURN BASELINE
NUM_UTURNS=1           # <--- Only do 1 step
NUM_TRAJECTORIES=50    # <--- Get 50 statistics for that step

# 4. NEW OUTPUT DIRECTORY
# Saves to a separate folder so we don't mix with sequential data
OUTPUT_DIR="/work/pcsl/Noam/sequential_diffusion/results/single_uturn_baseline"

#----------------------------------------------------------------#
# Logic to Map Task ID to (Image, Noise)
#----------------------------------------------------------------#

IMAGE_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_NOISE_LEVELS))
NOISE_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_NOISE_LEVELS))

CURRENT_NOISE_STEP=${NOISE_LEVELS[$NOISE_INDEX]}
START_IMAGE=$(sed -n "$((IMAGE_INDEX + 1))p" $IMAGE_LIST_FILE)

#----------------------------------------------------------------#
# Execution
#----------------------------------------------------------------#

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Image Index: $IMAGE_INDEX"
echo "Noise Level: $CURRENT_NOISE_STEP"
echo "Output Dir: $OUTPUT_DIR"

if [ -z "$START_IMAGE" ]; then
    echo "Error: Could not find image for index $IMAGE_INDEX"
    exit 1
fi

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

echo "Starting Python script..."
python sequential_uturns.py \
    --start_image_path "$START_IMAGE" \
    --num_uturns "$NUM_UTURNS" \
    --noise_step "$CURRENT_NOISE_STEP" \
    --num_trajectories "$NUM_TRAJECTORIES" \
    --output_dir "$OUTPUT_DIR"

echo "Job finished."