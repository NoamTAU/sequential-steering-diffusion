#!/bin/bash

#----------------------------------------------------------------#
# Slurm Job Array to Extend Sequential U-Turn Trajectories
#----------------------------------------------------------------#

#SBATCH --job-name=uturn_extend
#SBATCH --array=1-150                  # Total jobs = 3 noise levels * 50 trajectories
#SBATCH --output=slurm_logs/extend_%A_%a.out
#SBATCH --error=slurm_logs/extend_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00                # Time for 100 U-turns. 30 mins is a safe buffer.
#SBATCH --partition=h100
#SBATCH --gres=gpu:1

# --- KEY PARAMETERS FOR YOUR EXPERIMENT ---
# The noise levels you want to extend
NOISE_LEVELS_TO_EXTEND=(25 50 75)

# The number of trajectories you have for EACH noise level
TRAJECTORIES_PER_LEVEL=50

# The number of NEW U-turns you want to add
NEW_UTURNS_TO_ADD=100

# The path to the starting image (still needed by the script to create paths)
START_IMAGE="/work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG"

#----------------------------------------------------------------#
# Commands to run for EACH of the 150 array tasks
#----------------------------------------------------------------#

echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job started on $(hostname) at $(date)"

# Activate Conda environment
source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

# --- Map the Slurm Task ID to a Noise Level and Trajectory Index ---
# This math converts a single number (1-150) into a (noise, trajectory) pair.
TASK_ZERO_BASED=$((SLURM_ARRAY_TASK_ID - 1))
NOISE_INDEX=$((TASK_ZERO_BASED / TRAJECTORIES_PER_LEVEL))
TRAJ_INDEX=$((TASK_ZERO_BASED % TRAJECTORIES_PER_LEVEL))

CURRENT_NOISE_STEP=${NOISE_LEVELS_TO_EXTEND[$NOISE_INDEX]}

echo "This task will extend Trajectory ${TRAJ_INDEX} for Noise Level ${CURRENT_NOISE_STEP}"

# Run the Python script, targeting the specific trajectory to extend
echo "Starting Python script..."
python sequential_uturns.py \
    --start_image_path "$START_IMAGE" \
    --num_uturns "$NEW_UTURNS_TO_ADD" \
    --noise_step "$CURRENT_NOISE_STEP" \
    --trajectory_idx "$TRAJ_INDEX" # This tells the script which one to extend
echo "Python script finished."

echo "Job finished at $(date)"