#!/bin/bash

#----------------------------------------------------------------#
# Slurm Job Array for Sequential U-Turn Grid Search
#----------------------------------------------------------------#

#SBATCH --job-name=uturn_grid_search
#SBATCH --array=1-9                  # MOVED UP: More robust for parsing
#SBATCH --output=slurm_logs/uturns_%A_%a.out
#SBATCH --error=slurm_logs/uturns_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1

# --- KEY PARAMETERS FOR YOUR EXPERIMENT ---
NOISE_LEVELS=(25 50 75 100 150 200 300 400 500)


#----------------------------------------------------------------#
# Commands to run the experiment for EACH array task
#----------------------------------------------------------------#

echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job started on $(hostname) at $(date)"

# Activate Conda environment
echo "Loading Conda environment..."
source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics
echo "Environment 'llm_physics' activated."

# Check GPU status
echo "Checking GPU status:"
nvidia-smi

# --- Map the Slurm Task ID to a Noise Level ---
# Slurm task IDs are 1-based, but bash arrays are 0-based.
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CURRENT_NOISE_STEP=${NOISE_LEVELS[$INDEX]}
echo "This task will run with --noise_step = $CURRENT_NOISE_STEP"

# --- Define Your Other Parameters Here ---
START_IMAGE="/work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG"
NUM_UTURNS=100
NUM_TRAJECTORIES=50 # Run 50 random trajectories for each noise level

# Run the Python script with the dynamically assigned noise step
echo "Starting Python script..."
python sequential_uturns.py \
    --start_image_path "$START_IMAGE" \
    --num_uturns "$NUM_UTURNS" \
    --noise_step "$CURRENT_NOISE_STEP" \
    --num_trajectories "$NUM_TRAJECTORIES"
echo "Python script finished."

echo "Job finished at $(date)"