#!/bin/bash

#----------------------------------------------------------------#
# Slurm Job Array for Debug Sweep (Noise Levels) - Respaced Version
#----------------------------------------------------------------#

#SBATCH --job-name=debug_sweep_v2
#SBATCH --array=0-9                    # 10 Noise levels
#SBATCH --output=slurm_logs/debug_v2_%A_%a.out
#SBATCH --error=slurm_logs/debug_v2_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00                # Fast timeout (1 U-turn is very fast)
#SBATCH --partition=h100
#SBATCH --gres=gpu:1

# --- CONFIGURATION ---

# 1. The specific image you are testing
START_IMAGE="/work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG"

# 2. The noise levels to test
NOISE_LEVELS=(25 50 75 100 125 150 175 200 225 250)

# 3. Experiment settings
NUM_UTURNS=1           # Just one U-turn
NUM_TRAJECTORIES=10    # 10 realizations
RESPACING="250"        # IMPORTANT: Using the 250 step schedule

# 4. OUTPUT DIRECTORY (New separate folder)
DEBUG_OUTPUT_DIR="/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns_debug_respaced"

#----------------------------------------------------------------#
# Execution
#----------------------------------------------------------------#

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"

# Get the noise step for this task
CURRENT_NOISE_STEP=${NOISE_LEVELS[$SLURM_ARRAY_TASK_ID]}
echo "Running with Noise Step: $CURRENT_NOISE_STEP"

# Load Environment
source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

# Run Script with EXPLICIT output directory
echo "Starting Python script..."
python sequential_uturns.py \
    --start_image_path "$START_IMAGE" \
    --num_uturns "$NUM_UTURNS" \
    --noise_step "$CURRENT_NOISE_STEP" \
    --num_trajectories "$NUM_TRAJECTORIES" \
    --timestep_respacing "$RESPACING" \
    --output_dir "$DEBUG_OUTPUT_DIR"

echo "Job finished."