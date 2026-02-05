#!/bin/bash

#----------------------------------------------------------------#
# Slurm job directives for a sequential U-turn experiment
#----------------------------------------------------------------#

#SBATCH --job-name=seq_uturns_husky  # Descriptive job name
#SBATCH --output=seq_uturns_husky_%j.out # Standard output file
#SBATCH --error=seq_uturns_husky_%j.err   # Standard error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00                # Increased time limit to 2 hours for all trajectories
#SBATCH --gres=gpu:1
#SBATCH --partition=h100

#----------------------------------------------------------------#
# Commands to run the experiment
#----------------------------------------------------------------#

echo "Job started on $(hostname) at $(date)"

# Activate Conda environment
echo "Loading Conda environment..."
source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics
echo "Environment 'llm_physics' activated."

# Check GPU status
echo "Checking GPU status:"
nvidia-smi

# Define the path to your starting image
# This is a husky image
START_IMAGE="/work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG"

# Run the Python script with your desired parameters
echo "Starting Python script for sequential U-turns..."
python sequential_uturns.py \
    --start_image_path "$START_IMAGE" \
    --num_uturns 100 \
    --noise_step 300 \
    --num_trajectories 1
echo "Python script finished."

echo "Job finished at $(date)"