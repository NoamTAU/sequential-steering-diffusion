#!/bin/bash

#----------------------------------------------------------------#
# Slurm job directives.
#----------------------------------------------------------------#

#SBATCH --job-name=diffusion_fwd_back  # A name for your job
#SBATCH --output=diffusion_output_%j.out # File to save the standard output
#SBATCH --error=diffusion_error_%j.err   # File to save the standard error
#SBATCH --nodes=1                      # We need 1 node
#SBATCH --ntasks-per-node=1            # We need 1 task (our python script)
#SBATCH --cpus-per-task=4              # Request 4 CPU cores
#SBATCH --mem=32G                      # Request 32 Gigabytes of main RAM
#SBATCH --time=01:00:00                # Set a 1-hour time limit
#SBATCH --gres=gpu:1                   # Request 1 full GPU

# --- THIS IS THE FIX ---
#SBATCH --partition=h100               # <<< Specify the H100 partition

#----------------------------------------------------------------#
# The actual commands to run your experiment.
#----------------------------------------------------------------#

echo "Job started on $(hostname) at $(date)"

# 1. Load the conda module and activate your environment
echo "Loading Conda environment..."
source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics
echo "Environment 'llm_physics' activated."

# 2. Print GPU status to the output file for debugging
echo "Checking GPU status:"
nvidia-smi

# 3. Run your Python script
echo "Starting Python script..."
python dataset_sample-forw_back.py --num_samples 4 --batch_size 2
echo "Python script finished."

echo "Job finished at $(date)"