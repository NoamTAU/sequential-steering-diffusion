#!/bin/bash
#SBATCH --job-name=steer_dog2dog_prob_multi
#SBATCH --array=0-49
#SBATCH --output=slurm_logs/steer_dog2dog_prob_multi_%A_%a.out
#SBATCH --error=slurm_logs/steer_dog2dog_prob_multi_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

mkdir -p slurm_logs

IMAGE_LIST_FILE="scripts/image_list.txt"
IMG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$IMAGE_LIST_FILE")
OUT="/work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1_multi"
TARGET_PROB_STOP=0.90
RUN_TAG="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Image: $IMG"

if [ -z "$IMG" ]; then
    echo "No image found for index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

python scripts/steered_sequential_uturns_v4.py \
    --start_image_path "$IMG" \
    --output_dir "$OUT" \
    --orig_class_idx 0 \
    --target_class_idx 0 \
    --auto_orig_class True \
    --require_orig_in_dog_indices True \
    --auto_target_dog True \
    --auto_target_topk 5 \
    --num_steps 50 \
    --noise_step 100 \
    --timestep_respacing "250" \
    --score_mode target_prob \
    --penalty 0.0 \
    --batch_size 64 \
    --max_retries 4 \
    --require_target_increase True \
    --target_prob_stop ${TARGET_PROB_STOP} \
    --fail_behavior skip \
    --classifier_use_fp16 True \
    --run_tag "$RUN_TAG"
