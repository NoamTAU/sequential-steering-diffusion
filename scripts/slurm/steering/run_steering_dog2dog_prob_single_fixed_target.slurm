#!/bin/bash
#SBATCH --job-name=steer_dog2dog_single
#SBATCH --output=slurm_logs/steer_dog2dog_single_%A_%a.out
#SBATCH --error=slurm_logs/steer_dog2dog_single_%A_%a.err
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

IMG="${IMG:?Need IMG=/full/path/to/image.JPEG}"
ORIG_CLASS_IDX="${ORIG_CLASS_IDX:?Need ORIG_CLASS_IDX=<classifier index>}"
TARGET_CLASS_IDX="${TARGET_CLASS_IDX:?Need TARGET_CLASS_IDX=<classifier index>}"
OUT="${OUT:-/work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1_multi}"
BASE_SEED="${BASE_SEED:-20260408}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$((BASE_SEED + TASK_ID))
RUN_TAG="${SLURM_JOB_ID}_${TASK_ID}_rep${TASK_ID}"
TARGET_PROB_STOP="${TARGET_PROB_STOP:-0.90}"

echo "Image: $IMG"
echo "Task ID: $TASK_ID"
echo "Seed: $SEED"
echo "Fixed source: $ORIG_CLASS_IDX"
echo "Fixed target: $TARGET_CLASS_IDX"

if [ "$ORIG_CLASS_IDX" -eq "$TARGET_CLASS_IDX" ]; then
  echo "ORIG_CLASS_IDX and TARGET_CLASS_IDX must differ."
  exit 1
fi

python scripts/steered_sequential_uturns_v4.py \
    --start_image_path "$IMG" \
    --output_dir "$OUT" \
    --orig_class_idx "$ORIG_CLASS_IDX" \
    --target_class_idx "$TARGET_CLASS_IDX" \
    --auto_orig_class False \
    --require_orig_in_dog_indices True \
    --auto_target_dog False \
    --num_steps 50 \
    --noise_step 100 \
    --timestep_respacing "250" \
    --score_mode target_prob \
    --penalty 0.0 \
    --batch_size 64 \
    --max_retries 4 \
    --seed "$SEED" \
    --require_target_increase True \
    --target_prob_stop ${TARGET_PROB_STOP} \
    --fail_behavior skip \
    --classifier_use_fp16 True \
    --run_tag "$RUN_TAG"
