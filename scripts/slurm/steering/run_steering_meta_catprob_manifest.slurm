#!/bin/bash
#SBATCH --job-name=steer_meta_catprob_missing
#SBATCH --output=slurm_logs/steer_meta_catprob_missing_%A_%a.out
#SBATCH --error=slurm_logs/steer_meta_catprob_missing_%A_%a.err
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

MANIFEST_FILE="${MANIFEST_FILE:?Missing MANIFEST_FILE}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LINE_NO=$((TASK_ID + 2))
ROW=$(sed -n "${LINE_NO}p" "$MANIFEST_FILE")

if [ -z "$ROW" ]; then
    echo "No manifest row at line $LINE_NO in $MANIFEST_FILE"
    exit 1
fi

IMG=$(echo "$ROW" | cut -d, -f1)
REPEAT_INDEX=$(echo "$ROW" | cut -d, -f4)
SEED=$(echo "$ROW" | cut -d, -f5)
OUT="/work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2_multi"
TARGET_PROB_STOP=0.90
RUN_TAG="${SLURM_JOB_ID}_${TASK_ID}_rep${REPEAT_INDEX}"

echo "Manifest line: $LINE_NO"
echo "Repeat Index: $REPEAT_INDEX"
echo "Seed: $SEED"
echo "Image: $IMG"

python scripts/steered_sequential_uturns_meta.py \
    --start_image_path "$IMG" \
    --output_dir "$OUT" \
    --num_steps 50 \
    --noise_step 100 \
    --timestep_respacing "250" \
    --score_mode cat_prob \
    --penalty 0.0 \
    --batch_size 64 \
    --max_retries 4 \
    --seed "$SEED" \
    --require_top1_dog True \
    --require_target_increase True \
    --target_prob_stop ${TARGET_PROB_STOP} \
    --fail_behavior skip \
    --classifier_use_fp16 True \
    --run_tag "$RUN_TAG"
