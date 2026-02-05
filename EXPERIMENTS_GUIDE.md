# Sequential Steering Diffusion — Commands & Experiments

This file summarizes the scripts, Slurm jobs, and notebooks in this repo, plus example commands to launch each experiment. It is tailored to the PCSL/SSH paths and the `llm_physics` conda environment.

## 0) Assumptions and Common Paths

Common defaults used throughout scripts and notebooks:
- Conda env on SSH: `llm_physics`
- Unconditional 256x256 model checkpoint:
  - `/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt`
- Example image (husky):
  - `/work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG`
- Sequential U-turn outputs (default):
  - `/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns`
- Steering outputs (default):
  - `/work/pcsl/Noam/sequential_diffusion/results/steering_force_analysis`
- Manifold probe outputs (default):
  - `/work/pcsl/Noam/sequential_diffusion/results/manifold_probe`

If you change any of these, keep the SSH-side structure consistent to avoid clashes.

## 1) Repo Sync and Plot Download Helpers

**Sync (pull, commit, push):**
```bash
bash sync_repo.sh "Your commit message"
```
This script will `git pull --rebase`, auto-stash if dirty, then commit and push.

**Download plots from SSH:**
```bash
bash download_plots.sh nlevi@pcsl ~/Downloads/seq_plots
```
This copies from `/work/pcsl/Noam/sequential_diffusion/results/plots` via `rsync` or `scp`.

## 2) Core Generation: Sequential U-turns (Unguided)

**Primary script:** `scripts/sequential_uturns.py`
- Produces image trajectories and CLIP patch embeddings.
- Resumes automatically if a trajectory folder already has `uturn_*.jpeg`.

Example command:
```bash
python scripts/sequential_uturns.py \
  --start_image_path /work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG \
  --num_uturns 100 \
  --noise_step 100 \
  --num_trajectories 50 \
  --timestep_respacing 250 \
  --output_dir /work/pcsl/Noam/sequential_diffusion/results/sequential_uturns
```

Key outputs:
- `uturn_000.jpeg`, `uturn_001.jpeg`, …
- `trajectory_data.npz` (CLIP patch embeddings)

**Alternative/legacy:** `scripts/sequential_uturns_revised.py`, `scripts/sequential_uturns_old.py`
- `sequential_uturns_revised.py` includes manual diffusion loop and robust resume logic.

**Extend an existing trajectory:**
- Use `--trajectory_idx` to target one trajectory.
- Example:
```bash
python scripts/sequential_uturns.py \
  --start_image_path /work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG \
  --num_uturns 100 \
  --noise_step 50 \
  --trajectory_idx 7
```

## 3) Guided Steering (Dogs → Cats or Specific Classes)

**Steer between two specific ImageNet classes:** `scripts/steered_sequential_uturns_v4.py`
- Uses ConvNeXt classifier to select the best proposal.
- Scoring: `target_logit - penalty * orig_logit`
- Saves `steering_data.npz` and `force_stats.npz`

Example:
```bash
python scripts/steered_sequential_uturns_v4.py \
  --start_image_path /work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG \
  --output_dir /work/pcsl/Noam/sequential_diffusion/results/steering_force_analysis \
  --orig_class_idx 249 \
  --target_class_idx 281 \
  --num_steps 30 \
  --noise_step 100 \
  --timestep_respacing 250 \
  --penalty 1.0 \
  --batch_size 64 \
  --max_retries 4
```

**Meta-class steering (Dogs vs Cats):** `scripts/steered_sequential_uturns_meta.py`
- Scores: `logsumexp(cats) - penalty * logsumexp(dogs)`

Example:
```bash
python scripts/steered_sequential_uturns_meta.py \
  --start_image_path /work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG \
  --output_dir /work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2 \
  --num_steps 50 \
  --noise_step 100 \
  --penalty 1.0 \
  --batch_size 64 \
  --max_retries 4
```

## 4) Manifold Probe (Unguided + Full Logits)

**Script:** `scripts/probe_manifold.py`
- Runs unguided U-turns.
- Saves full 1000-class logits per step (`manifold_logits.npz`).

Example:
```bash
python scripts/probe_manifold.py \
  --start_image_path /work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG \
  --output_dir /work/pcsl/Noam/sequential_diffusion/results/manifold_probe \
  --trajectory_idx 0 \
  --num_steps 200 \
  --noise_step 40
```

## 5) Slurm Job Templates (H100)

These are ready-to-run job files. Edit the parameter block at the top as needed.

- `scripts/run_sequential.slurm`
  - Single sequential run (one image, one noise).
  - `sbatch scripts/run_sequential.slurm`

- `scripts/run_grid_search.slurm`
  - Array over noise levels for one image.
  - `sbatch scripts/run_grid_search.slurm`

- `scripts/run_production.slurm`
  - Array over images and noise levels using `scripts/image_list.txt`.
  - `sbatch scripts/run_production.slurm`

- `scripts/run_single_uturn_baseline.slurm`
  - One-step baseline across images/noise.
  - `sbatch scripts/run_single_uturn_baseline.slurm`

- `scripts/run_debug_sweep.slurm`
  - Debug sweep with respacing (fast single U-turn).
  - `sbatch scripts/run_debug_sweep.slurm`

- `scripts/extend_trajectories.slurm`
  - Extend existing trajectories by N U-turns.
  - `sbatch scripts/extend_trajectories.slurm`

- `scripts/run_steering_experiment.slurm`
  - Guided steering (dogs → cats example).
  - `sbatch scripts/run_steering_experiment.slurm`

- `scripts/run_manifold_probe.slurm`
  - Unguided probe with full logits.
  - `sbatch scripts/run_manifold_probe.slurm`

- `scripts/run_experiment.slurm`
  - Forward/back dataset sampling (`dataset_sample-forw_back.py`).

All Slurm scripts assume:
- `conda activate llm_physics`
- H100 partition
- Paths under `/work/pcsl/Noam/...`

## 6) Classifier Stats and Debugging

**Compute activation stats for whitening:** `scripts/classifier_statistics.py`
```bash
python scripts/classifier_statistics.py \
  --classifier_name convnext_base \
  --data_dir /work/pcsl/Noam/diffusion_datasets/selected_images \
  --num_samples 100 \
  --batch_size 10 \
  --output classifier_statistics
```

**Debug classifier outputs on a trajectory:** `scripts/debug_classifier_behavior.py`
```bash
python scripts/debug_classifier_behavior.py \
  --classifier_name convnext_base \
  --traj_dir /work/pcsl/Noam/sequential_diffusion/results/sequential_uturns/ILSVRC2012_val_00000729/noise_step_100/trajectory_000 \
  --steps 0 1 5 10 25 50 100
```

## 7) Evaluation of Sequential Trajectories

**Layer drift on one image, one noise:** `scripts/evaluate_sequential_classifier.py`
```bash
python scripts/evaluate_sequential_classifier.py \
  --data_dir /work/pcsl/Noam/sequential_diffusion/results/sequential_uturns \
  --image_name ILSVRC2012_val_00000729 \
  --noise_step 75 \
  --output_base sequential_analysis_results
```

**All images, one noise:** `scripts/evaluate_all_images.py`
```bash
python scripts/evaluate_all_images.py \
  --data_dir /work/pcsl/Noam/sequential_diffusion/results/sequential_uturns \
  --noise_step 100 \
  --output_base sequential_analysis_results
```

**All images, all noises:** `scripts/evaluate_all_images_all_noises.py`
```bash
python scripts/evaluate_all_images_all_noises.py \
  --data_dir /work/pcsl/Noam/sequential_diffusion/results/sequential_uturns \
  --output_base sequential_analysis_results
```

**Paired start/final evaluation (visual_examples structure):**
- `scripts/evaluate_classifier_paired.py`
- `scripts/evaluate_classifier_paired_debug.py` (debug for sequential single-step)

**Single U-turn debug analysis:** `scripts/evaluate_classifier_debug_single_uturn.py`

**Generic classifier eval across t-steps:** `scripts/evaluate_classifier.py`
- Reads data in `t_<step>_<respacing>_images` format.

## 8) CLIP and Correlation Analyses

**CLIP patch correlations:**
- `scripts/clip_correlations.py`
- `scripts/clip_correlations_seed_trajectory.py`

**Pixel correlation matrices:**
- `scripts/evaluate_vector_correlations.py`
- `scripts/evaluate_delta_magn_correlations.py`

**Correlation function from saved matrices:**
- `scripts/compute_correlation_function.py`

**Fourier / Wavelet coefficients:**
- `scripts/fourier_coefficients.py`
- `scripts/wavelet_coefficients.py` (depends on external `local_probability_models_of_images`)

Note: These scripts rely on the `t_<step>_<respacing>_images` dataset layout produced by sampling scripts.

## 9) Sampling, Training, and Baselines (Standard Guided Diffusion)

These are mostly standard OpenAI guided-diffusion scripts. See `README.md` for full flag sets.

- `scripts/image_sample.py`
- `scripts/classifier_sample.py`
- `scripts/image_sample-forward_backward.py`
- `scripts/image_sample-forw_back-classifier.py`
- `scripts/forward_backward_dataset_sample.py`
- `scripts/dataset_sample-forw_back.py`
- `scripts/image_train.py`
- `scripts/classifier_train.py`
- `scripts/super_res_sample.py`
- `scripts/super_res_train.py`
- `scripts/image_nll.py`

Tip: run `python scripts/<name>.py --help` to see full flags.

## 10) Notebooks (Analysis and Plotting)

- `notebooks/plot_generation_sequential.ipynb`
  - Main plotting notebook for paper figures.
  - Susceptibility, C2(t), latent cosine vs normalized U-turns, guided vs unguided comparisons, force/potential plots.
  - Uses SSH paths under `/work/pcsl/Noam/...`.
  - Saves figures and exports to `notebooks/_exports`.

- `notebooks/trajectories_visualization_sequential.ipynb`
  - Original sequential U-turn analysis notebook (covariance, steering, force definition).

- `notebooks/survival_probs.ipynb`
  - Survival probability analysis and latent separation.

- `notebooks/analysis_correlations.ipynb`
  - Correlation function analysis (spin-like C2, susceptibility).

- `notebooks/trajectories_visualization.ipynb` and `trajectories_visualization_noam.ipynb`
  - Earlier visualization and dataset analysis.

- `notebooks/test_clip*.ipynb`, `test_correlations.ipynb`, `test_wavelets.ipynb`
  - Scratch/test notebooks for CLIP and correlation code.

## 11) Quick Reference: Result File Patterns

Sequential U-turns:
```
<output_dir>/<image_name>/noise_step_<t>/trajectory_<idx>/
  uturn_000.jpeg
  uturn_001.jpeg
  ...
  trajectory_data.npz
```

Steering:
```
<output_dir>/<image_name>/target_<class>/noise_<t>_p<penalty>_b<batch>_r<retries>/
  step_000.jpeg
  step_001.jpeg
  ...
  steering_data.npz
  force_stats.npz
```

Manifold probe:
```
<output_dir>/<image_name>/probe_noise_<t>/traj_<idx>/
  step_000.jpeg
  ...
  manifold_logits.npz
```

## 12) Where to Look for Flags

Many scripts use `guided_diffusion.script_util.add_dict_to_argparser`. The fastest way to see full flags is:
```bash
python scripts/<script_name>.py --help
```

This is the canonical source of truth if you need to update a launch command.
