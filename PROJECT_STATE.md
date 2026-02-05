# Sequential-Steering-Diffusion — Project Memory

## High-Level Goal (My Understanding)
This repo is a fork of OpenAI’s guided-diffusion codebase, extended to study **sequential forward–backward (“U‑turn”) dynamics** on images and to **steer those trajectories** using a classifier. The central idea is:
- Start from a real image.
- Add noise to a chosen timestep (forward diffusion).
- Denoise back to a clean image (reverse diffusion).
- Repeat this process sequentially to generate a trajectory of images.
- Analyze how image semantics drift over many U‑turns (CLIP embeddings, classifier logits).
- Optionally *steer* the trajectory by sampling multiple candidate U‑turns and selecting the one that best increases a target class (while penalizing the original class).

## Key Scripts (Core Flow)
- `scripts/sequential_uturns.py`  
  Runs sequential U‑turns on a single starting image, saves `uturn_###.jpeg` and CLIP patch embeddings. Supports resuming trajectories.

- `scripts/sequential_uturns_revised.py`  
  Updated, robust version. **Manual loop replaces missing `p_sample_loop_forw_back`**, more reliable resume, incremental saving of embeddings after every step. Likely the preferred runner.

- `scripts/steered_sequential_uturns_v4.py`  
  Steering with a classifier (ConvNeXt). At each step:
  - Generate a batch of U‑turn proposals.
  - Score each as: `target_logit − penalty * orig_logit`.
  - Accept the best; retry up to `max_retries` to find improvement.
  - Saves images + `steering_data.npz` + `force_stats.npz`.

- `scripts/steered_sequential_uturns_meta.py`  
  “Meta‑class” steering (e.g., **cats vs dogs**): score uses log‑sum‑exp over class groups.

- `scripts/probe_manifold.py`  
  Runs unsteered U‑turns and logs **full classifier logits at every step** to probe the semantic manifold.

## Analysis / Evaluation Scripts
- `scripts/evaluate_sequential_classifier.py`  
  Computes layer‑wise activation drift across U‑turn steps (normalized L2 + cosine); optional whitening from stats.

- `scripts/clip_correlations_seed_trajectory.py`  
  Computes CLIP patch‑embedding correlations and correlation functions across trajectories.

- Other analysis helpers:
  - `scripts/evaluate_all_images.py`, `scripts/evaluate_all_images_all_noises.py`
  - `scripts/compute_correlation_function.py`, `scripts/evaluate_delta_magn_correlations.py`
  - `scripts/fourier_coefficients.py`, `scripts/wavelet_coefficients.py`
  - `scripts/debug_classifier_behavior.py`

## Model / Classifier
- Diffusion model: **256x256 unconditional** (default path hardcoded in scripts).
- Classifier: **ConvNeXt‑Base** (`guided_diffusion/torch_classifiers.py`).
- Many scripts assume **ImageNet 1000‑class indexing**.

## Output Structure (Typical)
Sequential U‑turns:
```
output_dir/<image_name>/noise_step_<t>/trajectory_<idx>/
  uturn_000.jpeg
  uturn_001.jpeg
  ...
  trajectory_data.npz   # CLIP patch embeddings (B, 49, 768)
```

Steering:
```
output_dir/<image_name>/target_<class>/noise_<t>_p<penalty>_b<batch>_r<retries>/
  step_000.jpeg
  step_001.jpeg
  ...
  steering_data.npz
  force_stats.npz
```

## HPC / Slurm Assumptions
Slurm scripts in `scripts/` run on an H100 partition and assume a conda env `llm_physics`. They reference paths like `/work/pcsl/Noam/...`. These need updating for local runs.

## Notes / Potential Pitfalls
- Several scripts call `diffusion.p_sample_loop_forw_back`, but that function **is not present** in `guided_diffusion/gaussian_diffusion.py`. The revised U‑turn script avoids this by manually looping.
- Many hardcoded paths point to HPC locations; adapt when running locally.
- `README.md` is still the vanilla guided‑diffusion README (not specific to sequential steering).

## Next Step When Resuming
Decide which experiment we want to run first:
1. **Unsteered U‑turns** (baseline drift + CLIP embeddings), or
2. **Steered U‑turns** (classifier‑guided trajectory), or
3. **Analysis pass** over existing trajectories.

## Recent Updates
- Added a clean plotting notebook: `notebooks/plot_generation_sequential.ipynb`.
  - Generates a horizontal “cartoon” of sequential U‑turns.
  - Uses hardcoded SSH paths under `/work/pcsl/Noam/...`.
  - Optional auto‑selection of a “nice” trajectory using a drift/smoothness proxy from saved CLIP patch embeddings.
- Added a discovery cell in `notebooks/plot_generation_sequential.ipynb` to scan available images, noise levels, trajectory counts, and approximate max U‑turns.
- Added susceptibility + C2 analysis section in `notebooks/plot_generation_sequential.ipynb` (single-image and all-images averages, multiple noise levels). Uses CLIP patch embeddings from `trajectory_data.npz`, optional flip-aware matching, and KMeans discretization for C2.
- Added error bars: bootstrap SEM across trajectories for susceptibility (single image), SEM across trajectories for C2, and SEM across images for all-images plots.
- Updated plots to use shaded SEM bands (no error bars), and label noise levels as ρ = t/tmax.
- Enforced consistent max U-turn length across noise levels for single-image plots (and optional for all-images).
- Added speedups for all-images C2 via optional global codebooks and sampling limits.
- Added notebook export helpers and download links in `notebooks/plot_generation_sequential.ipynb`:
  - Copies saved figures to `notebooks/_exports` (or `NOTEBOOK_EXPORT_DIR`).
  - Displays clickable download links in JupyterLab via `FileLink`.
- Added `download_plots.sh` helper to fetch `/work/pcsl/Noam/sequential_diffusion/results/plots` to local machine via rsync/scp.
- Added `sync_repo.sh` to pull (rebase), commit, and push in one command.
