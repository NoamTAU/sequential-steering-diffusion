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

### Paper / Overleaf Repo (Nested Git Repo)
This code repo contains a nested git repo used for the paper and Overleaf sync:
- Paper repo path: `-ICML2026-MC_diffusion/`
- Overleaf remote is named `overleaf` and uses branch `master`.
- Local branch is typically `main` and should be pushed to `overleaf/master`.
- As of the latest sync, the active paper source is `NeurIPS2026/main_tracked_1.tex`.
- The older `NeurIPS2026/main_noam.tex` source was removed upstream and the previous draft tree now lives under `old_main/`.

Typical workflow:
```bash
cd -ICML2026-MC_diffusion
git pull --rebase overleaf master
git push overleaf main:master
```
If the push is rejected, it usually means Overleaf has new commits; always pull/rebase first.

### Overleaf UI Comments
Overleaf web UI comments are **not** stored in the Git repo, so they will not appear locally after `git pull`. Only literal `.tex` content (including comment text written into the files) is versioned.

### Fix: `cannot lock ref 'refs/remotes/overleaf/master' ... expected ...`
Sometimes `git pull --rebase overleaf master` fails while updating the remote-tracking ref, e.g.
`error: cannot lock ref 'refs/remotes/overleaf/master': is at <sha> but expected <sha>`.

This is usually a stale lock file or an interrupted Git operation. In the Overleaf repo root:
```bash
rm -f .git/refs/remotes/overleaf/master.lock .git/packed-refs.lock
git fetch overleaf +master:refs/remotes/overleaf/master
git merge --ff-only overleaf/master
```
After that, `git pull --rebase overleaf master` should work again.

**Download plots from SSH:**
```bash
bash download_plots.sh nlevi@pcsl ~/Downloads/seq_plots
```
This copies from `/work/pcsl/Noam/sequential_diffusion/results/plots` via `rsync` or `scp`.

### SSH Update + Notebook Merge Conflicts (Practical Notes)
On the cluster we often update the repo using:
```bash
bash update_with_ipynb.sh
```
This stashes local notebook edits, pulls, and then pops the stash back.

If `notebooks/plot_generation_sequential.ipynb` conflicts during pull/rebase:
- If you *don’t* care about the cluster-side notebook edits: keep the remote version.
```bash
git checkout origin/main -- notebooks/plot_generation_sequential.ipynb
git add notebooks/plot_generation_sequential.ipynb
git rebase --continue  # only if you are mid-rebase
```
- If you *do* care about the notebook edits: resolve in Jupyter (or via `nbdime` if installed), then `git add` and continue.

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

### High-Noise Latent Sweep

For the ergodicity / latent-decorrelation paper, the current question is whether increasing the per-U-turn masking fraction changes the relative survival of low- vs high-level ConvNeXt latents.

The intended workflow is:
1. generate unguided sequential trajectories at larger `noise_step`
2. evaluate ConvNeXt latent cosine survival on those trajectories
3. use the notebook section `High-Noise Latent Regime Comparison` to make:
   - a row of same-observable latent-survival plots across noise
   - an AUC-based early-vs-late layer summary vs noise

**Submit a high-noise sweep over an image list:**
```bash
cd /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

IMAGE_LIST_FILE=/work/pcsl/Noam/sequential_diffusion/metadata/high_noise_image_list.txt \
NOISE_LEVELS_CSV=300,400,500 \
NUM_TRAJECTORIES=20 \
NUM_UTURNS=100 \
bash scripts/slurm/sequential/submit_high_noise_latent_sweep.sh
```

Notes:
- `IMAGE_LIST_FILE` can be any newline-separated list of image paths.
- `NOISE_LEVELS_CSV` should usually be the high-noise regime you want to test, e.g. `300,400,500` or `300,400,500,600`.
- the helper computes the Slurm array size automatically from `(num_images x num_noise_levels)`.

**Evaluate ConvNeXt latents only on that sweep:**
```bash
cd /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

sbatch \
  --export=ALL,DATA_DIR=/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns,OUTPUT_BASE=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/sequential_analysis_results,STATS_DIR=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/sequential_analysis_results,IMAGE_LIST_FILE=/work/pcsl/Noam/sequential_diffusion/metadata/high_noise_image_list.txt,NOISE_LEVELS_CSV=300,400,500 \
  scripts/slurm/sequential/run_high_noise_latent_eval.slurm
```

Under the hood this runs:
```bash
python scripts/evaluate_all_images_all_noises.py \
  --data_dir /work/pcsl/Noam/sequential_diffusion/results/sequential_uturns \
  --output_base /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/sequential_analysis_results \
  --stats_dir /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/sequential_analysis_results \
  --image_list /work/pcsl/Noam/sequential_diffusion/metadata/high_noise_image_list.txt \
  --noise_steps 300 400 500
```

`evaluate_all_images_all_noises.py` now accepts:
- `--image_list`: restrict evaluation to a newline-separated list of image paths
- `--noise_steps`: restrict evaluation to a specific subset of noise levels
- `--max_images`: optional cap for quick tests

**Check whether generation + latent evaluation are complete:**
```bash
python scripts/check_high_noise_latent_status.py \
  --image-list /work/pcsl/Noam/sequential_diffusion/metadata/high_noise_image_list.txt \
  --results-root /work/pcsl/Noam/sequential_diffusion/results/sequential_uturns \
  --analysis-root /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/sequential_analysis_results \
  --noise-steps 400 600 800 \
  --expected-trajectories 20 \
  --expected-uturns 100
```

This reports, for each `(image, noise)` pair:
- whether the expected number of trajectory directories exists
- whether those trajectories reached `uturn_100`
- whether `sequential_activations_v2.pk` has been produced

**Build a rerun manifest for missing / incomplete generation pairs:**
```bash
python scripts/build_high_noise_missing_manifest.py \
  --image-list /work/pcsl/Noam/sequential_diffusion/metadata/high_noise_image_list.txt \
  --results-root /work/pcsl/Noam/sequential_diffusion/results/sequential_uturns \
  --noise-steps 400 600 800 \
  --expected-trajectories 20 \
  --expected-uturns 100 \
  --out-csv /work/pcsl/Noam/sequential_diffusion/metadata/high_noise_missing_manifest.csv
```

**Rerun only those missing / incomplete pairs:**
```bash
MANIFEST_CSV=/work/pcsl/Noam/sequential_diffusion/metadata/high_noise_missing_manifest.csv \
NUM_TRAJECTORIES=20 \
NUM_UTURNS=100 \
BACKUP_PARTIAL=1 \
bash scripts/slurm/sequential/submit_high_noise_latent_manifest.sh
```

This manifest workflow is important because `scripts/sequential_uturns.py` treats `--num_uturns` as *new* U-turns to add on top of an existing trajectory. For incomplete pilot pairs we therefore rerun the whole `(image, noise)` pair from scratch after backing up the partial directory, rather than resuming in place.

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

**Target-probability steering (no dog-vs-cat penalty):**
```bash
python scripts/steered_sequential_uturns_v4.py \
  --start_image_path /work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG \
  --output_dir /work/pcsl/Noam/sequential_diffusion/results/steering_force_analysis \
  --orig_class_idx 249 \
  --target_class_idx 281 \
  --num_steps 30 \
  --noise_step 100 \
  --timestep_respacing 250 \
  --score_mode target_prob \
  --batch_size 64 \
  --max_retries 4 \
  --require_target_increase True \
  --fail_behavior skip
```

**Auto-select a target dog class (dog→dog):**
```bash
python scripts/steered_sequential_uturns_v4.py \
  --start_image_path /work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG \
  --output_dir /work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1 \
  --orig_class_idx 249 \
  --target_class_idx 0 \
  --auto_target_dog True \
  --auto_target_topk 5 \
  --num_steps 50 \
  --noise_step 100 \
  --timestep_respacing 250 \
  --penalty 1.0 \
  --batch_size 64 \
  --max_retries 4
```

**Dog→dog with target-probability steering:**
```bash
python scripts/steered_sequential_uturns_v4.py \
  --start_image_path /work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG \
  --output_dir /work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1 \
  --orig_class_idx 249 \
  --target_class_idx 0 \
  --auto_target_dog True \
  --auto_target_topk 5 \
  --num_steps 50 \
  --noise_step 100 \
  --timestep_respacing 250 \
  --score_mode target_prob \
  --batch_size 64 \
  --max_retries 4 \
  --require_target_increase True \
  --fail_behavior skip
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

**Meta-class with cat probability only (no dog penalty):**
```bash
python scripts/steered_sequential_uturns_meta.py \
  --start_image_path /work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG \
  --output_dir /work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2 \
  --num_steps 50 \
  --noise_step 100 \
  --score_mode cat_prob \
  --batch_size 64 \
  --max_retries 4 \
  --require_target_increase True \
  --fail_behavior skip \
  --target_prob_stop 0.9
```

**Build a master image list first:**
- `scripts/image_list.txt` is only the existing curated 50-image subset under `selected_images`
- for a real multi-image steering study, build a list from the full ImageNet validation directory on the cluster
```bash
python scripts/build_image_list.py \
  --image-dir /path/to/ILSVRC2012_val \
  --out scripts/imagenet_val_image_list.txt \
  --pattern '*.JPEG'
```

**Then build the dog-only list from that master list:**
```bash
python scripts/build_dog_image_list.py \
  --image-list scripts/imagenet_val_image_list.txt \
  --dog-list-out scripts/dog_image_list.txt \
  --summary-csv-out scripts/dog_image_summary.csv \
  --ground-truth-file /path/to/ILSVRC2012_validation_ground_truth.txt \
  --require-true-dog \
  --classifier-use-fp16
```

This tells you how many relevant dog-start images you actually have in the full validation set.

If you want to be stricter and keep only images whose classifier top-1 also agrees with the true ImageNet label, add:
```bash
  --require-classifier-match
```

Recommended policy for the paper:
- select images by true ImageNet validation label (`--require-true-dog`)
- optionally report classifier agreement statistics from `scripts/dog_image_summary.csv`
- only use `--require-classifier-match` if we decide we want a cleaner but potentially smaller subset

To avoid rebuilding local metadata inside the git repo after every pull, prefer syncing it once into the cluster work directory:
```bash
bash scripts/sync_steering_metadata.sh \
  /work/pcsl/Noam/sequential_diffusion/metadata \
  /work/pcsl/Noam/diffusion_datasets/all_images \
  /work/pcsl/imagenet/imagenet1k/devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt \
  100 \
  20260407
```

This creates:
- `/work/pcsl/Noam/sequential_diffusion/metadata/imagenet_val_image_list.txt`
- `/work/pcsl/Noam/sequential_diffusion/metadata/dog_image_summary.csv`
- `/work/pcsl/Noam/sequential_diffusion/metadata/dog_image_list_strict.txt`
- `/work/pcsl/Noam/sequential_diffusion/metadata/dog_image_list_strict_100.txt`

The notebook and example-selector now prefer these external metadata files automatically when they exist.

**Build a reproducible strict subset for the steering study:**
```bash
python scripts/sample_image_list.py \
  --input scripts/dog_image_list_strict.txt \
  --output scripts/dog_image_list_strict_100.txt \
  --num-images 100 \
  --seed 20260407
```

Recommended pilot for the image section:
- `100` strict images
- `4` steering trajectories per image
- same seed pairing across `dog->cat` and `dog->dog` for each `(image, repeat)` pair
- total jobs: `100 * 4 * 2 = 800`

**Multi-image meta-class steering (dog$\rightarrow$cat, probability only):**
- Slurm array over the filtered `scripts/dog_image_list.txt`

**Multi-image within-class steering (dog$\rightarrow$dog, probability only):**
- Original class is auto-detected from the classifier top-1 prediction
- The repeated-trajectory multi-image jobs are submitted together with:
```bash
bash scripts/slurm/steering/submit_multi_image_steering.sh scripts/dog_image_list_strict_100.txt 4 20260407
```

This submit helper:
- counts the relevant dog-start images in the chosen list
- sets the exact Slurm array size
- launches both `dog$\rightarrow$cat` and `dog$\rightarrow$dog` jobs
- uses deterministic paired seeds across the two regimes for each `(image, repeat)` pair
- uses `REPEATS=4` in the example above, meaning 4 steering trajectories per image per regime

**Aggregate steering runs for notebook analysis:**
```bash
python scripts/summarize_steering_runs.py \
  --root /work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2_multi \
  --out-csv /work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2_multi/steering_summary.csv

python scripts/summarize_steering_runs.py \
  --root /work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1_multi \
  --out-csv /work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1_multi/steering_summary.csv
```

The summary script identifies images from the run directory layout, so if the multi-image run structure changes, rebuild the CSVs before using the notebook.

**Check whether the `100 x 4` pilot is actually complete:**
```bash
python scripts/check_steering_pilot_status.py \
  --active-image-list scripts/dog_image_list_strict_100.txt \
  --meta-csv /work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2_multi/steering_summary.csv \
  --dog-csv /work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1_multi/steering_summary.csv \
  --expected-repeats 4 \
  --require-repeat-index
```

This reports:
- how many active pilot images are present
- how many images have any `dog->cat` runs
- how many image/source/target triples have any clean `dog->dog` runs
- how many images / triples have the full repeat count
- how many images have the full repeat count in both regimes
- the current best clean matched example image/source/target triple

Use this before trusting the notebook tables. If the `common images with >=4 repeats in both regimes` count is well below `100`, the pilot is not complete yet.

**Resume only the missing pilot tasks instead of resubmitting everything:**
```bash
bash scripts/slurm/steering/resume_multi_image_steering.sh \
  scripts/dog_image_list_strict_100.txt \
  4 \
  20260407 \
  /work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2_multi/steering_summary.csv \
  /work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1_multi/steering_summary.csv \
  /work/pcsl/Noam/sequential_diffusion/metadata/manifests
```

This builds two manifests under shared storage and only relaunches missing `(image, repeat)` tasks.
Do not use `/tmp` for these manifests, because compute nodes may not see the login node's local `/tmp`.

The plotting notebook now contains a dedicated section for these multi-image summaries:
- `## Multi-image Steering Statistics (Paper-Ready)`
- it will prefer `scripts/imagenet_val_image_list.txt` when present and otherwise fall back to `scripts/image_list.txt`
- it can also use the official ImageNet validation ground-truth file when available on the cluster
- it will build the dog-only image list if missing
- it will reuse existing run-summary CSVs when present
- otherwise it rebuilds them from the saved steering runs before plotting
- it aggregates in two stages:
  - within each image across repeated steering trajectories
  - then across images for the final comparison
- the paper-facing summary is a one-row violin plot over image-level means with short panel labels, 10pt-style PDF text, and black mean±SEM markers

The notebook also contains:
- `## Per-Image Averaged Steering Curves`
- by default it searches only the recent multi-image roots
- older single-image roots can be re-enabled manually if needed
- it picks an image with repeated runs in both `dog->cat` and `dog->dog`
- for `dog->dog`, it can require that the recorded source class matches the classifier top-1, which avoids mixing in older runs with inconsistent source/target bookkeeping
- it can require structured repeated runs (`_rep...`) only, which avoids legacy naming/layout artifacts
- it reports target-monotonicity violations before plotting, so we can detect if the selected run set still contains trajectories inconsistent with the current `skip + require_target_increase` setup
- it now plots all selected trajectories as thin probability curves, with one representative trajectory highlighted for each regime
- when averaging, it carries the final value forward after early stopping instead of dropping finished runs from the mean
- it marks the crossing point of the representative trajectory, so the line plot and the montage refer to the same run
- it now also exports a representative-trajectory montage for the same selected image in both regimes, using the same single-trajectory visual language as the older image-steering cells
- for the selected paper example, figure titles use class names such as `Kerry blue terrier -> cat` and `Kerry blue terrier -> Lakeland terrier`, not raw image IDs or target indices

For the multi-image regime summary cell:
- it can be restricted to the active experiment image list (currently `scripts/dog_image_list_strict_100.txt`)
- it supports an optional full-repeat filter, so final paper statistics can be computed only after the desired repeat count is reached in both regimes

If you want to deepen statistics for one specific figure image without waiting for the full pilot, use:
```bash
bash scripts/slurm/steering/submit_single_image_extension.sh \
  /work/pcsl/Noam/diffusion_datasets/selected_images/ILSVRC2012_val_00000729.JPEG \
  249 \
  248 \
  22 \
  24 \
  20260408
```

This will add:
- `22` more `dog->cat` runs
- `24` more `dog->dog` runs with fixed source `249` and fixed target `248`

For the new per-image averaged-curves notebook cell, rebuilding the summary CSVs is not required, because that cell scans the raw run directories directly.

Important:
- for single-image `dog->dog` extensions, do not rely on auto-detected source class
- fix both source and target explicitly from the validated run family you want to extend
- otherwise, if the classifier top-1 changes or differs from the original run setup, the extra jobs can be wasted
- fixed-target `dog->dog` extension runs now write `auto_target.json`; the analysis parsers also infer source/target metadata for older fixed-target runs that predate that bookkeeping fix
- for the selected terrier paper example (`ILSVRC2012_val_00038116`, source `183` Kerry blue terrier, target `189` Lakeland terrier), the raw dog->dog result tree was verified to contain `36` strict passing fixed-target runs; if the notebook only shows the original `4`, restart/reload the notebook after pulling the latest parser fixes

If you want the code to choose a clean paper example from the recent pilot and extend it automatically to a target count, use:
```bash
bash scripts/slurm/steering/submit_best_steering_example.sh \
  scripts/dog_image_list_strict_100.txt \
  20 \
  20260408
```

If you want to preview several clean candidate images before choosing one, use:
```bash
python scripts/select_best_steering_example.py \
  --active-image-list /work/pcsl/Noam/sequential_diffusion/metadata/dog_image_list_strict_100.txt \
  --image-summary-csv /work/pcsl/Noam/sequential_diffusion/metadata/dog_image_summary.csv \
  --target-total 20 \
  --require-repeat-index \
  --top-k 10
```

This prints the top clean image/source/target candidates using the actual image paths from the active pilot list, and it now includes:
- classifier top-1 confidence (`top1_prob`)
- top-1 class index
- true class index
- whether top-1 matches the true label

Ranking is now:
1. strongest matched support across `dog->cat` and `dog->dog`
2. total support
3. classifier confidence on the start image

So the shortlist is statistically convenient and classifier-clean, but you should still inspect the candidate images visually before choosing the paper figure.

To debug why a specific single-image extension is or is not being picked up by the notebook, scan the raw run tree directly:
```bash
python scripts/debug_single_image_steering_runs.py \
  --root /work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1_multi \
  --image-name ILSVRC2012_val_00038116 \
  --target-idx 189
```

This prints each run directory as `PASS` or `DROP` with the parsed repeat, source, top-1, and target metadata. The per-image trajectory notebook cell should agree with the `PASS` count, except that it may intentionally cap the plotted curves via `MAX_RUNS_PER_REGIME`.

This will:
- search the recent multi-image pilot only
- require structured repeated runs
- require `dog->dog` runs where source class matches the classifier top-1
- choose the best-supported image/source/target combination
- submit just enough extra `dog->cat` and `dog->dog` runs to reach the requested total

Recommended order on the cluster:
1. rebuild the two steering summary CSVs
2. run `check_steering_pilot_status.py`
3. if the pilot is incomplete, run `resume_multi_image_steering.sh`
4. only if you need a deeper matched figure example, run `submit_best_steering_example.sh`

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

- `scripts/slurm/sequential/run_sequential.slurm`
  - Single sequential run (one image, one noise).
  - `sbatch scripts/slurm/sequential/run_sequential.slurm`

- `scripts/slurm/sequential/run_grid_search.slurm`
  - Array over noise levels for one image.
  - `sbatch scripts/slurm/sequential/run_grid_search.slurm`

- `scripts/slurm/sequential/run_production.slurm`
  - Array over images and noise levels using `scripts/image_list.txt`.
  - `sbatch scripts/slurm/sequential/run_production.slurm`

- `scripts/slurm/baselines/run_single_uturn_baseline.slurm`
  - One-step baseline across images/noise.
  - `sbatch scripts/slurm/baselines/run_single_uturn_baseline.slurm`

- `scripts/slurm/baselines/run_debug_sweep.slurm`
  - Debug sweep with respacing (fast single U-turn).
  - `sbatch scripts/slurm/baselines/run_debug_sweep.slurm`

- `scripts/slurm/sequential/extend_trajectories.slurm`
  - Extend existing trajectories by N U-turns.
  - `sbatch scripts/slurm/sequential/extend_trajectories.slurm`

- `scripts/slurm/steering/run_steering_experiment.slurm`
  - Guided steering (dogs → cats example).
  - `sbatch scripts/slurm/steering/run_steering_experiment.slurm`

- `scripts/slurm/steering/run_steering_meta.slurm`
  - Meta steering (dog↔cat) using `steered_sequential_uturns_meta.py`.
  - `sbatch scripts/slurm/steering/run_steering_meta.slurm`

- `scripts/slurm/steering/run_steering_meta_catprob.slurm`
  - Meta steering using cat probability only (no dog penalty).
  - `sbatch scripts/slurm/steering/run_steering_meta_catprob.slurm`

- `scripts/slurm/steering/run_steering_dog2dog.slurm`
  - Auto-selected dog→dog steering using `steered_sequential_uturns_v4.py`.
  - `sbatch scripts/slurm/steering/run_steering_dog2dog.slurm`

- `scripts/slurm/steering/run_steering_dog2dog_prob.slurm`
  - Dog→dog steering using target probability only.
  - `sbatch scripts/slurm/steering/run_steering_dog2dog_prob.slurm`

- `scripts/slurm/manifold/run_manifold_probe.slurm`
  - Unguided probe with full logits.
  - `sbatch scripts/slurm/manifold/run_manifold_probe.slurm`

- `scripts/slurm/misc/run_experiment.slurm`
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
  - Recent additions (2026-02-25):
    - Transition highlight: marks the first step where target >= orig (dog→dog) or cat >= dog (meta).
    - Montages:
      - `MONTAGE_SHOW_TITLES` toggles montage suptitle.
      - `MONTAGE_SHOW_FRAME_LABELS` keeps per-frame `t=` labels even when titles are off.
    - Probability-mass plots:
      - `PLOT_SHOW_TITLES` toggles plot titles.
      - `HIGHLIGHT_TRANSITION` toggles vertical marker + point at the transition step.
      - Unguided curves now trim to `u_len = min(len(xu), len(unguided_probs_*))` to avoid x/y mismatch errors.
    - Dog→dog guided run auto-select now picks the most recent run by mtime (prefers `steering_data.npz`).
  - Plot styling note (Matplotlib):
    - Setting only `plt.rcParams["font.size"]` is often not enough if titles/ticks/legends have explicit sizes.
    - For consistently larger text in saved PDFs, also set:
      - `axes.titlesize`, `axes.labelsize`
      - `xtick.labelsize`, `ytick.labelsize`
      - `legend.fontsize`

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
