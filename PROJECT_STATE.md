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
  - Score each proposal via `--score_mode`:
    - `target_minus_penalty_orig_logit` (default): `target_logit − penalty * orig_logit`
    - `target_prob`: maximize the target class probability `P(target)` (no explicit penalty term)
  - Accept the best; retry up to `max_retries` to find improvement.
  - Saves images + `steering_data.npz` + `force_stats.npz`.

- `scripts/steered_sequential_uturns_meta.py`  
  “Meta‑class” steering (e.g., **cats vs dogs**): score uses log‑sum‑exp over class groups.
  - Also supports `--score_mode cat_prob` to directly maximize meta-class probability mass `P(cat)` (no explicit dog penalty).

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
Notes:
- Newer steering runs include `score_<mode>` in the run directory name.
- `steering_data.npz` records `score_mode` so downstream plots can distinguish logits- vs probability-based steering.

## HPC / Slurm Assumptions
Slurm scripts in `scripts/` run on an H100 partition and assume a conda env `llm_physics`. They reference paths like `/work/pcsl/Noam/...`. These need updating for local runs.

## Notes / Potential Pitfalls
- Several scripts call `diffusion.p_sample_loop_forw_back`, but that function **is not present** in `guided_diffusion/gaussian_diffusion.py`. The revised U‑turn script avoids this by manually looping.
- Many hardcoded paths point to HPC locations; adapt when running locally.
- `README.md` is still the vanilla guided‑diffusion README (not specific to sequential steering).
- When updating on HPC, `.ipynb` merges can conflict easily. If you don’t intend to keep local notebook edits on the cluster, the simplest resolution is to discard local notebook changes and check out the remote version (see `EXPERIMENTS_GUIDE.md` for the exact commands we use).

## Next Step When Resuming
Decide which experiment we want to run first:
1. **Unsteered U‑turns** (baseline drift + CLIP embeddings), or
2. **Steered U‑turns** (classifier‑guided trajectory), or
3. **Analysis pass** over existing trajectories.

## Recent Updates
- Session update (2026-04-15):
  - Project split:
    - the work is now being separated into two streams:
      1. a theory / ergodicity paper focused on sequence-level relaxation predictions in language and images
      2. a steering paper focused on steering results, which will require additional data collection
  - Current image priority:
    - the active image question is now whether the observed slow decay of high-level ConvNeXt latents persists at higher masking fractions per U-turn
    - operationally, this means sweeping larger `noise_step` values in the unguided sequential-U-turn pipeline and regenerating the latent-survival figure
  - Tooling added for this pivot:
    - added `scripts/slurm/sequential/run_high_noise_latent_sweep.slurm`
    - added `scripts/slurm/sequential/submit_high_noise_latent_sweep.sh`
    - added `scripts/slurm/sequential/run_high_noise_latent_eval.slurm`
    - extended `scripts/evaluate_all_images_all_noises.py` so the latent evaluator can be restricted to a target image list and target noise steps instead of reprocessing the full historical tree
    - appended a new notebook section to `notebooks/plot_generation_sequential.ipynb` that:
      - renders a row of same-observable latent-survival plots across selected noise levels
      - computes an early-layer vs late-layer AUC summary across noise
      - reports the regime gap `high-level AUC - low-level AUC`, which should cross zero if higher-level latents start decorrelating faster than lower-level ones at high noise

- Added a clean plotting notebook: `notebooks/plot_generation_sequential.ipynb`.
  - Generates a horizontal “cartoon” of sequential U-turns.
  - Uses hardcoded SSH paths under `/work/pcsl/Noam/...`.
  - Optional auto‑selection of a “nice” trajectory using a drift/smoothness proxy from saved CLIP patch embeddings.
- Added a discovery cell in `notebooks/plot_generation_sequential.ipynb` to scan available images, noise levels, trajectory counts, and approximate max U‑turns.
- Added susceptibility + C2 analysis section in `notebooks/plot_generation_sequential.ipynb` (single-image and all-images averages, multiple noise levels). Uses CLIP patch embeddings from `trajectory_data.npz`, optional flip-aware matching, and KMeans discretization for C2.
- Added error bars: bootstrap SEM across trajectories for susceptibility (single image), SEM across trajectories for C2, and SEM across images for all-images plots.
- Updated plots to use shaded SEM bands (no error bars), and label noise levels as ρ = t/tmax.
- Single-image plots now group noise levels by max U-turn length (e.g., 100-step vs 200-step) and generate separate figures per group.
- Added speedups for all-images C2 via optional global codebooks and sampling limits.
- Added notebook export helpers and download links in `notebooks/plot_generation_sequential.ipynb`:
  - Copies saved figures to `notebooks/_exports` (or `NOTEBOOK_EXPORT_DIR`).
  - Displays clickable download links in JupyterLab via `FileLink`.
- Added `download_plots.sh` helper to fetch `/work/pcsl/Noam/sequential_diffusion/results/plots` to local machine via rsync/scp.
- Added `sync_repo.sh` to pull (rebase), commit, and push in one command.
- Added guided vs unguided latent comparison section to `notebooks/plot_generation_sequential.ipynb`:
  - Loads steering meta-cat trajectory (`steering_meta_v2`) and unguided sequential trajectory.
  - Computes ConvNeXt layer cosine similarity vs step 0 (guided vs unguided) and plots a grid.
  - Plots dog/cat meta-class probability mass over steps for guided vs unguided.
- Session update (2026-02-25):
  - Steering updates:
    - Added probability-only steering modes (`target_prob` for dog→dog and `cat_prob` for meta-cat) and made the output directory include `score_<mode>`.
    - `steering_data.npz` now records `score_mode` so analyses can distinguish logits vs probability steering.
    - Added Slurm jobs for probability-only steering: `run_steering_dog2dog_prob.slurm`, `run_steering_meta_catprob.slurm`.
  - Plotting notebook updates:
    - Dog→dog guided run auto-select now chooses the most recent run by mtime (prefers `steering_data.npz` timestamp).
    - Added transition highlighting to probability-mass plots and montages (first step where target >= orig).
    - Added toggles to control montage suptitles vs per-frame `t=` labels:
      - `MONTAGE_SHOW_TITLES` (suptitle on/off)
      - `MONTAGE_SHOW_FRAME_LABELS` (per-frame `t=` labels)
      - `PLOT_SHOW_TITLES` and `HIGHLIGHT_TRANSITION` for probability-mass plots
    - Fixed unguided plot length mismatch by trimming to `u_len = min(len(xu), len(unguided_probs_*))`.
  - Ops / status:
    - Latest `origin/main` includes the above notebook fixes; when updating on HPC, prefer `git checkout origin/main -- notebooks/plot_generation_sequential.ipynb` if stash/rebase conflicts arise.

- Session update (2026-03-05):
  - Paper drafting (Overleaf nested repo):
    - Updated `-ICML2026-MC_diffusion/NeurIPS2026/main_noam.tex` Images + Conclusion to match the “define once, then reference” style:
      - Images section now explicitly documents probability-only steering (monotone target-probability acceptance, batching, retries, skip behavior, and respacing) and contrasts dog→cat vs dog→dog with the same steering rule.
      - Removed force-specific language from the Images section (figures remain probability/montage only).
    - Reminder: the local sandbox cannot `git fetch/pull` from Overleaf, so keep syncing via the normal Overleaf workflow on your machine.

- Session update (2026-03-08):
  - Overleaf ops:
    - Overleaf web UI comments are not stored in Git (won’t appear locally after `git pull`).
    - Documented a recovery for a common Overleaf Git error:
      - `cannot lock ref 'refs/remotes/overleaf/master' ... expected ...`
      - Fix: remove stale lock files, force-refresh `refs/remotes/overleaf/master`, then fast-forward.
  - Paper drafting:
    - Cleaned up duplicated Language-observable definitions in `-ICML2026-MC_diffusion/NeurIPS2026/main_noam.tex` to preserve “define once, reference later” coherence.

- Session update (2026-04-07):
  - Overleaf sync:
    - Pulled the nested paper repo successfully from `overleaf/master`.
    - The paper repo is now aligned with `overleaf/master`.
  - Draft layout change:
    - `NeurIPS2026/main_noam.tex` was removed in the pulled Overleaf version.
    - The apparent active source file is now `-ICML2026-MC_diffusion/NeurIPS2026/main_tracked_1.tex`.
    - The earlier NeurIPS draft tree was moved into `-ICML2026-MC_diffusion/old_main/`.
    - `main_noam.pdf` still exists as a build artifact, but it is no longer the source of truth.
  - Current paper shape:
    - title: `Sampling and Steering Data with Chains of U-turns in Diffusion Models`
    - main sections: formalism, RHM theory, language, images, conclusion, then appendices
    - the current draft is heavily tracked/edited inline, with `\ins{}` and `\del{}` macros active
    - there is still at least one visible inline review note in the body (`MW: For all figures, x-y axis must be specified...`) and several tracked-change blocks in appendix captions, so this is a working review draft rather than a clean submission file
  - Current figure set in the draft now includes the new cartoon panels:
    - `figs/cartoon_mc_sampling.png`
    - `figs/cartoon_mc_steering.png`
    - `figs/cartoon_mc_percolation.png`
    - plus updated language/image plots such as `love_sadness_scores.png` and `latent_cosine_norm_uturns_...`

- Session update (2026-04-07, post-meeting):
  - Overall assessment from the meeting:
    - the paper is interesting but currently reads as exploratory and synthetic rather than as a single sharp breakthrough
    - the weak points are steering evidence, especially in images, and the current draft length/clarity
  - Main scientific risks identified:
    - image steering needs statistics, not just illustrative examples
    - text steering must show that the steered paragraphs remain sensible and not overly corrupted
    - the paper must better isolate what is genuinely new relative to prior single-U-turn work
  - Drafting / writing decisions:
    - reduce the paper toward a 9-page main version with a cleaner appendix
    - streamline section 3 and other long passages, but verify any Claude/ChatGPT edits for factual correctness
    - treat the current tracked draft as working material, not near-final copy
  - Language-observable corrections requested:
    - review Claude-introduced text for factual inaccuracies, especially around the semantic vs syntactic framing
    - rephrase claims around token survival probability, word embeddings, and document embeddings so they do not overstate semantic/syntactic distinctions
    - if necessary, remove the semantic vs syntactic labeling where it is not technically defensible
  - Figure changes requested:
    - revert Figure 4A to cumulative, non-incremental tree edit distance
    - update the text to acknowledge that the layer-3 vs layer-4 distinction cannot presently be resolved cleanly
    - improve captions so each panel explicitly says what is shown on each axis and what each subpanel represents
    - likely remove or revise the last panel of Figure 1, which was judged confusing
  - Steering presentation changes requested:
    - include text examples in the main paper, not just the appendix
    - choose moderate steering examples where the text remains coherent and in-distribution
    - replace offensive or inappropriate text examples
    - if time permits, explore text steering setups that better parallel the image case, such as similar vs opposite sentiments
  - Novelty framing requested:
    - make the contribution relative to previous single-U-turn results more precise
    - emphasize the sequence-level generalization and the new observables, especially for text
  - Immediate action priority:
    - get to a working Friday version with sharper novelty framing, shorter prose, fixed Figure 1 / Figure 4 presentation, and better steering evidence

- Session update (2026-04-07, image-section follow-up):
  - Decided to upgrade image steering from single-image evidence to multi-image evaluation.
  - Added safe multi-image Slurm array jobs:
    - `scripts/slurm/steering/run_steering_meta_catprob_multi.slurm`
    - `scripts/slurm/steering/run_steering_dog2dog_prob_multi.slurm`
  - Added `scripts/build_image_list.py` so the cluster workflow can start from the full `ILSVRC2012_val` directory rather than the old curated 50-image subset.
  - Added `scripts/build_dog_image_list.py` to classify the master image list, optionally join against the official ImageNet validation ground-truth labels, and write a dog-only subset plus a CSV summary of top-1 classes.
  - Recommended image-selection policy is now based on true ImageNet dog labels first, with classifier agreement available as an optional stricter filter.
  - Added `scripts/sample_image_list.py` for reproducible subset selection from the strict dog list.
  - Multi-image steering arrays now use deterministic paired seeds across `dog->cat` and `dog->dog` for the same `(image, repeat)` pair.
  - Current recommended image-steering pilot is `100` strict images x `4` trajectories each (`800` jobs total across both regimes).
  - Fixed the multi-image summary parser so `image_name` is read from the actual image directory level in the results tree, not from the root results folder.
  - Added a notebook cell for per-image averaged steering curves, which searches across existing `dog->cat` and `dog->dog` runs, selects a well-supported common image, and plots mean ± std probability trajectories with the averaged crossing marker.
  - Tightened notebook defaults so the image analysis now prefers the recent strict-100 pilot, excludes inconsistent legacy `dog->dog` runs by default, and can later require full repeat counts for final paper statistics.
  - The per-image averaged-curves cell now also checks target monotonicity explicitly and uses carry-forward averaging after early stopping, which is the correct treatment for the `skip` steering runs.
  - Added a focused single-image extension workflow so a chosen paper example can be deepened to higher trajectory counts with a fixed `dog->dog` target, independently of the broader multi-image pilot.
  - Tightened that single-image extension workflow: `dog->dog` extensions now require both fixed source and fixed target, because relying on auto-detected source can silently invalidate the intended comparison.
  - Added `select_best_steering_example.py` and `submit_best_steering_example.sh` so we can automatically choose a clean image/source/target triple from the recent pilot and top it up to a requested number of trajectories for the paper figure.
  - Fixed `select_best_steering_example.py` to use the real image paths from the active pilot list instead of reconstructing paths under `selected_images`, and added `--top-k` preview output so we can inspect several clean candidates before launching a single-image extension.
  - Extended `select_best_steering_example.py` so the preview now joins against `scripts/dog_image_summary.csv`, reports classifier confidence / true-label agreement, and ranks clean candidates by support first and classifier confidence second.
  - Added `scripts/sync_steering_metadata.sh` and switched the notebook / selector defaults to prefer `/work/pcsl/Noam/sequential_diffusion/metadata`, so generated image lists and dog-summary CSVs can live outside the git repo and survive pulls cleanly on the cluster.
  - Changed the per-image paper cell from mean±std trajectory summaries to all-trajectory probability overlays with one representative run highlighted, and coupled that highlighted run to the representative montage so the displayed crossing point and the montage refer to the same trajectory.
  - Changed the multi-image steering summary figure to a compact one-row violin-plot layout over image-level means, with shorter labels and black mean±SEM markers.
  - Tightened paper-facing image figure typography: both the multi-image violin plots and the selected trajectory overlays/montages now use 10pt-style PDF text, compact two-column figure widths, and class-name titles instead of image IDs / raw class indices.
  - Fixed fixed-target dog→dog extension bookkeeping: future runs now write `auto_target.json`, and the summary/selector/notebook parsers infer missing fixed-target metadata from the run path plus `start_image_info.json` so already-finished extension runs are not filtered out.
  - Verified the selected terrier paper example (`ILSVRC2012_val_00038116`, source `183` Kerry blue terrier, dog target `189` Lakeland terrier): the raw dog→dog result tree contains `36` strict passing fixed-target runs. If the notebook shows only the original `4`, the issue is stale notebook/kernel state or stale parser code, not missing run data.
  - Added `scripts/debug_single_image_steering_runs.py` to scan one image under a result root and print which run directories pass/drop the strict parsing rules (`steering_data.npz`, repeat index, source/top-1 match, and requested target).
  - Added `scripts/check_steering_pilot_status.py` so we can verify, from the cluster, whether the strict-100 repeated steering pilot is actually complete in each regime and on the image intersection before trusting the notebook tables.
  - Added `scripts/build_missing_steering_manifest.py` plus manifest-based Slurm wrappers so the strict-100 pilot can be resumed by relaunching only missing `(image, repeat)` tasks instead of resubmitting the whole batch.
  - Fixed the manifest-resume path so these manifests are written to shared storage under `/work/pcsl/Noam/sequential_diffusion/metadata/manifests` rather than `/tmp`, which was causing resumed compute-node jobs to fail with missing-manifest errors.
  - Added `scripts/slurm/steering/submit_multi_image_steering.sh` so the exact Slurm array size is computed from the dog-only image list and a requested repeat count.
  - The multi-image steering workflow is now designed to run repeated steering trajectories per image, not just one run per image.
  - Added start-image auto-classification support to the steering scripts:
    - dog→dog can now auto-detect the original class and then auto-select a different dog target
    - dog→cat meta steering can now require the start image top-1 class to be a dog, otherwise skip
  - Added `scripts/summarize_steering_runs.py` to aggregate saved `steering_data.npz` files into CSV summaries with:
    - crossing status and first crossing step
    - final / max target probability
    - total attempts
    - start-image class metadata
  - This should make it possible to plot image-steering statistics across images instead of relying only on the single husky example.
  - Added a new section to `notebooks/plot_generation_sequential.ipynb`:
    - `## Multi-image Steering Statistics (Paper-Ready)`
    - it checks for a dog-only image list and aggregated run CSVs, rebuilding them if needed
    - it first aggregates within image across repeated steering trajectories
    - it then aggregates across images for the final comparison
    - current metrics plotted:
      - crossing rate per image
      - first crossing step per image
      - maximum target probability per image
      - total proposal attempts per image
