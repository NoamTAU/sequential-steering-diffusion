# Sequential Steering Diffusion

This repository extends OpenAI's `guided-diffusion` codebase to study repeated forward-backward diffusion steps ("U-turn chains"), their ergodicity properties, and selection-based steering in image and text domains.

## What This Repo Contains

- generation code for unsteered sequential U-turn trajectories
- classifier-guided steering code for image-space U-turn chains
- evaluation scripts for CLIP/correlation/classifier analyses
- notebooks for figure generation and postprocessing
- a nested paper repo at [`-ICML2026-MC_diffusion`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/-ICML2026-MC_diffusion)

## Current Project Status

- The code repo is on `main` and currently ahead of `origin/main` by 2 local commits.
- The paper repo has been updated from Overleaf and is now aligned with `overleaf/master`.
- The active Overleaf draft is no longer `NeurIPS2026/main_noam.tex`.
- The current live draft source is [`NeurIPS2026/main_tracked_1.tex`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/-ICML2026-MC_diffusion/NeurIPS2026/main_tracked_1.tex).
- The previous NeurIPS draft tree has been archived under [`old_main/`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/-ICML2026-MC_diffusion/old_main).

## Main Code Entry Points

- [`scripts/sequential_uturns.py`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/scripts/sequential_uturns.py)
  Baseline sequential U-turn generation.
- [`scripts/sequential_uturns_revised.py`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/scripts/sequential_uturns_revised.py)
  More robust sequential runner with manual diffusion loop and resume support.
- [`scripts/steered_sequential_uturns_v4.py`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/scripts/steered_sequential_uturns_v4.py)
  Classifier-guided steering for specific classes.
- [`scripts/steered_sequential_uturns_meta.py`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/scripts/steered_sequential_uturns_meta.py)
  Meta-class steering, e.g. dog vs cat.
- [`scripts/probe_manifold.py`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/scripts/probe_manifold.py)
  Unguided probing with full classifier-logit recording.

## Main Analysis Surfaces

- [`notebooks/plot_generation_sequential.ipynb`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/notebooks/plot_generation_sequential.ipynb)
  Main figure-generation and trajectory-visualization notebook.
- [`PROJECT_STATE.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/PROJECT_STATE.md)
  Session memory and current project notes.
- [`EXPERIMENTS_GUIDE.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/EXPERIMENTS_GUIDE.md)
  Concrete commands, cluster assumptions, and Overleaf workflow notes.

## Current Paper Shape

The current Overleaf draft in [`NeurIPS2026/main_tracked_1.tex`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/-ICML2026-MC_diffusion/NeurIPS2026/main_tracked_1.tex) is organized around:

- general formalism for U-turn chains
- RHM theory of ergodicity and relaxation
- language experiments
- image experiments
- selection-based steering

The current title is:

`Sampling and Steering Data with Chains of U-turns in Diffusion Models`

This is still a tracked review draft, not a cleaned submission source:

- inline tracked-change macros are active (`\ins`, `\del`, `\DSTART`, `\ISTART`)
- at least one visible reviewer note remains in the body
- some appendix figure captions are still in tracked replacement form

The main figure set currently referenced by the paper includes:

- `figs/cartoon_mc_sampling.png`
- `figs/cartoon_mc_steering.png`
- `figs/cartoon_mc_percolation.png`
- `figs/rhm_ergodicity_combined.pdf`
- `figs/love_sadness_scores.png`
- `figs/latent_cosine_norm_uturns_ILSVRC2012_val_00000729_noise100.pdf`
- `figs/latent_cosine_norm_uturns_ILSVRC2012_val_00000729_noise200.pdf`
- `figs/guided_vs_unguided_prob_mass.pdf`
- `figs/guided_vs_unguided_dog2dog_prob_mass.pdf`
- `figs/montage_guided_force.pdf`
- `figs/montage_guided_dog2dog.pdf`

## Common Workflows

Code/results workflow:

```bash
python scripts/sequential_uturns_revised.py ...
python scripts/steered_sequential_uturns_v4.py ...
python scripts/probe_manifold.py ...
```

Paper workflow:

```bash
git -C -ICML2026-MC_diffusion pull --rebase overleaf master
git -C -ICML2026-MC_diffusion push overleaf main:master
```

Cluster helper:

```bash
bash sync_repo.sh "commit message"
```

## Important Caveats

- Many scripts still assume PCSL/H100 paths and the `llm_physics` conda environment.
- Several notebooks and Slurm scripts are HPC-specific and will need path edits for local execution.
- The older `guided-diffusion` upstream README is no longer the right operational reference for this repo.
- The paper repo now has stale build artifacts for `main_noam` even though the active source file is `main_tracked_1.tex`; use the `.tex` source tree, not the old PDF artifact names, as the source of truth.
