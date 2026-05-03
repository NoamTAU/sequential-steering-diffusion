# Session Handoff

Last updated: 2026-05-03.

This file is the first thing to read at the start of the next session for this project.

## Recovery Order

To recover context with minimal loss, read in this order:

1. This file: [`SESSION_HANDOFF.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/SESSION_HANDOFF.md)
2. Current project summary: [`PROJECT_STATE.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/PROJECT_STATE.md)
3. Operational workflow / commands: [`EXPERIMENTS_GUIDE.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/EXPERIMENTS_GUIDE.md)
4. Current git state and recent repo history:
   ```bash
   git status --short --branch
   ```
   ```bash
   git log --oneline -15
   ```
5. Main analysis notebook:
   [`notebooks/plot_generation_sequential.ipynb`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/notebooks/plot_generation_sequential.ipynb)

If the next session is about cluster status rather than notebook interpretation, also rerun the status check commands in the `Data Status Checks` section below before doing anything else.

## Codex / Kuma Protocol

Codex can access Kuma directly from the local machine through the SSH alias
`kuma` after the user has logged in / approved SSH access:

```bash
ssh kuma 'hostname'
```

Use this for cluster status checks, repo updates, `sbatch`, `squeue`, and log
tails. Current shared convention for Kuma projects:

```text
/home/nlevi/Noam/<project>       code checkout and small git-synced outputs
/work/pcsl/Noam/<project>        heavy data, raw outputs, checkpoints, caches, logs
/work/pcsl/Noam/<project>/conda_envs
                                      large CUDA conda envs when home quota is tight
```

For this guided-diffusion project:

```text
code:  /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion
work:  /work/pcsl/Noam/sequential_diffusion
env:   llm_physics
```

The intended workflow for future sessions is direct Kuma operation from Codex:

1. inspect local and cluster git status
2. inspect cluster job/data status directly with `ssh kuma`
3. submit jobs directly with `ssh kuma 'cd ... && sbatch ...'`
4. copy compact outputs directly with `scp` when needed
5. commit/push only after local and cluster state have been reconciled

This avoids the older intermediate loop of: local git -> user pulls on Kuma ->
cluster writes files -> user pushes -> local pulls.

Before making changes on Kuma, inspect:

```bash
ssh kuma 'git -C /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion status --short --branch'
```

If a future Kuma project has no remote, Codex can sync it using a local git
bundle copied with `scp`; prefer normal git remotes when available. Avoid
destructive cluster operations unless the exact target is known and explicitly
approved.

## Current Session Snapshot

Status checked locally and on Kuma on 2026-05-03 13:36 CEST:

- local repo is on `main`, tracking `origin/main`
- latest local commit before this documentation update was
  `fd5bbc0 Document work storage for Kuma environments`
- local working tree had the pre-existing `PROJECT_STATE.md` documentation
  update; this session incorporates it
- Kuma code checkout exists at
  `/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion`
- Kuma checkout was behind local GitHub state and had a dirty
  `notebooks/plot_generation_sequential.ipynb` autosave; do not overwrite that
  notebook without first deciding whether cluster-side notebook edits matter
- current documentation files were copied to the older Kuma checkout for
  readability, so cluster `git status` may also show
  `SESSION_HANDOFF.md`, `PROJECT_STATE.md`, and `EXPERIMENTS_GUIDE.md` as
  local changes/untracked until the checkout is reconciled with GitHub
- source-of-truth documentation commit on GitHub is
  `5fdf350 Document diffusion Kuma startup protocol`
- the dirty cluster notebook was backed up at
  `/work/pcsl/Noam/sequential_diffusion/metadata/notebook_backups/plot_generation_sequential.cluster_dirty_20260503_1336.ipynb`
- cluster data completeness checks were rerun:
  - `noise_step = 100, 200, 400, 600, 800`: generation/evaluation complete
    `100/100`
  - `noise_step = 999`: generation/evaluation complete `20/20`
- no notebook sections were rerun in this documentation session

Recommended first cluster action in the next diffusion session:

```bash
ssh kuma 'git -C /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion status --short --branch'
```

If only `notebooks/plot_generation_sequential.ipynb` is dirty on Kuma and the
next session does not need those cluster-side notebook edits, close the notebook
and restore it before pulling:

```bash
ssh kuma 'cd /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion && git restore notebooks/plot_generation_sequential.ipynb && git pull --rebase'
```

If the notebook edits may matter, inspect or copy the notebook first instead of
restoring it.

If the copied documentation files are the only non-notebook obstacle to a pull,
prefer reading them locally/from GitHub, then reconcile the cluster checkout
directly in that session. Do not delete the notebook autosave unless the backup
above has been checked or the session explicitly decides it is disposable.

## Current Focus

The project has split into two streams:

1. Theory / ergodicity paper
   - current focus
   - question: how latent decorrelation changes with sequential U-turns and noise level
   - central observation: low-level and high-level latent ordering inverts at sufficiently high noise

2. Steering paper
   - deprioritized for now
   - more data will be needed later

Current active work is on sequential U-turn latent analysis, not steering.

## Current Scientific State

Use the sequential dataset as the canonical source for both:

- single-U-turn summaries
- multi-U-turn summaries

Do not mix in the separate historical single-U-turn analysis tree for headline results unless explicitly doing a reconciliation or comparison.

Current conclusions from the sequential dataset:

- Step-1 inversion exists.
- The ordering is:
  - lower noise: high-level latents remain more similar than low-level latents after one U-turn
  - high noise: this ordering flips
- Current sequential-only crossover estimate is around `rho ~ 0.75`.
- The classifier head can bias the highest-layer summary, so results should be shown both:
  - with classifier
  - without classifier

Interpretation of statistical sufficiency:

- The current dataset is sufficient for the main inversion claim.
- More images would help more than more trajectories if tighter crossover localization is needed.

## Canonical Data Paths

### Image list used for the high-noise sequential pilot

```text
/work/pcsl/Noam/sequential_diffusion/metadata/high_noise_image_list.txt
```

### Sequential generation outputs

```text
/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns
```

### Sequential latent analysis outputs

```text
/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/sequential_analysis_results
```

ConvNeXt-specific analyzed files live under:

```text
/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/sequential_analysis_results/convnext_base/<image_name>/noise_<t>/sequential_activations_v2.pk
```

## Current Dataset Coverage

Sequential latent sweep status:

- `noise_step = 100, 200, 400, 600, 800`
  - `20` trajectories per image
- `noise_step = 999`
  - `10` trajectories per image
- `noise_step = 0`
  - analytic only in notebook, not generated on cluster

The `999` point is a valid high-noise extension point but is not trajectory-matched to the `20`-trajectory sweeps.

## Notebook Sections That Matter

In [`notebooks/plot_generation_sequential.ipynb`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/notebooks/plot_generation_sequential.ipynb), the relevant later sections are:

1. `High-Noise Latent Regime Comparison`
2. `Multi-Image Latent Survival And Single-U-Turn Noise Sweep`
3. `Curve Stability Diagnostics`
4. `Classifier Sensitivity And Sequential Relaxation Ordering`

These sections are the current source of truth for the theory-facing figures.

## How Grouped Layer Statistics Are Defined

Layer ordering:

- latent layers are sorted by ConvNeXt depth
- classifier / head is treated as the last layer

Grouped summaries currently use:

- `LOW_LAYER_COUNT = 3`
- `HIGH_LAYER_COUNT = 3`

So:

- `low` = first 3 sorted layers
- `high` = last 3 sorted layers

The notebook now includes variants:

- with classifier
- without classifier

This matters because the classifier layer can distort the top-end ordering.

## Averaging Convention

The intended hierarchy is:

1. average over trajectories first
2. then average across selected layers within each image if making grouped low/high summaries
3. then average across images

This is the correct convention because images, not trajectories, are treated as the ensemble unit.

For relaxation ordering, the notebook now shows both AUC conventions:

1. Integrate image-averaged curves
2. Average per-image AUCs

It also shows a half-life ordering summary.

## Relevant Recent Changes

Important recent commits include:

- `de38ea7` Update sequential latent experiment documentation
- `d299499` Fix mathtext labels in relaxation plot
- `1a7a387` Show both AUC aggregation methods in relaxation plot
- `86ef89d` Add classifier sensitivity and relaxation ordering plots
- `afa2b9f` Add sequential step-one all-layer noise plot
- `a943898` Ignore analytic zero-noise point in crossing estimate
- `e5ebe56` Make latent stability diagnostics use sequential data only
- `d8dc46d` Add latent curve stability diagnostics to notebook
- `8c2a8aa` Make sequential resume target a final U-turn index
- `eda897f` Harden sequential resume after interrupted embedding saves
- `e267fcc` Add zero-noise latent baseline and fix trajectory embedding saves

If behavior seems inconsistent with this handoff, inspect these commits first.

## Data Status Checks

Run these first on the cluster if there is any doubt about current completeness.

For `100, 200, 400, 600, 800`:

```bash
python scripts/check_high_noise_latent_status.py \
  --image-list /work/pcsl/Noam/sequential_diffusion/metadata/high_noise_image_list.txt \
  --results-root /work/pcsl/Noam/sequential_diffusion/results/sequential_uturns \
  --analysis-root /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/sequential_analysis_results \
  --noise-steps 100 200 400 600 800 \
  --expected-trajectories 20 \
  --expected-uturns 100
```

For `999`:

```bash
python scripts/check_high_noise_latent_status.py \
  --image-list /work/pcsl/Noam/sequential_diffusion/metadata/high_noise_image_list.txt \
  --results-root /work/pcsl/Noam/sequential_diffusion/results/sequential_uturns \
  --analysis-root /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/sequential_analysis_results \
  --noise-steps 999 \
  --expected-trajectories 10 \
  --expected-uturns 100
```

## Known Operational Pitfalls

### Notebook pull conflicts on cluster

The notebook often becomes dirty due to Jupyter autosave. If `git pull --rebase` fails because of the notebook:

1. close the notebook tab
2. make sure the tree is clean
3. then pull

If needed:

```bash
git restore --source=HEAD --staged --worktree notebooks/plot_generation_sequential.ipynb
git pull --rebase
```

Do not keep fighting `git stash` if the notebook is the only dirty file.

### `noise = 0`

`noise = 0` is analytic in the notebook:

- step-1 cosine = `1`
- full sequential curve = flat at `1`

No cluster generation is required.

### `noise = 999`

This point exists as a real sequential run and is not synthetic, but it currently uses `10` trajectories rather than `20`.

## What To Do At The Start Of The Next Session

Use this exact boot sequence for a separate diffusion Codex session:

1. Start the new session with:
   ```text
   We are resuming the diffusion project. Read /Users/noamlevi/My Drive/Research/Codex/sequential-steering-diffusion/SESSION_HANDOFF.md first, then PROJECT_STATE.md and EXPERIMENTS_GUIDE.md. Use direct Kuma access via ssh kuma for cluster status and jobs.
   ```
2. Enter the repo and check local state:
   ```bash
   cd /Users/noamlevi/My\ Drive/Research/Codex/sequential-steering-diffusion
   git status --short --branch
   ```
3. Pull latest repo if the tree is clean:
   ```bash
   git pull --rebase
   ```
4. Read this handoff file.
5. Read [`PROJECT_STATE.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/PROJECT_STATE.md).
6. If the session involves commands, reruns, or cluster work, read [`EXPERIMENTS_GUIDE.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/EXPERIMENTS_GUIDE.md).
7. Check Kuma directly:
   ```bash
   ssh kuma 'git -C /home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion status --short --branch'
   ssh kuma 'squeue -u nlevi'
   ```
8. Decide the session mode before touching the notebook:
   - paper / analysis mode: rerun only the relevant late notebook sections, not the whole notebook
   - status mode: rerun the high-noise latent status checks below before interpreting missing data
   - ops mode: start from the commands in `EXPERIMENTS_GUIDE.md`
9. If analysis figures are the goal, prioritize exporting the theory-facing sequential latent plots and keep the headline claim sequential-only.

## Immediate Next Work Options

Most likely next useful tasks:

- turn the established sequential-only inversion result into paper-ready figure exports
- decide whether the main text excludes the classifier head by default and shows inclusion as sensitivity
- write the theory-facing interpretation of the low/high inversion without relying on the classifier head
- only collect more data if tighter crossover localization is needed; more images are more valuable than more trajectories

## What Still Needs Judgment Next Time

These are active open decisions, not resolved facts:

- whether the classifier layer should appear in the main text or only in supplement
- whether to present low/high grouped summaries using:
  - classifier excluded by default
  - classifier included as sensitivity check
- whether more images are needed to tighten the crossover estimate
- how to write the theory-facing interpretation of the inversion and sequential relaxation ordering

## If You Need To Reconstruct Context Fast

If time is limited, the shortest reliable recovery path is:

1. read this file
2. read `PROJECT_STATE.md`
3. run the two status checks
4. inspect the last two notebook sections
5. inspect `git log --oneline -10`

That is enough to recover the current state without rereading the full prior discussion.
