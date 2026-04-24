# Session Handoff

This file is the first thing to read at the start of the next session for this project.

## Recovery Order

To recover context with minimal loss, read in this order:

1. This file: [`SESSION_HANDOFF.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/SESSION_HANDOFF.md)
2. Current project summary: [`PROJECT_STATE.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/PROJECT_STATE.md)
3. Operational workflow / commands: [`EXPERIMENTS_GUIDE.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/EXPERIMENTS_GUIDE.md)
4. Recent repo history:
   ```bash
   git log --oneline -15
   ```
5. Main analysis notebook:
   [`notebooks/plot_generation_sequential.ipynb`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/notebooks/plot_generation_sequential.ipynb)

If the next session is about cluster status rather than notebook interpretation, also rerun the status check commands in the `Data Status Checks` section below before doing anything else.

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

1. Pull latest repo:
   ```bash
   git pull --rebase
   ```
2. Read this handoff file.
3. Read [`PROJECT_STATE.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/PROJECT_STATE.md).
4. If the session involves commands or reruns, read [`EXPERIMENTS_GUIDE.md`](/Users/noamlevi/My%20Drive/Research/Codex/sequential-steering-diffusion/EXPERIMENTS_GUIDE.md).
5. If the session is analysis-focused, open and rerun only the relevant late notebook sections, not the whole notebook.
6. If the session is status-focused, rerun the status-check commands above before making any assumption.

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
