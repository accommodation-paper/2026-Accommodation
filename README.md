# Experimental Sandbox for the Accommodation Layer

This repository contains the code used to study **observable latent-space reorganisation** with accommodation heads, linear heads, and frozen-backbone variants.

The current codebase is organised around a single training entrypoint and a notebook-based visualization workflow.

## Scope

The repository currently supports six datasets:

- Vision: `MNIST`, `FMNIST`, `CIFAR10`
- Text: `AGNEWS`, `IMDB`, `YELP`

The project includes three experiment families:

- `accommodation`: standard end-to-end training with the accommodation head
- `frozen-backbone`: train a backbone first, freeze it, then train the accommodation head
- `linear`: linear-head baselines

## Repository Layout

The main pieces are:

```text
code/
└── accommodation/
    ├── main.py
    ├── train_accommodation_grid.py
    ├── train_frozen_backbone_grid.py
    ├── train_linear_grid.py
    ├── dataset_visualization_builder.py
    ├── datasets/
    ├── experiments/
    └── model/

notebooks/
└── result_viewer.ipynb
```

What each file does:

- [code/accommodation/main.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/main.py): single entrypoint for running all configured training grids
- [code/accommodation/train_accommodation_grid.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/train_accommodation_grid.py): accommodation grid definition and execution
- [code/accommodation/train_frozen_backbone_grid.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/train_frozen_backbone_grid.py): frozen-backbone grid definition and execution
- [code/accommodation/train_linear_grid.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/train_linear_grid.py): linear baselines
- [code/accommodation/dataset_visualization_builder.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/dataset_visualization_builder.py): cache construction and dataset-level visualization utilities
- [notebooks/result_viewer.ipynb](/Users/josejuan/PycharmProjects/2026-accommodation/notebooks/result_viewer.ipynb): main notebook used to generate and inspect figures

## Requirements

- Python `>= 3.14`
- PyTorch
- torchvision
- matplotlib
- pandas
- scikit-learn

Install with:

```bash
pip install -r requirements.txt
```

Or install the project in editable mode:

```bash
pip install -e .
```

## Data

By default, experiments expect the data directory to be:

```text
data/
```

Vision datasets are downloaded automatically by the PyTorch dataset wrappers.

Text datasets must be placed manually in `data/` with these exact filenames:

- `AGNews_dataset.csv`
- `IMDB_dataset.csv`
- `YELP_dataset.csv`

The prepared datasets can be found on Kaggle:
[accommodation-paper-datasets](https://www.kaggle.com/datasets/accommodationpaper/accommodation-paper-datasets).

Expected columns:

- `AGNews_dataset.csv`: either `text`, or `title` + `description`, and a `label` column
- `IMDB_dataset.csv`: `review`, `sentiment`
- `YELP_dataset.csv`: `text`, `stars`

Notes:

- file names are case-sensitive
- the training grids use relative paths by default, so `data/` and `results/` are expected at repository root

## Training

All training runs from a single file:

- [code/accommodation/main.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/main.py)

That file contains:

- the global toggles deciding which experiment families run
- the dataset lists
- the grid settings
- the rule sets used by the accommodation and frozen-backbone sweeps

### How to run

From the repository root:

```bash
PYTHONPATH=code python -m accommodation.main
```

### What to edit

The intended workflow is to edit [code/accommodation/main.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/main.py) directly.

The main switches are:

- `RUN_ACCOMMODATION`
- `RUN_FROZEN_BACKBONE`
- `RUN_LINEAR`

The main configuration blocks are:

- `ACCOMMODATION_SETTINGS`
- `FROZEN_BACKBONE_SETTINGS`
- `LINEAR_SETTINGS`

### Step-by-step reproduction

If you want to reproduce the full workflow from scratch:

1. Install dependencies.
2. Place the text CSV files in `data/`.
3. Edit [code/accommodation/main.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/main.py) to choose:
   - which experiment families to run
   - which datasets to run
   - whether you want the full grids or a temporary reduced subset
4. Run training:

```bash
PYTHONPATH=code python -m accommodation.main
```

5. If needed, rebuild the visualization cache:

```bash
PYTHONPATH=code python -m accommodation.main_build_visualization_cache
```

6. Open [notebooks/result_viewer.ipynb](/Users/josejuan/PycharmProjects/2026-accommodation/notebooks/result_viewer.ipynb) and generate the figures from the cache.

This is the intended end-to-end path for reproducing both training and paper figures.

### Default experiment coverage

The accommodation grid currently spans:

- baseline runs
- differentiation-only runs
- plasticity-only runs
- joint differentiation + plasticity runs
- potents-per-class sweeps

The frozen-backbone grid currently includes:

- baseline frozen-backbone runs
- `lambda = 0.8`, `gamma = 5` frozen-backbone runs

The linear grid runs one baseline configuration per dataset.

### Temporary reductions for quick runs

The repository is usually configured to expose the full dataset lists and rule grids. If you want a smaller run for debugging or limited hardware, the intended way is to edit [code/accommodation/main.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/main.py) temporarily and reduce:

- `datasets`
- `rules`
- `num-cycles`
- `epochs`

The code is designed so that these changes are easy to make in one place and then restore afterwards.

### Resuming incomplete runs

The grid runners automatically inspect the result folders and only schedule cycles that are missing or invalid. In other words, the main training flow is already resume-aware and does not blindly retrain everything.

This applies to:

- accommodation runs
- frozen-backbone runs
- linear runs

## Results Layout

Results are stored under configuration-specific folders inside `results/`.

Typical accommodation/frozen-backbone layout:

```text
results/
└── PLFalse_GM5_DF0_PPC5/
    ├── accommodation/
    │   └── DATASET/
    │       ├── policy-*.json
    │       └── Likelihood/
    │           └── POS...-NEU...-NEG.../
    │               └── SEED/
    │                   ├── snapshot_01.pt
    │                   ├── ...
    │                   └── snapshot_80.pt
    └── frozen-backbone/
        └── DATASET/
            ├── policy-*.json
            ├── pretrained_backbones/
            └── Likelihood/
```

Typical linear layout:

```text
results/
└── linear/
    └── DATASET/
        └── SEED/
            ├── cycle_0.json
            ├── cycle_0.pt
            ├── cycle_1.json
            ├── cycle_1.pt
            └── ...
```

## Visualization Workflow

The current workflow is notebook-first.

The main notebook is:

- [notebooks/result_viewer.ipynb](/Users/josejuan/PycharmProjects/2026-accommodation/notebooks/result_viewer.ipynb)

This notebook uses:

- cached visualization payloads in `visualization_cache/`
- plotting utilities from [code/accommodation/experiments/helpers/visualization_utils.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/experiments/helpers/visualization_utils.py)

### Visualization cache

The cache is built from serialized results and is meant to preserve everything needed for downstream plots without keeping all original training artifacts around forever.

Helper functions for this live in:

- [code/accommodation/dataset_visualization_builder.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/dataset_visualization_builder.py)

There is also a dedicated cache-building entrypoint:

- [code/accommodation/main_build_visualization_cache.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/main_build_visualization_cache.py)

The cache is used for:

- metric curves
- observational-fidelity analyses
- frozen-backbone plots
- temporal CKA
- differentiation matrices
- potent trajectories
- potent semantics

### When the cache is generated

At the moment:

- `frozen-backbone` builds visualization cache automatically at the end of each dataset
- `accommodation` also builds visualization cache automatically at the end of each dataset
- `linear` does not build visualization cache on its own

If you already have results from an older run, or if you want to rebuild everything cleanly, use:

```bash
PYTHONPATH=code python -m accommodation.main_build_visualization_cache
```

### What the cache is for

The intended lightweight archival format for visualization is:

- `visualization_cache/`

The main reason is disk usage: raw `results/` can become very large, especially for text datasets, while the cache is designed to keep the information needed for figure generation in a much smaller form.

In practice:

- keep `results/` if you still want the option to recompute or inspect training artifacts directly
- keep `visualization_cache/` if your main goal is to regenerate figures
- the notebook primarily expects `visualization_cache/`

### What can be deleted

If disk space becomes a problem, the safest order is:

1. build or refresh `visualization_cache/`
2. verify that the notebook figures still load correctly
3. only then consider pruning heavy raw artifacts in `results/`

If you delete `results/` entirely, you will generally lose:

- original checkpoints
- snapshot-level recovery
- some rich recomputation paths that rely on raw saved models

But you can still generate the cache-based figures as long as `visualization_cache/` has already been built with the needed payloads.

### Reproducing figures

The intended figure workflow is:

1. train models with [code/accommodation/main.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/main.py), or obtain a prepared `results/`
2. ensure `visualization_cache/` exists and is up to date
3. open [notebooks/result_viewer.ipynb](/Users/josejuan/PycharmProjects/2026-accommodation/notebooks/result_viewer.ipynb)
4. run the notebook cells to create the paper figures

The notebook is the canonical place for figure generation. The cache builder prepares the data; the notebook renders the figures.

## Reproducibility Notes

- seeds are handled at the cycle level, starting from the configured `base-seed`
- accommodation and frozen-backbone grids support parallel cycle execution
- text and image datasets have dataset-specific defaults such as embedding dimension where needed
- some analyses depend on cached summaries rather than raw checkpoints, so keeping `visualization_cache/` is the recommended lightweight archival format for plots
- accommodation and frozen-backbone caches are refreshed per dataset after training

## Practical Advice

If you want to work with the code as it is now:

1. Put the text CSV files in `data/`.
2. Edit [code/accommodation/main.py](/Users/josejuan/PycharmProjects/2026-accommodation/code/accommodation/main.py) to choose the experiment families and datasets you want.
3. Run `PYTHONPATH=code python -m accommodation.main`.
4. Build or refresh visualization cache when needed with `PYTHONPATH=code python -m accommodation.main_build_visualization_cache`.
5. Open [notebooks/result_viewer.ipynb](/Users/josejuan/PycharmProjects/2026-accommodation/notebooks/result_viewer.ipynb) to inspect and export figures.
