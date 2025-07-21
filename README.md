<file name=0 path=/Users/tree/Projects/deep-route-poc/README.md># Deep-Route PoC (Code Only)

This repository contains *code only* for the Deep-Route Proof of Concept — an
LSTM-based next-location / route recommendation experiment that integrates
Porto Taxi trajectories with ERA5‑Land weather data.

**Large data files and model checkpoints are NOT included** in this repository.
See the Data Preparation section below to rebuild them from public sources.

---

## Contents

- [Quick Start](#quick-start)
- [Project Architecture](#project-architecture)
- [Data Sources](#data-sources)
- [End-to-End Pipeline](#end-to-end-pipeline)
  - [0. Environment Setup](#0-environment-setup)
  - [1. Filter Porto Raw](#1-filter-porto-raw)
  - [2. Clip/Down‑sample](#2-clipdown-sample)
  - [3. Expand Trips & Grid Encode](#3-expand-trips--grid-encode)
  - [4. Fetch ERA5-Land Weather](#4-fetch-era5-land-weather)
  - [5. Merge Trajectory + Weather](#5-merge-trajectory--weather)
  - [6. Build Sequences (Streaming)](#6-build-sequences-streaming)
  - [7. Split Train / Val](#7-split-train--val)
  - [8. Remap Sparse Cell IDs](#8-remap-sparse-cell-ids)
- [Training the LSTM](#training-the-lstm)
- [Evaluation & Weather Ablation](#evaluation--weather-ablation)
- [Metrics](#metrics)
- [Reproducing the PoC Run](#reproducing-the-poc-run)
- [Understanding the Model](#understanding-the-model)
  - [What the LSTM Learns](#what-the-lstm-learns)
  - [What are Top-1 / Top-5?](#what-are-top1--top5)
  - [Warm vs Cold Users](#warm-vs-cold-users)
- [Performance Tips (Mac / M-Series)](#performance-tips-mac--m-series)
- [Known Limitations](#known-limitations)
- [Next Steps & Extensions](#next-steps--extensions)
- [Repository Layout](#repository-layout)
- [Citations & Acknowledgements](#citations--acknowledgements)
- [Weather Integration Details](#weather-integration-details)
- [Git Workflow](#git-workflow)

---

## Quick Start

```bash
# 1. Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train (example: 8 M subsample, win=8, Apple M‑series)
python src/models/train_route_lstm.py \
  --h5 data/processed/porto_sequences_win10.h5 \
  --train-idx data/splits/porto_win10/train_idx.npy \
  --val-idx   data/splits/porto_win10/val_idx.npy \
  --subsample-train 8000000 \
  --n-cells 27025 \
  --win 8 \
  --batch-size 1024 \
  --epochs 5 \
  --device mps

# 4. Quick sanity‑check evaluation on validation slice
python evaluation_route_lstm.py

# 5. Full benchmark across Combined/Warm/Cold splits + PNG plot
python scripts/eval_and_plot.py \
  --h5 data/processed/porto_sequences_win10.h5 \
  --ckpt checkpoints/route_epoch7.pt \
  --idx-files \
        data/splits/porto_win10/test_idx.npy \
        data/splits/porto_win10/warm_test_idx.npy \
        data/splits/porto_win10/cold_idx.npy \
  --labels Combined Warm Cold \
  --out   eval_route_epoch7.png \
  --device mps
```
---

## Project Architecture

The system builds fixed-length history windows from user GPS traces and predicts the next grid cell. Weather at the current time/location is concatenated to each timestep embedding, letting the network learn how environmental context affects movement.

High-level flow:
```bash
Porto raw GPS  ─┐
                  ├── Resample / clean / grid encode ─┐
ERA5-Land weather ┘                                   ├── Merge
                                                      ├── Build sequences (HDF5)
                                                      ├── Split train / val
                                                      ├── Remap cell IDs
                                                      └── Train LSTM → checkpoints → eval
```
---

## Data Sources

| Dataset | Description | Columns Used | Use in Project | Notes |
|---|---|---|---|---|
| [Porto Taxi Trajectory][porto] | 1‑year GPS traces from ~442 taxis in Porto, Portugal (2013/07 – 2014/07) | `taxi_id`, `timestamp`, `lat`, `lon` | Core sequential movement signal; becomes training history cells | Already regular 15‑second sampling; requires year filter, grid encoding, trip expansion & down‑sampling |
| [ERA5-Land][era5] | Hourly global reanalysis (temperature, precipitation, etc.) | `t2m` (2m air temp), `tp` (total precipitation), others optional | Weather context joined to each GPS row (temporal + bilinear spatial interpolation) | Download large monthly NetCDF; convert Kelvin→°C; precipitation unit scaling required. |

[porto]: https://www.kaggle.com/datasets/andresionek/porto-taxi-trip-dataset "Porto Taxi Trajectory dataset"
[era5]: https://cds.climate.copernicus.eu/#!/search?type=dataset&text=ERA5-Land "ERA5-Land on CDS"
---
## End-to-End Pipeline

```bash
Download Porto raw CSV  ─┐
                   ├─► Resample trajectories (regular timestep)
                   │
                   │     ┌─ Download ERA5-Land monthly NetCDF (bbox from traj + buffer)
                   └────►┴─ Merge weather features to each trajectory row (interp in time & space)
                            │
                            ├─ Encode lat/lon to grid cell IDs
                            ├─ Build fixed-length sliding windows (WIN history)
                            ├─ Store compactly in HDF5:  X[seq], y[next cell], U[user]
                            ├─ Remap sparse cell IDs → contiguous [0..N-1] (reduces embedding size)
                            └─ Split train/val (+ optional cold-user masking)
                                      │
                                      ├─ Train LSTM (cells + repeated weather at each step)
                                      └─ Evaluate Top-1/Top-5; Weather Ablation; Warm vs Cold
```
---
## Training the Model

### The model lives in src/models/train_route_lstm.py.

Architecture\
	•	Embedding: lookup for discrete grid cell IDs.\
	•	LSTM: sequence encoder over history window.\
	•	Weather injection: weather vector (e.g., temp, precip) broadcast to each timestep & concatenated to embedding.\
	•	Linear head: predicts logits over n_cells possible next grid cells.\
	•	Loss: CrossEntropyLoss.
## Key Arguments
| Arg                  | Meaning                                                | Example                             |
|----------------------|--------------------------------------------------------|-------------------------------------|
| `--h5`               | HDF5 sequences file (remapped)                         | `data/processed/sequences_remap.h5` |
| `--train-idx`        | NumPy indices for training rows                        | `data/splits/train_idx.npy`         |
| `--val-idx`          | NumPy indices for validation rows                      | `data/splits/val_idx.npy`           |
| `--subsample-train`  | Use only first N train sequences (memory/time saver)   | `8000000`                           |
| `--n-cells`          | Vocabulary size after remap                            | `27025`                             |
| `--win`              | History length (timesteps)                             | `8`                                 |
| `--batch-size`       | Minibatch size                                          | `1024`                              |
| `--epochs`           | Training passes                                         | `5`                                 |
| `--device`           | `cpu`, `cuda`, or `mps` (Apple Silicon)                | `mps`   
---
## Evaluating (Top-1 / Top-5 / Weather Ablation / Warm-vs-Cold)
Run:
```bash
python evaluation_last_epoch.py
```
What it does:\
	1.Locates latest checkpoints/route_epoch*.pt.\
	2.Loads sequences_remap.h5 to infer weather-dim & dataset size.\
	3.Subsamples validation set (default 5M rows; change in script).\
	4.Runs Full model (weather on) inference → Top-1, Top-5.\
	5.Runs Ablation (weather zeroed out) → compare.\
	6.Computes Warm vs Cold user metrics:\
		•Warm user: appears in train set.\
		•Cold user: appears only in validation (never seen in train).\
	7.Plots & prints summary.
 > If your subsampled validation contains only cold users (happened in the PoC), warm metrics will show “no data.” That’s expected; you can reduce the subsample slice or stratify to ensure coverage.
---
## Rebuild the Data Artifacts
> These steps produce the large files we exclude from git.
### 0. Paths
We assume project root $PROJ and working inside it.
```bash
PROJ=.
DATA=$PROJ/data
mkdir -p $DATA/raw $DATA/external $DATA/interim $DATA/processed $DATA/splits
```
### 1. Filter & Prepare Porto Dataset

```bash
# Filter to the study period (2013‑07 → 2014‑06)
python src/data/porto_filter_year.py \
  --in-csv data/raw/taxi_porto.csv \
  --out    data/interim/porto_1yr.csv

# Optional: clip / down‑sample for rapid experiments
python src/data/porto_clip_downsample.py \
  --in-csv  data/interim/porto_1yr.csv \
  --out     data/interim/porto_clipped.csv \
```

### 2. Expand Trips & Grid‑Encode

```bash
# Convert polyline trips to per‑point rows, then grid‑encode lat/lon
python src/data/porto_expand_trips.py \
  --in-csv  data/interim/porto_clipped.csv \
  --out     data/interim/traj_raw.parquet

python src/data/porto_grid_encode.py \
  --in-parquet  data/interim/traj_raw.parquet \
  --out-parquet data/interim/traj_grid.parquet \
  --cell-size 500
```

### 3. Fetch + Merge ERA5‑Land Weather
First list months you need based on trajectory span:
```bash
MONTHS=$(ls data/external/era5_cache/era5_*.nc \
  | sed -E 's/.*era5_([0-9]{4})_([0-9]{2})\.nc/\1-\2/' \
  | sort -u | tr '\n' ',' | sed 's/,$//')
```
Then run merge:
```bash
python src/data/fetch_weather_era5_nc.py \
  --traj   data/interim/traj_resampled.parquet \
  --out    data/interim/traj_weather.parquet \
  --cache  data/external/era5_cache \
  --buffer 0.2 \
  --vars   t2m,tp \
  --offline \
  --months "$MONTHS"
```
Outputs:\
	•	traj_weather.parquet with t2m_C, total_precipitation, etc.
### 4.Build Fixed-Length Sequences (streaming, low-memory)
```bash
python src/data/build_sequence.py \
  --input data/interim/traj_weather.parquet \
  --out   data/processed/sequences
```
Produces: data/processed/sequences.h5 (auto extension) with:\
	•	X: [cell_id(t-W+1) ... cell_id(t), weather...]\
	•	y: cell_id(t+1)\
	•	U: user_id\
### 5. Remap Sparse Cell IDs → Dense Range
```bash
python src/data/remap_cells.py \
  --h5-in  data/processed/sequences.h5 \
  --h5-out data/processed/sequences_remap.h5
```
Saves mapping & new HDF5 (n_cells printed).
### 6. Split Train / Val (+ Optional Cold Users)
```bash
python src/data/split_sequences.py \
  --h5          data/processed/sequences_remap.h5 \
  --out-dir     data/splits \
  --train-ratio 0.7 \
  --cold-ratio  0.2 \
  --test-ratio  0.1 \
  --seed        42
```
Outputs:\
	•	train_idx.npy\
	•	val_idx.npy

#### Example Split Sizes (PoC run)

| Split file        | Size (MB) | # sequences (approx) |
|-------------------|-----------|----------------------|
| train_idx.npy     | 175       | 22.8 M |
| test_idx.npy      | 90        | 11.6 M |
| cold_idx.npy      | 65        | 7.5 M |
| val_idx.npy       | 50        | 7.064 |
| warm_test_idx.npy | 25        | 4.1 M |
___
## Known Limitations
	•	Achieved Top-1 accuracy of ~65% and Top-5 accuracy of ~95% on hold-out test set.
	•	Skewed validation: If subsample picks mostly unseen users, warm metrics disappear.
	•	Grid encoding coarse: 500m grid; finer grid increases vocab and sparsity.
	•	Weather repeated at all timesteps: Simple broadcast; could use time-diff encoding.
	•	MacBook-class hardware: Training times constrained; many scripts favor streaming + chunked HDF5.
---
## Citation / References
> Moreira‑Matias, L., et al. **“Porto Taxi Trajectory Dataset and Trip Prediction.”** KDD Cup 2015.
---
## Git Workflow

This repository is *code‑only*. Large data artifacts and model checkpoints are ignored via `.gitignore`.  
Typical workflow:

```bash
# Stage updated docs, scripts, and plot
git add README.md report*.md scripts/ src/ eval_route_epoch7.png

# Commit with a clear message
git commit -m "Docs+scripts: full PoC workflow and evaluation plot"

# Push (main or feature branch)
git push origin main
```

## Maintainer

Dhinna Tretarnthip\
Deep-Route PoC — LSTM Route Recommendation + Weather Context\
MacBook Air M3 · Python 3.12 · Apple Metal (MPS)
___
## Changelog
	•	2025-07-14: Initial public code-only release; data & checkpoints excluded.
	•	2025-07-16: README expanded; bilingual data source tables; clarified rebuild steps.
</file>
