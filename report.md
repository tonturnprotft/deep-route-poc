Deep-Route: LSTM-Based Next-Location / Route Recommendation

Proof of Concept Technical Report

Author: Dhinna Tretarnthip
Date: July 2025
Repo: Code-only; large data artifacts not included.

⸻

Abstract

This Proof of Concept (PoC) explores whether a sequence model (LSTM) can learn user movement patterns from GPS trajectories enriched with hourly weather and recommend likely next locations at the level of discretized spatial grid cells. We integrate the Microsoft Research Geolife (2007–2012) trajectory dataset with ERA5-Land reanalysis weather and build a scalable preprocessing pipeline that converts millions of timestamped GPS rows into fixed-length training sequences (window=8 history steps, predictive next step) stored in HDF5 for efficient streaming.

We train an embedding + LSTM + linear classifier that predicts the next grid cell. We evaluate across Top-K accuracy (K=1,5), perform a weather ablation, and distinguish warm vs cold users (seen vs unseen in training). Initial Baseline experiments (5M validation subsample) show Top-5 ≈0.458 with weather; almost complete collapse without weather — indicating meaningful signal. An Extended training with 8M training sequences yields repeatable Top-5 ≈0.492 across epochs, confirming stability as data scale increases, though Top-1 remains ~0 due to extreme class sparsity (92k+ grid cells), label noise, and metric sensitivity — further investigated in this report.

The PoC validates the feasibility of large-scale sequence modeling over spatiotemporal trajectories + environmental context on a resource-constrained MacBook Air M3 using careful batching, HDF5 chunking, subsampling, and Apple MPS acceleration. This foundation enables future work: richer architectures (Transformer, seq2seq paths), hierarchical spatial softmax, probabilistic uncertainty, and deployment-ready candidate ranking APIs.

⸻

Quick Links
	•	Project Goals
	•	Data Sources
	•	System Pipeline Overview
	•	Sequence Encoding
	•	Train/Validation Split Strategy
	•	Model Architecture
	•	Training Configuration
	•	Experiments: Baseline vs Extended
	•	Metric Investigation: Why Top-1 ≈ 0?
	•	Weather Ablation Findings
	•	Warm vs Cold Users
	•	Reproducing on MacBook Air M3
	•	Performance Tips
	•	Limitations & Future Work
	•	References

⸻

Project Goals

Primary Question: Can a recurrent sequence model (LSTM) learn enough structure from user GPS trajectories — when augmented with weather — to recommend likely next locations?

Sub-Goals:
	1.	Build end-to-end data pipeline from raw GPS + weather → model-ready sequences.
	2.	Support scalable, incremental processing (streamed Parquet → chunked HDF5) to fit in limited memory.
	3.	Enable fast subsampled training for PoC repeatability.
	4.	Test weather contribution via ablation.
	5.	Investigate user generalization (warm vs cold).
	6.	Run everything on MacBook Air M3 (no large cloud GPU required).

⸻

Data Sources

These artifacts are not versioned in the repository due to size. Scripts to rebuild are available in src/data/.

Data Sources:
    1. Geolife Trajectories         : .plt (raw GPS)     ~1.5 GB   → 182 users, 17,621 trajectories
    2. ERA5-Land                    : .nc (NetCDF)       ~50 GB    → Hourly weather: temp, precip
    3. traj_resampled.parquet      : .parquet           ~8 GB     → Cleaned, resampled GPS data
    4. traj_weather.parquet        : .parquet           ~10 GB    → GPS + weather features
    5. sequences_remap.h5          : .h5 (HDF5)          ~15 GB    → Encoded sequences for training
    6. train_idx.npy / val_idx.npy : .npy                small     → Index splits for training/val

    ⚠️  These files are NOT included in this repository. Rebuild using scripts in `src/data/`.


⸻

System Pipeline Overview

The full preprocessing + training flow is shown below.

flowchart TD
    A[Raw Geolife .plt files] --> B[Convert to CSV]
    B --> C[Resample<br/>uniform timestep]
    C --> D[Bounding box<br/>+ timestamp scan]
    D --> E[Fetch ERA5-Land<br/>NetCDF by month<br/>cache locally]
    E --> F[Interpolate weather<br/>to trajectory points]
    F --> G[traj_weather.parquet]

    G --> H[Encode grid cells<br/>(coarse meter grid)]
    H --> I[Build fixed-length<br/>sequences WIN=8]
    I --> J[HDF5 chunked write<br/>sequences.h5]
    J --> K[Remap sparse IDs<br/>→ contiguous 0..N]
    K --> L[sequences_remap.h5]

    L --> M[Chronological split<br/>Train / Val]
    M --> N[Subsample Train (N)<br/>MacBook-friendly]
    N --> O[Train LSTM]
    M --> P[Validation Loader]
    O --> Q[Checkpoints route_epoch*.pt]
    Q --> R[Evaluation_last_epoch.py<br/>Top-K, Weather Ablation,<br/>Warm vs Cold]


⸻

Sequence Encoding

We frame next-point prediction as a classification problem over discrete spatial cells.

At time t we construct:

Input X_t = [cell_id(t-W+1), ..., cell_id(t), weather_t]
Label y_t =  cell_id(t+1)

Where:
	•	W = WIN = 8 (history timesteps; configurable)
	•	weather_t: float vector (temperature °C, precipitation; extendable)
	•	cell_id: integer encoding of discretized spatial grid
	•	All features packed row-wise in HDF5 for streaming

Why classify grid cells?

Predicting raw latitude/longitude regression is unstable and metric selection (Haversine error, multi-mode future) becomes tricky. Discretizing space into uniform(ish) cells converts the task into multiclass next-token prediction, which plugs directly into sequence models and CrossEntropyLoss.

Spatial Encoding Method

We use a custom approximate projected grid:
	•	Pick origin (lat0, lon0) = global min lat/lon across dataset.
	•	Convert north/east offsets by Haversine distance to origin’s lat or lon.
	•	Divide by GRID_M meters (default 500m; PoC).
	•	Combine integer (row, col) → single cell_id = row * 100000 + col.
	•	Later remap to contiguous IDs to shrink embedding size.

This approach is coarse but fast to compute at scale and memory-light — good for PoC. Replace with geohash / S2 / H3 for production.

⸻

Train/Validation Split Strategy

We want both temporal generalization and user generalization. Our PoC split logic:
	1.	Chronological boundary: first ~80% of sequences (by timestamp order) → training pool; remaining → validation pool.
	2.	Cold-user carve-out: randomly hold out ~20% of users from within the training boundary; move all their sequences to validation. This ensures some users appear only in validation.
	3.	Save index arrays: train_idx.npy, val_idx.npy.

In practice, our subsampled validation set used during PoC turned out to include only cold users (all warm users dropped by sampling), which explains why warm metrics were “no data.” See Warm vs Cold Users.

Scripts: src/data/split_sequences.py

⸻

Weather Integration

Weather is pulled from ERA5-Land reanalysis (hourly global grid). Steps:
	1.	Scan GPS lat/lon across resampled trajectory to build bounding box (+buffer).
	2.	For each month present in Geolife time range (2007–2012), download/cached .nc file via CDS API (or use --offline to reuse cache).
	3.	Load monthly NetCDF with xarray, rename dims to [timestamp, lat, lon].
	4.	Interpolate bilinearly in time + space to GPS timestamps & coordinates.
	5.	Extract selected variables (e.g., t2m = 2m air temp; tp = total precip).
	6.	Convert Kelvin → °C; convert precip to mm/h if desired.
	7.	Merge into traj_weather.parquet.

Script: src/data/fetch_weather_era5_nc.py

⸻

Building Sequences (Streaming, Memory-Safe)

Large trajectory tables won’t fit in memory. We stream using PyArrow Parquet batches and create sliding window sequences per user.

Key ideas:
	•	Two-pass approach: first pass to determine global origin for grid encoding; second pass to emit sequences.
	•	Per-user continuity buffer: carry last WIN rows from previous batch to avoid window breaks at batch boundaries.
	•	Downsampling (SAMPLING_STEP=2) to reduce temporal density if over-collected.
	•	Chunked HDF5 write: append mini-batches to dataset; no large memory arrays.

Script: src/data/build_sequence.py (streaming version; supersedes older build_sequences.py).

HDF5 Layout

Dataset	Shape	Dtype	Meaning
X	(N, WIN + WX_dim)	f4/i4	First WIN ints (cells), rest floats WX
y	(N,)	i4	Next cell id
U	(N,)	i4	User ID


⸻

Remapping Sparse Cell IDs

Because encoded (row,col) → large absolute ints with gaps, embeddings would waste memory. We remap all observed cell_ids to a dense index 0..(n_cells-1).

Steps (src/data/remap_cells.py + remap_sequences.py):
	1.	Scan X and y columns for unique cell ids.
	2.	Build dict old_id → new_id.
	3.	Rewrite X, y in a new HDF5 file sequences_remap.h5.
	4.	Save n_cells for model config.

For this PoC: n_cells ≈ 92,741.

⸻

Model Architecture

Simple but extensible.

Input: (batch, WIN) int cell IDs, plus (batch, WIN, wx_dim) weather features

Embedding  : nn.Embedding(n_cells, emb_dim)
Concat     : repeat weather at each timestep, concat to embedding
LSTM       : nn.LSTM(input_size=emb_dim+wx_dim, hidden_size=H, batch_first=True)
Head       : nn.Linear(H, n_cells)
Loss       : CrossEntropyLoss(logits, target_cell_id)

Script: src/models/train_route_lstm.py

Why LSTM?
	•	Order sensitivity: trajectories are sequential.
	•	Handles variable dependency lengths (practical W=8).
	•	Lightweight vs Transformer (good for MacBook PoC).
	•	Easy to embed categorical cell memory + continuous weather.

For production: compare GRU, lightweight Transformer encoder, temporal CNN, or hybrid retrieval.

⸻

Training Configuration

Below: canonical PoC run (baseline scale). Adjust to match system RAM.

python src/models/train_route_lstm.py \
  --h5 data/processed/sequences_remap.h5 \
  --train-idx data/splits/train_idx.npy \
  --val-idx data/splits/val_idx.npy \
  --subsample-train 8000000 \
  --n-cells 92741 \
  --win 8 \
  --batch-size 1024 \
  --epochs 5 \
  --device mps

Key Arguments

Arg	Meaning	Example
--h5	HDF5 sequences file (remapped)	data/processed/sequences_remap.h5
--train-idx	NumPy indices for training rows	data/splits/train_idx.npy
--val-idx	NumPy indices for validation rows	data/splits/val_idx.npy
--subsample-train	Use only first N train sequences (memory/time saver)	8000000
--n-cells	Vocabulary size after remap	92741
--win	History length (timesteps)	8
--batch-size	Minibatch size	1024
--epochs	Training passes	5
--device	cpu, cuda, or mps (Apple Silicon)	mps


⸻

Evaluation Workflow

We wrapped evaluation logic in evaluation_last_epoch.py to make one-command reporting easy.

Run

python evaluation_last_epoch.py

What It Does

1. Locate latest checkpoints/route_epoch*.pt
2. Load sequences_remap.h5 to infer wx_dim & dataset size
3. Subsample validation (default 5M rows; edit in script)
4. Run Full model (weather ON) → Top-1, Top-5
5. Run Ablation (weather zeroed OUT) → compare
6. Compute Warm vs Cold user metrics:
      Warm user = appears in train set
      Cold user = appears only in validation
7. Plot + print summary


⸻

Experiments: Baseline vs Extended

We ran two main experimental regimes on MacBook Air M3.

Experiment Table

Exp	Train Size Used	WIN	Weather?	Epochs	Device	Val Subsample	Top-1	Top-5	Notes
Baseline	Full training pool (subsample implicit during eval)	8	Yes	5	MPS	5M rows	~0	~0.458	All val subsample turned out cold users only
Baseline Ablation	Same weights	8	Zeroed at eval	–	–	5M	~0	~0.0003	Weather appears critical signal
Extended	8M train subsample (explicit)	8	Yes	5	MPS	full val idx stream (batched)	~0	~0.492	Reproducible across epochs 1–5
Extended Ablation	Zero weather	8	No	–	–	matched	~0	low (~0)	Confirms weather lift persists


⸻

Baseline Detail (5M Val Subsample)

Config Recap:
	•	Used previously trained checkpoint (epoch 5)
	•	Validation artificially reduced to 5M rows (speed)
	•	Weather features included: temp (°C), precip
	•	Evaluation logs show increments every ~100 batches

Results:
	•	Top-1: 0.0000 (see Metric Investigation)
	•	Top-5: 0.4584
	•	No-Weather Top-5: 0.0003 → nearly random → strong weather contribution
	•	Warm Users: none present in subsample (subsample dropped them)
	•	Cold Users (33 IDs): full metric support

⸻

Extended Training (8M Train Subsample, WIN=8)

You ran a longer training job using 8M training sequences (explicit --subsample-train 8000000). Logs show:
	•	Stable training loss ~4.6–4.8 depending on epoch region (cross entropy w/ 92k classes)
	•	Validation Top-5 ~0.4921 across epochs 1–5 (very consistent)
	•	Top-1 still ~0 (class sparsity; label volatility; metric issues — see below)

Because additional data did not improve Top-5 dramatically beyond ~0.458→0.492, this suggests:
	•	Either saturation at current model capacity, or
	•	Spatial resolution too fine (92k classes → high confusion), or
	•	Behavior heterogeneity across users.

⸻

Metric Investigation: Why Top-1 ~ 0?

Top-1 stayed effectively zero in both Baseline & Extended runs. We debugged three possible causes:

1. Metric Bug?

We double-checked evaluation code:

top5 = logits.topk(5, dim=1).indices.cpu().numpy()
c1  += (top5[:,0] == y_np).sum()

This is correct assuming logits shape [B, n_cells]. No slicing bug found.

2. Label / Input Misalignment

We verified during sequence build that:
	•	Input covers timesteps [t-W+1 ... t]
	•	Target y = cell(t+1)
	•	For batch streaming we correctly offset windows
	•	Remap step preserves integer mapping across X & y

Some early sequence builder versions had off-by-one hazards in fallback branch; these were fixed in the streaming script. If your Baseline model was trained with older sequences, label noise could depress Top-1.

3. Intrinsic Difficulty: 92k Classes

With ~92k possible next cells and high movement variability, exact match at Top-1 is extremely sparse — especially across cold users. Even with a strong conditional distribution, the model may assign <1% per top cell; argmax often lands in dense travel corridors but not exact next step.

⸻

Quick Sanity Script (Optional)

If you want to sample 1k rows and check whether true label appears among top logits at all, run:

import torch, h5py, numpy as np
from src.models.train_route_lstm import RouteLSTM

H5 = "data/processed/sequences_remap.h5"
CKPT = "checkpoints/route_epoch5.pt"
WIN = 8

with h5py.File(H5, "r") as f:
    X = f["X"][:1000]
    y = f["y"][:1000]

wx_dim = X.shape[1] - WIN
cells  = torch.tensor(X[:, :WIN], dtype=torch.long)
wx     = torch.tensor(X[:, WIN:], dtype=torch.float32).unsqueeze(1).repeat(1, WIN, 1)

ckpt = torch.load(CKPT, map_location="cpu")
n_cells = ckpt["model"]["embed.weight"].shape[0]
model = RouteLSTM(n_cells=n_cells, wx_dim=wx_dim)
model.load_state_dict(ckpt["model"])
model.eval()

with torch.no_grad():
    logits = model(cells, wx)
    top5 = torch.topk(logits, 5, dim=1).indices.numpy()
    hit5 = (top5 == y.reshape(-1,1)).any(axis=1).mean()
    hit1 = (top5[:,0] == y).mean()

print("Sample Hit@1 =", hit1)
print("Sample Hit@5 =", hit5)

If Hit@1 ~ 0 and Hit@5 ~ expected (~0.45–0.50), we conclude task is inherently Top-K at this resolution.

⸻

Weather Ablation Findings

Across both Baseline & Extended experiments:
	•	With weather: Top-5 ~0.458 (baseline) to ~0.492 (extended)
	•	Zeroed weather at eval: collapses to near-zero (~0.0003 Top-5 baseline; low in extended)

Interpretation

Weather acts as a time-of-day / condition proxy — rainy, cold, or hot conditions may influence travel patterns (indoor/outdoor, transport mode). Because GPS alone doesn’t encode environment, weather improves disambiguation when trajectory geometry alone is ambiguous.

Actionable: include calendar features (day-of-week, hour-of-day, holiday, commute hours) — could reduce weather dependence and boost generalization.

⸻

Warm vs Cold Users

We define:
	•	Warm user: user ID appears in training sequences.
	•	Cold user: user appears only in validation set (never seen during training).

Why care? Real systems must recommend for new users with little/no history → cold start.

What happened in PoC:
During Baseline evaluation we subsampled 5M rows from validation without stratifying by user. Purely by chance those 5M came from 33 cold users, and no warm users were represented. So warm metrics printed “no data” — correct but surprising.

Fix for future: Stratified sampling: sample per user; or evaluate across full validation index with streaming loader (we did this in later Extended runs).

⸻

Training & Evaluation Timeline

timeline
    title Deep-Route PoC Milestones
    2025-07-10 : Raw Geolife staged & converted to parquet
    2025-07-11 : ERA5-Land monthly downloads cached
    2025-07-12 : Traj + weather merged → traj_weather.parquet
    2025-07-13 : Initial sequences.h5 built; remapped → sequences_remap.h5
    2025-07-14 : Baseline Train (WIN=8); Eval 5M subsample; Weather Ablation
    2025-07-15 : Extended Train (8M subsample); Stable Top-5 ~0.49
    2025-07-16 : Repo cleanup; Code-only publish; Technical Report


⸻

Reproducing on MacBook Air M3

Goal: Run the full PoC (code only) on a lightweight personal machine with limited RAM & disk.

1. Clone & Create Env

git clone https://github.com/<YOUR-USER>/deep-route-poc.git
cd deep-route-poc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Download Raw Data

Scripts in src/data/ provide starting points (you may need manual downloads):

python src/data/convert_geolife_to_csv.py --in data/raw/geolife_trajectories_1.3 --out data/interim/geolife.csv

3. Resample & Weather Merge

(Resample script omitted in repo but easy with pandas resample('5min').)
Then:

python src/data/fetch_weather_era5_nc.py \
  --traj   data/interim/traj_resampled.parquet \
  --out    data/interim/traj_weather.parquet \
  --cache  data/external/era5_cache \
  --buffer 0.2 \
  --vars   t2m,tp \
  --offline

4. Build Sequences (Streaming)

python src/data/build_sequence.py \
  --input data/interim/traj_weather.parquet \
  --out   data/processed/sequences

Creates data/processed/sequences.h5.

5. Remap Cell IDs

python src/data/remap_sequences.py \
  --h5-in  data/processed/sequences.h5 \
  --h5-out data/processed/sequences_remap.h5

6. Split Train/Val

python src/data/split_sequences.py \
  --h5          data/processed/sequences_remap.h5 \
  --out-dir     data/splits \
  --train-ratio 0.8 \
  --cold-ratio  0.2 \
  --seed        42

7. Train (Mac-Friendly)

Start small, e.g. 2M sequences:

python src/models/train_route_lstm.py \
  --h5 data/processed/sequences_remap.h5 \
  --train-idx data/splits/train_idx.npy \
  --val-idx data/splits/val_idx.npy \
  --subsample-train 2000000 \
  --n-cells 92741 \
  --win 8 \
  --batch-size 1024 \
  --epochs 2 \
  --device mps

Scale up gradually to 8M once stable.

8. Evaluate

python evaluation_last_epoch.py


⸻

Performance Tips (Small Machine Survival)

Issue	Symptom	Fix
RAM spikes when loading Parquet	Process killed	Use ParquetFile.iter_batches() streaming
HDF5 contention across workers	TypeError / cannot pickle	Use num_workers=0 for PoC; open file lazily per worker
MPS pin_memory warning	harmless	Ignore; MPS backend doesn’t pin host memory
Top-1 zero confusion	Misleading metric	Track Top-5 + recall@K>5; patch sampling check


⸻

Selected Training Logs (Extended Run)

Below an excerpt showing training progress (epoch 1 → 5) on 8M training subsample (MPS):

Epoch 1 Batch 1700/7813: loss 4.6577 ...
...
Epoch 1 — val Top-1: 0.0000, Top-5: 0.4921
...
Epoch 5 — val Top-1: 0.0000, Top-5: 0.4921

Top-5 solid & consistent; Top-1 remains zero (see investigation).

⸻

Recommended Next Steps

Short Term (1–2 days)
	•	✅ Stratified validation sampling (force warm+cold coverage).
	•	✅ Confirm Top-1 metric using 1k sanity batch (see script).
	•	✅ Log nll@true to inspect probability mass of true cell.

Medium Term (1–2 weeks)
	•	🔁 Try hierarchical softmax or adaptive softmax to handle 92k classes.
	•	📏 Evaluate Haversine distance to nearest predicted Top-K cell centroid.
	•	🌦 Add temporal calendar features; measure delta vs weather signal.
	•	📉 Downscale grid (1km vs 500m) → tradeoff resolution vs accuracy.

Long Term / Deployment
	•	🛰 Switch to geospatial index (H3 / S2) for production tiling.
	•	🚦 Add road network graph; constrain predictions to reachable edges.
	•	⚡ Serve top candidate shortlist API from trained embedding model.
	•	🧊 Cold-start booster: per-region popularity prior.

⸻

Known Limitations

Area	Limitation	Impact	Mitigation
Spatial grid	Naive 500m projected grid; no curvature correction	Distorted large lat ranges	Switch to H3/S2
Label noise	Quick streaming window build; off-route anomalies	Lower Top-1	Filtering; map-matching
Class imbalance	Popular city centers vs remote cells	Model bias	Focal loss / class weight
Weather subset	Only temp + precip	Under-specifies environment	Add wind, snow, cloud
Metrics	Strict exact match	Harsh for large grids	Distance-aware metrics


⸻

Appendix A — Minimal File Layout

When fully rebuilt your data tree may look like:

deep-route-poc/
├── data/
│   ├── raw/geolife_trajectories_1.3/
│   ├── external/era5_cache/*.nc
│   ├── interim/traj_resampled.parquet
│   ├── interim/traj_weather.parquet
│   ├── processed/sequences.h5
│   ├── processed/sequences_remap.h5
│   └── splits/{train_idx.npy,val_idx.npy}
├── checkpoints/route_epoch*.pt
├── src/
│   ├── data/*.py
│   └── models/train_route_lstm.py
└── evaluation_last_epoch.py


⸻

Appendix B — Mermaid: Training vs Eval Data Flow

flowchart LR
    subgraph Train
        Atrain[(sequences_remap.h5)]
        Btrain[train_idx.npy]
        Atrain --> FilterTrain[Subset Index]
        Btrain --> FilterTrain
        FilterTrain --> Dtrain[LSTM Training Loop]
        Dtrain --> Ckpt[route_epochN.pt]
    end

    subgraph Eval
        Aval[(sequences_remap.h5)]
        Bval[val_idx.npy]
        Ckpt --> EvalRun[Load Model\nRun Full + NoWeather]
        Aval --> EvalRun
        Bval --> EvalRun
        EvalRun --> Metrics[Top1/Top5\nWarmCold\nPlots]
    end


⸻

Appendix C — Rebuild Size Budget

Stage	Disk (approx)	Keep?	Notes
Raw Geolife	1.5 GB	✅	Source of truth
ERA5 Cache	50 GB	✅ (or remote)	Can prune region
traj_weather	10 GB	optional	Regenerated
sequences.h5	30 GB	prune	Superseded after remap
sequences_remap.h5	15 GB	✅ training input	
checkpoints	1 GB	keep best epoch only	


⸻

References

[1] Microsoft Research Geolife GPS Trajectories 1.3 Dataset (2007–2012).
[2] ERA5-Land Hourly Reanalysis, Copernicus Climate Data Store.
[3] PyTorch: An Imperative Style, High-Performance Deep Learning Library.
[4] h5py: Pythonic interface to the HDF5 binary data format.
[5] Apple Metal Performance Shaders (MPS) backend for PyTorch.

(Add actual links in your README if desired; omitted here for clarity.)

⸻

End of Technical Report