# Deep-Route PoC (Code Only)

This repository contains *code only* for the Deep-Route Proof of Concept — an
LSTM-based next-location / route recommendation experiment that integrates
Geolife GPS trajectories with ERA5-Land weather data.

**Large data files and model checkpoints are NOT included** in this repository.
See the Data Preparation section below to rebuild them from public sources.

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/models/train_route_lstm.py \
  --h5 data/processed/sequences_remap.h5 \
  --train-idx data/splits/train_idx.npy \
  --val-idx   data/splits/val_idx.npy \
  --subsample-train 8000000 \
  --n-cells 92741 \
  --win 8 \
  --batch-size 1024 \
  --epochs 5 \
  --device mps
python evaluation_last_epoch.py
Repository Layout
	•	src/data/ – scripts to download Geolife, fetch ERA5, build sequences, remap IDs, split train/val.
	•	src/models/ – training script + LSTM model.
	•	evaluation_last_epoch.py – quick validation + weather ablation + warm/cold stats.
	•	notebooks/ – exploratory data & prep notes
