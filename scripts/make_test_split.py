# scripts/make_test_split.py
import numpy as np, pathlib, h5py

SPLIT_DIR = pathlib.Path("data/splits/porto_win10")
H5_PATH   = pathlib.Path("data/processed/porto_sequences_win10.h5")
TEST_PCT  = 0.10
SEED      = 42

train = np.load(SPLIT_DIR / "train_idx.npy")
val   = np.load(SPLIT_DIR / "val_idx.npy")

with h5py.File(H5_PATH, "r") as h5:
    total_rows = h5["X"].shape[0]

all_rows = np.arange(total_rows, dtype=np.int32)
mask     = np.isin(all_rows, np.concatenate([train, val]), invert=True)
unused   = all_rows[mask]

rng = np.random.default_rng(SEED)
rng.shuffle(unused)

test = unused[: int(TEST_PCT * len(unused))]
np.save(SPLIT_DIR / "test_idx.npy", test)
print(f"Saved {len(test):,} test rows to {SPLIT_DIR/'test_idx.npy'}")