import h5py, numpy as np, pandas as pd

H5 = "data/processed/porto_sequences_win10.h5"
# index paths
TRAIN_IDX = "data/splits/porto_win10/train_idx.npy"
VAL_IDX   = "data/splits/porto_win10/val_idx.npy"

# load indices
train_idx = np.load(TRAIN_IDX)
val_idx   = np.load(VAL_IDX)
with h5py.File(H5, "r") as f:
    U = f["U"][:]
train_u = np.unique(U[train_idx])
val_u   = np.unique(U[val_idx])
cold_u  = np.setdiff1d(val_u, train_u)

print("unique users  :", len(np.unique(U)))
print("warm users    :", len(train_u))
print("cold users    :", len(cold_u))