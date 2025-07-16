#!/usr/bin/env python3
"""
src/data/remap_sequences.py

Read your original sequences.h5, build a compact integer mapping
for *all* observed cell IDs (in X *and* in y), and write out
sequences_remap.h5 with X, y and U all remapped.
"""
import h5py
import numpy as np
import tqdm

def main(src_path="data/processed/sequences.h5",
         dst_path="data/processed/sequences_remap.h5",
         win=8):
    # 1) Load all unique IDs from X and y
    with h5py.File(src_path, "r") as f:
        X = f["X"]  # shape (N, win+wx_dim)
        y = f["y"]  # shape (N,)
        N = f["y"].shape[0]
        batch = 50_000
        # collect from y
        ids = set()
        for i in tqdm.trange(0, N, batch, desc="Scanning targets"):
            stop = min(i + batch, N)
            ids.update(y[i:stop].tolist())
        # collect from X[:, :win]
        for i in tqdm.trange(0, N, batch, desc="Scanning window inputs"):
            stop = min(i + batch, N)
            chunk = X[i:stop, :win]
            ids.update(np.unique(chunk).tolist())

    unique = np.fromiter(ids, dtype=np.int64)
    unique.sort()
    remap = {old:i for i,old in enumerate(unique)}
    n_cells = len(unique)
    print(f"Found {n_cells} unique cells → remapping to [0…{n_cells-1}]")

    # 2) Create new HDF5 and copy over datasets, remapped
    with h5py.File(src_path, "r") as src, \
         h5py.File(dst_path, "w") as dst:
        N, total_dim = src["X"].shape
        # same dims for X and y and U
        dst.create_dataset("X", shape=(N, total_dim), dtype=src["X"].dtype,
                           chunks=True)
        dst.create_dataset("y", shape=(N,), dtype=src["y"].dtype,
                           chunks=True)
        if "U" in src:
            dst.create_dataset("U", shape=(N,), dtype=src["U"].dtype,
                               chunks=True)

        # process in chunks to save memory
        batch = 50_000
        for i in tqdm.trange(0, N, batch):
            stop = min(i + batch, N)
            Xb = src["X"][i:stop]
            yb = src["y"][i:stop]
            # remap X window columns
            Xb_cells = np.vectorize(remap.__getitem__)(Xb[:, :win])
            Xb[:, :win] = Xb_cells
            # remap y
            yb = np.vectorize(remap.__getitem__)(yb)
            # write out
            dst["X"][i:stop] = Xb
            dst["y"][i:stop] = yb
            if "U" in src:
                Ub = src["U"][i:stop]
                # if you want remapped user_ids too, do similarly; otherwise copy raw
                dst["U"][i:stop] = Ub

    print("✅ Wrote remapped HDF5 →", dst_path)
    print("Now use `--n-cells", n_cells, "` when running train_route_lstm.py")

if __name__ == "__main__":
    main()