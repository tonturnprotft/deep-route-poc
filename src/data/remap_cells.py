#!/usr/bin/env python3
"""
Remap cell‐IDs in sequences.h5 → a zero‐based [0..n_cells-1] contiguous space.
"""
import h5py
import numpy as np
import argparse

def remap(h5_in, h5_out, win, chunk_size=5_000_000):
    # First pass: collect all unique cell‐IDs in the first `win` columns
    uniques = set()
    with h5py.File(h5_in, "r") as f:
        X = f["X"]
        N, D = X.shape
        for i in range(0, N, chunk_size):
            block = X[i : min(i+chunk_size, N), :win]
            uniques.update(np.unique(block))
    uniques = np.array(sorted(uniques), dtype=int)
    print(f"Found {len(uniques)} unique old cell‐IDs → will remap 0..{len(uniques)-1}")

    # Prepare search‐sorted lookup
    # new_id = np.searchsorted(uniques, old_id)
    # this works because `uniques` is sorted.

    # Second pass: rewrite into a new file
    with h5py.File(h5_in, "r") as f_in, h5py.File(h5_out, "w") as f_out:
        # copy y and U untouched
        f_in.copy("y", f_out)
        if "U" in f_in:
            f_in.copy("U", f_out)

        # create new X of the same shape
        dX = f_out.create_dataset("X",
                                  shape=f_in["X"].shape,
                                  dtype=f_in["X"].dtype,
                                  compression="gzip",
                                  chunks=True)

        # remap in chunks
        for i in range(0, N, chunk_size):
            block = f_in["X"][i : min(i+chunk_size, N)]
            cells = block[:, :win].astype(int)
            remapped = np.searchsorted(uniques, cells)
            block[:, :win] = remapped
            dX[i : i+block.shape[0]] = block

    print(f"✅ Wrote remapped sequences to {h5_out}")
    print(f"Now use `--n-cells {len(uniques)}` when running train_route_lstm.py")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5-in",   required=True,
                   help="path to original sequences.h5")
    p.add_argument("--h5-out",  required=True,
                   help="path for remapped sequences.h5")
    p.add_argument("--win",     type=int, required=True,
                   help="history window length (must match build step)")
    p.add_argument("--chunk",   type=int, default=5_000_000,
                   help="rows per I/O chunk (tune for your RAM)")
    args = p.parse_args()
    remap(args.h5_in, args.h5_out, args.win, args.chunk)