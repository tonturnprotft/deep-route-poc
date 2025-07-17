#!/usr/bin/env python3
"""
porto_build_sequences.py

Build fixed-length next-step training sequences from the *grid-encoded* Porto
taxi + weather parquet (output of porto_grid_encode.py).

Framing:
    X = [cell(t-W+1) ... cell(t)] + weather(t)  # per-step weather
    y =  cell(t+1)

We produce chunked HDF5 datasets (/X, /y, /U). Cell IDs are the *raw* dense
IDs from porto_grid_encode (row * n_cols + col). We'll remap later to shrink
embedding size if we trim to active cells.

Default WIN=12.
"""

import argparse, gc, pathlib, sys
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import h5py


# ------------------------------------------------------------------ #
# Stream user-contiguous blocks (assumes global sort by user,timestamp)
# ------------------------------------------------------------------ #
def iter_sorted_user_blocks(ds: pads.Dataset, user_col: str, sort_cols: List[str], columns: List[str], chunksize: int) -> Iterator[pd.DataFrame]:
    buffer_df = None

    for batch in ds.to_batches(columns=columns, batch_size=chunksize):
        df = batch.to_pandas()
        df = df.sort_values(sort_cols, kind="mergesort")  # stable
        if buffer_df is None:
            buffer_df = df
        else:
            buffer_df = pd.concat([buffer_df, df], ignore_index=True)

        users = buffer_df[user_col].to_numpy()
        change_idx = np.nonzero(users[1:] != users[:-1])[0]
        if len(change_idx):
            cutoff = change_idx[-1] + 1
            complete = buffer_df.iloc[:cutoff]
            for _, grp in complete.groupby(user_col, sort=False):
                yield grp
            buffer_df = buffer_df.iloc[cutoff:].reset_index(drop=True)

    if buffer_df is not None and not buffer_df.empty:
        for _, grp in buffer_df.groupby(user_col, sort=False):
            yield grp


# ------------------------------------------------------------------ #
def build_user_sequences(df_user: pd.DataFrame, win: int, cell_col: str, wx_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    cells = df_user[cell_col].to_numpy(dtype=np.int64, copy=False)
    wx = df_user[wx_cols].to_numpy(dtype=np.float32, copy=False)
    T = len(df_user)
    wx_dim = wx.shape[1]

    if T <= win:
        return np.empty((0, win + wx_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)

    n_seq = T - win
    X = np.empty((n_seq, win + wx_dim), dtype=np.float32)
    y = np.empty((n_seq,), dtype=np.int64)

    for i in range(n_seq):
        # history cells
        X[i, :win] = cells[i : i + win].astype(np.float32, copy=False)
        # weather at current step (t = i+win-1)
        X[i, win:] = wx[i + win - 1]
        y[i] = cells[i + win]

    return X, y


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Build sliding next-step sequences from Porto taxi grid+wx parquet.")
    ap.add_argument("--in", dest="inp", required=True, help="Input parquet (porto_grid_encode output).")
    ap.add_argument("--out", dest="out", required=True, help="Output HDF5 sequences file.")
    ap.add_argument("--win", type=int, default=12, help="History length.")
    ap.add_argument("--user-col", default="taxi_id")
    ap.add_argument("--cell-col", default="cell_id")
    ap.add_argument("--wx-cols", default="t2m_C,tp_mm", help="Comma-separated weather columns.")
    ap.add_argument("--chunksize", type=int, default=5_000_000, help="Arrow read batch size.")
    args = ap.parse_args()

    wx_cols = [c.strip() for c in args.wx_cols.split(",") if c.strip()]

    ds = pads.dataset(args.inp, format="parquet")
    schema = ds.schema.names
    required = [args.user_col, args.cell_col, "timestamp"] + wx_cols
    miss = [c for c in required if c not in schema]
    if miss:
        sys.exit(f"Missing required columns: {miss}")

    sort_cols = [args.user_col, "timestamp"]
    if "seq" in schema:
        sort_cols.append("seq")

    # Build user map (contiguous ids)
    uniq = []
    for b in ds.to_batches(columns=[args.user_col]):
        uniq.extend(b.column(0).to_pylist())
    uniq = pd.Index(pd.unique(pd.Series(uniq))).sort_values()
    user_map: Dict = {u: i for i, u in enumerate(uniq)}
    print(f"[users] {len(user_map)} unique users.")

    out_path = pathlib.Path(args.out)
    if out_path.exists():
        out_path.unlink()

    wx_dim = len(wx_cols)
    h5 = h5py.File(out_path, "w")
    X_ds = h5.create_dataset("X", shape=(0, args.win + wx_dim), maxshape=(None, args.win + wx_dim), dtype="float32", chunks=(65536, args.win + wx_dim))
    y_ds = h5.create_dataset("y", shape=(0,), maxshape=(None,), dtype="int64", chunks=(65536,))
    U_ds = h5.create_dataset("U", shape=(0,), maxshape=(None,), dtype="int32", chunks=(65536,))

    total = 0
    cols = list(schema)  # load all columns
    for df_user in iter_sorted_user_blocks(ds, args.user_col, sort_cols, cols, args.chunksize):
        df_user = df_user.sort_values(sort_cols, kind="mergesort")
        X_u, y_u = build_user_sequences(df_user, args.win, args.cell_col, wx_cols)
        if not len(X_u):
            continue
        uid = user_map[df_user[args.user_col].iloc[0]]

        n_new = len(X_u)
        X_ds.resize(total + n_new, axis=0)
        y_ds.resize(total + n_new, axis=0)
        U_ds.resize(total + n_new, axis=0)

        X_ds[total : total + n_new] = X_u
        y_ds[total : total + n_new] = y_u
        U_ds[total : total + n_new] = uid

        total += n_new
        if total % 1_000_000 < n_new:
            print(f"  [emit] total sequences={total:,}")

        del X_u, y_u, df_user
        gc.collect()

    # metadata
    h5.attrs["win"] = args.win
    h5.attrs["wx_dim"] = wx_dim
    h5.attrs["n_users"] = len(user_map)
    h5.attrs["note"] = "porto_build_sequences.py PoC (per-step weather; current-step features)"
    gm = h5.create_group("user_map")
    dt = h5py.string_dtype("utf-8", None)
    gm.create_dataset("user_vals", data=np.array(list(user_map.keys()), dtype=object), dtype=dt)
    gm.create_dataset("user_ids", data=np.array(list(user_map.values()), dtype=np.int32))

    h5.close()
    print(f"[done] wrote sequences -> {out_path}  (total={total:,})")


if __name__ == "__main__":
    main()