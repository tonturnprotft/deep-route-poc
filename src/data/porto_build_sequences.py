#!/usr/bin/env python3
"""
porto_build_sequences.py

Build fixed-length next-step training sequences from the *grid-encoded* Porto
taxi + weather parquet (output of porto_grid_encode.py).

Framing:
    X    = [cell(t‑W+1) ... cell(t)]                    # window of grid cells
    WX   = weather(t)                                   # per‑step weather
    CAT  = categorical context at t (day_type, call_type,…)
    y    = cell(t+1)                                    # next grid cell

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

CAT_MAP_DEFAULT = {"A": 0, "B": 1, "C": 2}


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
def build_user_sequences(
    df_user: pd.DataFrame,
    win: int,
    cell_col: str,
    wx_cols: List[str],
    cat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_cells (int32), WX (float32), CAT (int8), y (int64)."""
    cells = df_user[cell_col].to_numpy(dtype=np.int32, copy=False)
    wx = df_user[wx_cols].to_numpy(dtype=np.float32, copy=False)
    cats = np.vstack(
        [
            df_user[c].map(CAT_MAP_DEFAULT).to_numpy(dtype=np.int8, copy=False)
            for c in cat_cols
        ]
    ).T  # shape (T, cat_dim)

    T = len(df_user)
    wx_dim = wx.shape[1]
    cat_dim = cats.shape[1]

    if T <= win:
        return (
            np.empty((0, win), dtype=np.int32),
            np.empty((0, wx_dim), dtype=np.float32),
            np.empty((0, cat_dim), dtype=np.int8),
            np.empty((0,), dtype=np.int64),
        )

    n_seq = T - win
    X_cells = np.empty((n_seq, win), dtype=np.int32)
    WX_out = np.empty((n_seq, wx_dim), dtype=np.float32)
    CAT_out = np.empty((n_seq, cat_dim), dtype=np.int8)
    y = np.empty((n_seq,), dtype=np.int64)

    for i in range(n_seq):
        X_cells[i] = cells[i : i + win]
        WX_out[i] = wx[i + win - 1]
        CAT_out[i] = cats[i + win - 1]
        y[i] = cells[i + win]

    return X_cells, WX_out, CAT_out, y


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Build sliding next-step sequences from Porto taxi grid+wx parquet.")
    ap.add_argument("--in", dest="inp", required=True, help="Input parquet (porto_grid_encode output).")
    ap.add_argument("--out", dest="out", required=True, help="Output HDF5 sequences file.")
    ap.add_argument("--win", type=int, default=12, help="History length.")
    ap.add_argument("--user-col", default="taxi_id")
    ap.add_argument("--cell-col", default="cell_id")
    ap.add_argument("--wx-cols", default="t2m_C,tp_mm", help="Comma-separated weather columns.")
    ap.add_argument("--cat-cols", default="day_type,call_type",
                    help="Comma‑separated categorical columns to embed.")
    ap.add_argument("--chunksize", type=int, default=5_000_000, help="Arrow read batch size.")
    args = ap.parse_args()

    wx_cols = [c.strip() for c in args.wx_cols.split(",") if c.strip()]
    cat_cols = [c.strip() for c in args.cat_cols.split(",") if c.strip()]

    ds = pads.dataset(args.inp, format="parquet")
    schema = ds.schema.names
    required = [args.user_col, args.cell_col, "timestamp"] + wx_cols + cat_cols
    miss = [c for c in required if c not in schema]
    if miss:
        sys.exit(f"Missing required columns: {miss}")

    sort_cols = [args.user_col, "timestamp"]
    if "seq" in schema:
        sort_cols.append("seq")

    # Build user map (contiguous ids)
    # Note: X stores ONLY the cell history (win); WX & CAT are separate.
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
    # store ONLY the cell‐history window here; per‑step weather goes to WX_ds
    X_ds = h5.create_dataset(
        "X",
        shape=(0, args.win),
        maxshape=(None, args.win),
        dtype="int32",
        chunks=(65536, args.win),
    )
    y_ds = h5.create_dataset("y", shape=(0,), maxshape=(None,), dtype="int64", chunks=(65536,))
    U_ds = h5.create_dataset("U", shape=(0,), maxshape=(None,), dtype="int32", chunks=(65536,))

    WX_ds = h5.create_dataset(
        "WX",
        shape=(0, len(wx_cols)),
        maxshape=(None, len(wx_cols)),
        dtype="float32",
        chunks=(65536, len(wx_cols)),
    )
    CAT_ds = h5.create_dataset(
        "CAT",
        shape=(0, len(cat_cols)),
        maxshape=(None, len(cat_cols)),
        dtype="int8",
        chunks=(65536, len(cat_cols)),
    )

    total = 0
    cols = list(schema)  # load all columns
    for df_user in iter_sorted_user_blocks(ds, args.user_col, sort_cols, cols, args.chunksize):
        df_user = df_user.sort_values(sort_cols, kind="mergesort")
        X_cells_u, WX_u, CAT_u, y_u = build_user_sequences(
            df_user, args.win, args.cell_col, wx_cols, cat_cols
        )
        if not len(X_cells_u):
            continue
        uid = user_map[df_user[args.user_col].iloc[0]]

        n_new = len(X_cells_u)
        for ds_ in (X_ds, WX_ds, CAT_ds, y_ds, U_ds):
            ds_.resize(total + n_new, axis=0)

        X_ds[total : total + n_new] = X_cells_u
        WX_ds[total : total + n_new] = WX_u
        CAT_ds[total : total + n_new] = CAT_u
        y_ds[total : total + n_new] = y_u
        U_ds[total : total + n_new] = uid

        total += n_new
        if total % 1_000_000 < n_new:
            print(f"  [emit] total sequences={total:,}")

        del X_cells_u, WX_u, CAT_u, y_u, df_user
        gc.collect()

    # metadata
    h5.attrs["win"] = args.win
    h5.attrs["wx_dim"] = wx_dim
    h5.attrs["cat_dim"] = len(cat_cols)
    h5.attrs["n_users"] = len(user_map)
    h5.attrs["note"] = "porto_build_sequences.py PoC (per-step weather; current-step features)"
    gm = h5.create_group("user_map")
    user_vals_arr = np.array(list(user_map.keys()), dtype=np.int64)  # taxi_id values
    gm.create_dataset("user_vals", data=user_vals_arr, dtype="int64")
    gm.create_dataset("user_ids", data=np.array(list(user_map.values()), dtype=np.int32))

    h5.close()
    print(f"[done] wrote sequences -> {out_path}  (total={total:,})")


if __name__ == "__main__":
    main()