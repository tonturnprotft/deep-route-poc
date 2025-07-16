#!/usr/bin/env python3
"""
Convert merged trajectory-weather rows into fixed-length sequences
for next-point prediction (streaming, low-memory).

X  = [cell_id(t-7) … cell_id(t),  t2m_C(t), total_precipitation(t)]
y  =  cell_id(t+1)
"""

import argparse, pathlib, math, gc
import pandas as pd, numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view
import h5py
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# parameters
GRID_M       = 500         # coarser grid to reduce unique cells
WIN          = 8           # history length
FEAT_WX      = ["t2m_C", "total_precipitation"]   # include temperature and precipitation
BATCH_SIZE   = 10_000      # smaller batches to lower memory footprint
SAMPLING_STEP = 2          # downsample every 2nd point

# --- utility functions ------------------------------------------------
def haversine(lat, lon, lat0, lon0):
    R = 6_371_000
    phi1, phi2 = np.radians(lat0), np.radians(lat)
    dphi  = np.radians(lat - lat0)
    dlamb = np.radians(lon - lon0)
    a = (np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlamb/2)**2)
    return 2*R*np.arcsin(np.sqrt(a))


def encode_cell(lat, lon, lat0, lon0, grid=GRID_M):
    """Approximate E/N offset from an origin and quantise to grid."""
    north_m  = haversine(lat, lon0,  lat0, lon0)
    east_m   = haversine(lat0, lon,  lat0, lon0)
    north_m *= np.sign(lat - lat0)
    east_m  *= np.sign(lon - lon0)
    row = np.floor_divide(north_m, grid).astype(int)
    col = np.floor_divide(east_m,  grid).astype(int)
    return row * 100_000 + col  # unique integer
# ----------------------------------------------------------------------

def main(args):
    input_path = pathlib.Path(args.input)
    out_path   = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # open Parquet for streaming
    pf = pq.ParquetFile(input_path)
    # compute total rows & batches before streaming
    total_rows    = pf.metadata.num_rows
    total_batches = math.ceil(total_rows / BATCH_SIZE)
    print(f"→ Expect {total_rows:,} rows → {total_batches} batches total")

    # 1) first pass: compute origin (lat0, lon0)
    lat0, lon0 = np.inf, np.inf
    for rb in pf.iter_batches(batch_size=BATCH_SIZE, columns=["lat","lon"]):
        tbl = pa.Table.from_batches([rb])
        dfb = tbl.to_pandas()
        lat0 = min(lat0, dfb.lat.min())
        lon0 = min(lon0, dfb.lon.min())
    print(f"Origin set to lat0={lat0:.6f}, lon0={lon0:.6f}")

    # buffers for cross-batch continuity
    prev_buf = {}  # user_id -> DataFrame tail of last WIN rows

    # prepare HDF5 output for chunked write
    h5_path = out_path.with_suffix(".h5")
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    f_h5 = h5py.File(h5_path, "w")

    # explicitly set number of features = window length + label + weather dims
    num_features = WIN + 1 + len(FEAT_WX)

    dX = f_h5.create_dataset(
        "X",
        shape=(0, num_features),
        maxshape=(None, num_features),
        dtype="f4",
        chunks=True,
    )
    dy = f_h5.create_dataset(
        "y",
        shape=(0,),
        maxshape=(None,),
        dtype="i4",
        chunks=True,
    )
    dU = f_h5.create_dataset(
        "U",
        shape=(0,),
        maxshape=(None,),
        dtype="i4",
        chunks=True,
    )
    offset = 0
    cols_needed = ["user_id", "timestamp", "lat", "lon"] + FEAT_WX

    # 2) second pass: streaming sequence build
    for i, rb in enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=cols_needed), start=1):
        print(f"[batch {i}] rows={rb.num_rows}", flush=True)
        tbl = pa.Table.from_batches([rb])
        df = tbl.to_pandas()
        # ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # fallback if user_id missing
        if "user_id" not in df.columns:
            df["user_id"] = 0

        # process each user
        for uid, group in df.groupby("user_id", sort=False):
            g = group.copy()
            # downsample trajectory points to reduce sequence length
            g = g.iloc[::SAMPLING_STEP].reset_index(drop=True)
            # prepend previous buffer if exists
            if uid in prev_buf:
                g = pd.concat([prev_buf[uid], g], ignore_index=True)

            # sort and reset
            g.sort_values(["timestamp"], inplace=True)
            g.reset_index(drop=True, inplace=True)

            # encode cells & extract weather
            cells = encode_cell(g.lat.values, g.lon.values, lat0, lon0)
            wx    = g[FEAT_WX].astype(np.float32).values

            # generate sequences and write chunk to HDF5
            if len(cells) > WIN + 1:
                sw = sliding_window_view(cells, window_shape=WIN+1)[:-1]
                labels = cells[WIN+1:]
                wx_feat = wx[WIN:-1]
                X_batch = np.hstack([sw.astype(np.int32), wx_feat.astype(np.float32)])
                y_batch = labels.astype(np.int32)
            else:
                # fallback per-row sequence
                XB, yB = [], []
                for j in range(WIN, len(g) - 1):
                    x_cells = encode_cell(g.lat.values[j-WIN:j+1], g.lon.values[j-WIN:j+1], lat0, lon0)
                    x_wx    = wx[j]
                    XB.append(np.concatenate([x_cells, x_wx]))
                    yB.append(cells[j+1])
                X_batch = np.array(XB, dtype=np.int32)
                y_batch = np.array(yB, dtype=np.int32)

            # append batch to HDF5
            n = X_batch.shape[0]
            if n > 0:
                dX.resize(offset + n, axis=0)
                dX[offset:offset+n] = X_batch
                dy.resize(offset + n, axis=0)
                dy[offset:offset+n] = y_batch
                # record user_id for each sequence
                uid_array = np.full(n, uid, dtype=np.int32)
                dU.resize(offset + n, axis=0)
                dU[offset:offset+n] = uid_array
                offset += n

            # update buffer tail
            prev_buf[uid] = g.iloc[-WIN:].copy()

        # free memory
        gc.collect()

    # close HDF5 and report
    f_h5.close()
    print(f"✅ wrote HDF5 → {h5_path}  total sequences={offset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="data/interim/traj_weather.parquet")
    parser.add_argument("--out",   default="data/processed/sequences.npz")
    args = parser.parse_args()
    main(args)