#!/usr/bin/env python3
"""
porto_grid_encode.py

Encode Porto taxi trajectory points (already clipped + weather-merged) into a
fixed-meter 2D grid (default 500 m). Adds integer `row`, `col`, and `cell_id`
columns. Also writes a JSON sidecar describing grid geometry.

Example
-------
python src/data/porto_grid_encode.py \
  --in  data/interim/porto_traj_portobox_30s_wx_clean.parquet \
  --out data/interim/porto_traj_portobox_30s_wx_grid.parquet \
  --lat-min 40.8 --lat-max 41.4 \
  --lon-min -9.1 --lon-max -7.9 \
  --grid-m 500 \
  --chunksize 5_000_000 \
  --clip
"""

import argparse, json, math, pathlib
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow as pa
import pyarrow.parquet as pq


# ------------------------------------------------------------------ #
# Grid math helpers                                                   #
# ------------------------------------------------------------------ #
def compute_grid_meta(lat_min, lat_max, lon_min, lon_max, grid_m):
    lat0 = 0.5 * (lat_min + lat_max)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = math.cos(math.radians(lat0)) * 111_320.0

    north_m_span = (lat_max - lat_min) * m_per_deg_lat
    east_m_span  = (lon_max - lon_min) * m_per_deg_lon

    rows = int(math.ceil(north_m_span / grid_m))
    cols = int(math.ceil(east_m_span  / grid_m))

    return {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat0": lat0,
        "m_per_deg_lat": m_per_deg_lat,
        "m_per_deg_lon": m_per_deg_lon,
        "north_m_span": north_m_span,
        "east_m_span": east_m_span,
        "grid_m": grid_m,
        "rows": rows,
        "cols": cols,
        "cell_id_formula": "row * cols + col",
        "note": "equirectangular approx anchored at lat0; ~1% error over Porto box",
    }


def encode_chunk(df: pd.DataFrame, meta: dict, clip: bool):
    lat = df["lat"].to_numpy(copy=False)
    lon = df["lon"].to_numpy(copy=False)

    lat_min = meta["lat_min"]
    lon_min = meta["lon_min"]
    m_per_deg_lat = meta["m_per_deg_lat"]
    m_per_deg_lon = meta["m_per_deg_lon"]
    grid_m = meta["grid_m"]
    rows = meta["rows"]
    cols = meta["cols"]

    north_m = (lat - lat_min) * m_per_deg_lat
    east_m  = (lon - lon_min) * m_per_deg_lon

    row = np.floor_divide(north_m, grid_m).astype(np.int32)
    col = np.floor_divide(east_m,  grid_m).astype(np.int32)

    if clip:
        mask = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
        if not mask.all():
            df = df.loc[mask].copy()
            row = row[mask]
            col = col[mask]

    cell_id = (row.astype(np.int64) * np.int64(cols)) + col.astype(np.int64)

    df["row"] = row
    df["col"] = col
    df["cell_id"] = cell_id
    return df


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Encode Porto taxi points into fixed-meter grid cells.")
    ap.add_argument("--in", dest="inp", required=True, help="Input parquet (clean porto traj + weather).")
    ap.add_argument("--out", dest="out", required=True, help="Output parquet with row,col,cell_id.")
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--grid-m", type=float, default=500.0)
    ap.add_argument("--chunksize", type=int, default=5_000_000)
    ap.add_argument("--clip", action="store_true", help="Drop rows outside bbox.")
    args = ap.parse_args()

    meta = compute_grid_meta(args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.grid_m)

    ds = pads.dataset(args.inp, format="parquet")
    nrows = ds.count_rows()
    print(f"[info] input rows={nrows:,}")

    cols = ds.schema.names
    out_path = pathlib.Path(args.out)
    if out_path.exists():
        out_path.unlink()
    writer = None

    processed = 0
    for batch in ds.to_batches(columns=cols, batch_size=args.chunksize):
        df = batch.to_pandas()
        df = encode_chunk(df, meta, clip=args.clip)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        processed += len(df)
        print(f"  wrote chunk; cum={processed:,}")

    if writer is not None:
        writer.close()

    meta_path = out_path.with_suffix(out_path.suffix + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {processed:,} rows -> {out_path}")
    print(f"[meta] grid info -> {meta_path}")


if __name__ == "__main__":
    main()