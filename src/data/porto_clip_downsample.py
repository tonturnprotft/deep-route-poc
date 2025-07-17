#!/usr/bin/env python3
"""
Clip Porto taxi trajectory points to a fixed geographic bounding box
and coarsen temporal density (default: ~30s) in a streaming,
memory-conscious way.

Input columns expected (case-insensitive checked, lowercased internally):
    trip_id (str or int)
    taxi_id (str or int)
    timestamp (int unix seconds OR datetime string)
    lat, lon (float)
    day_type (optional)
    call_type (optional)

The script was designed to consume the output of porto_expand_trips.py.

Example:
    python src/data/porto_clip_downsample.py \
      --in  data/interim/porto_traj_20130701_20140630.parquet \
      --out data/interim/porto_traj_portobox_30s.parquet \
      --lat-min 40.8 --lat-max 41.4 \
      --lon-min -9.1 --lon-max -7.9 \
      --step-sec 30

"""

import argparse, pathlib, math, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input", required=True,
                   help="Input Parquet of expanded Porto trajectories.")
    p.add_argument("--out", dest="out", required=True,
                   help="Output Parquet (clipped + downsampled).")
    # bounding box
    p.add_argument("--lat-min", type=float, default=40.8)
    p.add_argument("--lat-max", type=float, default=41.4)
    p.add_argument("--lon-min", type=float, default=-9.1)
    p.add_argument("--lon-max", type=float, default=-7.9)
    # temporal downsample
    p.add_argument("--step-sec", type=int, default=30,
                   help="Target sampling seconds. If input is 15s uniform, we stride by 2.")
    p.add_argument("--robust-time", action="store_true",
                   help="Use timestamp-based resample instead of simple stride; slower but safer.")
    p.add_argument("--chunksize", type=int, default=5_000_000,
                   help="Row batch to read from Parquet at a time.")
    return p.parse_args()


# ------------------------------------------------------------------
def _standardize_cols(df):
    """Lowercase column names and standardize expected set."""
    df.columns = [c.lower() for c in df.columns]
    colmap = {}
    # Accept synonyms
    if "trip_id" not in df.columns:
        for cand in ("tripid", "trip", "id"):
            if cand in df.columns:
                colmap[cand] = "trip_id"
                break
    if "taxi_id" not in df.columns:
        for cand in ("taxiid", "taxi"):
            if cand in df.columns:
                colmap[cand] = "taxi_id"
                break
    if "timestamp" not in df.columns:
        for cand in ("time", "ts"):
            if cand in df.columns:
                colmap[cand] = "timestamp"
                break
    if colmap:
        df = df.rename(columns=colmap)
    return df


def _ensure_datetime(df):
    """
    Ensure `timestamp` column is timezone-aware UTC datetime64[ns].

    Handles three common cases:
        1) Already datetime-like (with or without tz)  -> convert/locate to UTC.
        2) Integer/float epoch seconds                  -> interpret as UNIX seconds UTC.
        3) Object/string                                -> parse with pandas.to_datetime (utc=True).

    Any rows that fail to parse become NaT and are dropped.
    """
    import pandas.api.types as pdt

    if "timestamp" not in df.columns:
        raise KeyError("Expected 'timestamp' column in dataframe.")

    col = df["timestamp"]

    if pdt.is_datetime64_any_dtype(col):
        # If tz-naive, localize; if tz-aware, convert to UTC
        try:
            if getattr(col.dt, "tz", None) is None:
                df["timestamp"] = col.dt.tz_localize("UTC")
            else:
                df["timestamp"] = col.dt.tz_convert("UTC")
        except Exception:
            # Fallback parse if something odd slipped in
            df["timestamp"] = pd.to_datetime(col, utc=True, errors="coerce")
    elif pdt.is_numeric_dtype(col):
        # Treat as UNIX seconds
        df["timestamp"] = pd.to_datetime(col.astype("int64"), unit="s", utc=True, errors="coerce")
    else:
        # Generic parse
        df["timestamp"] = pd.to_datetime(col, utc=True, errors="coerce")

    # Drop rows that failed to parse
    before = len(df)
    df = df[df["timestamp"].notna()].copy()
    if len(df) != before:
        # optional: could log count dropped, but avoid noisy prints in inner loops
        pass

    return df


# ------------------------------------------------------------------
def clip_and_downsample_batch(
    df,
    lat_min, lat_max, lon_min, lon_max,
    step_sec=30,
    robust_time=False,
):
    """
    Clip to bbox (drop rows outside), then downsample per trip.
    Returns new dataframe (possibly empty).
    """
    if df.empty:
        return df

    df = _standardize_cols(df)
    df = _ensure_datetime(df)

    # bbox mask
    m = (
        (df["lat"].between(lat_min, lat_max)) &
        (df["lon"].between(lon_min, lon_max))
    )
    df = df.loc[m].copy()
    if df.empty:
        return df

    # sort within trip (just in case chunk order changed)
    df.sort_values(["trip_id", "timestamp"], inplace=True)

    # fast path stride: assume uniform 15s; take every k'th row
    if not robust_time:
        # compute stride k = ceil(step_sec / median_dt)
        # median inter-point delta per trip could vary; we assume ~15s; fallback k>=1
        # We'll approximate globally using diff on first 10k rows:
        # (but faster: if step_sec >= 15 and < 45 -> k=2; else -> scale)
        k = max(1, int(round(step_sec / 15.0)))
        # groupby trip_id and stride
        df = df.groupby("trip_id", sort=False, group_keys=False).nth(slice(None, None, k))
        df = df.reset_index()  # trip_id becomes column
        return df

    # robust timestamp-based: resample to step-sec grid
    out_frames = []
    freq = f"{step_sec}S"
    for tid, g in df.groupby("trip_id", sort=False):
        g = g.set_index("timestamp").sort_index()
        # mark duplicates (if any)
        g = g[~g.index.duplicated(keep="first")]
        g2 = g.resample(freq).first()  # take first point in interval
        g2["trip_id"] = tid
        out_frames.append(g2.reset_index())
    if out_frames:
        df2 = pd.concat(out_frames, ignore_index=True)
    else:
        df2 = pd.DataFrame(columns=df.columns)
    return df2


# ------------------------------------------------------------------
def main():
    args = _parse_args()
    in_path  = pathlib.Path(args.input)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read metadata to get row group count
    pqf = pq.ParquetFile(in_path)
    num_rows = pqf.metadata.num_rows
    num_row_groups = pqf.metadata.num_row_groups
    print(f"[info] input rows={num_rows:,} row_groups={num_row_groups}")

    writer = None
    total_in = total_kept_bbox = total_kept_resample = 0

    for rg in range(num_row_groups):
        batch_tbl = pqf.read_row_group(rg)
        df = batch_tbl.to_pandas()

        n_in = len(df)
        total_in += n_in

        # clip + downsample
        df_out_bbox = df[
            (df["lat"].between(args.lat_min, args.lat_max)) &
            (df["lon"].between(args.lon_min, args.lon_max))
        ].copy()
        kept_bbox = len(df_out_bbox)
        total_kept_bbox += kept_bbox

        if kept_bbox == 0:
            print(f"[rg {rg+1}/{num_row_groups}] dropped all rows (out of bbox).")
            continue

        df_out_bbox = _standardize_cols(df_out_bbox)
        df_out_bbox = _ensure_datetime(df_out_bbox)
        df_out_bbox.sort_values(["trip_id", "timestamp"], inplace=True)

        # stride downsample
        if args.robust_time:
            df_out = clip_and_downsample_batch(
                df_out_bbox,
                args.lat_min, args.lat_max, args.lon_min, args.lon_max,
                args.step_sec, robust_time=True,
            )
        else:
            k = max(1, int(round(args.step_sec / 15.0)))
            df_out = df_out_bbox.groupby("trip_id", sort=False, group_keys=False).nth(slice(None, None, k))
            df_out = df_out.reset_index()  # restore trip_id

        kept_resample = len(df_out)
        total_kept_resample += kept_resample

        # Write batch
        table_out = pa.Table.from_pandas(df_out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table_out.schema, compression="snappy")
        writer.write_table(table_out)

        print(
            f"[rg {rg+1}/{num_row_groups}] in={n_in:,} "
            f"bbox_kept={kept_bbox:,} "
            f"downsample_kept={kept_resample:,}"
        )

    if writer is not None:
        writer.close()
    else:
        # nothing written
        pq.write_table(pa.Table.from_pandas(pd.DataFrame()), out_path)

    print("-----------------------------------------------------------")
    print(f"[done] input rows         : {total_in:,}")
    print(f"[done] in-bbox rows       : {total_kept_bbox:,} ({100*total_kept_bbox/total_in:.2f}%)")
    print(f"[done] after downsample   : {total_kept_resample:,}")
    if total_kept_bbox:
        print(f"[done] dsample reduction  : {100*(1-total_kept_resample/total_kept_bbox):.2f}%")
    print(f"[done] wrote → {out_path}")
    print("-----------------------------------------------------------")


if __name__ == "__main__":
    main()