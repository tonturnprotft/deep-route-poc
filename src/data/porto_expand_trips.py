#!/usr/bin/env python3
"""
Expand the Kaggle Porto taxi trip CSV into a point-level Parquet table.

Each trip row with a POLYLINE list of [lon, lat] pairs (15-sec sampling) is
exploded so that every coordinate becomes its own row with a derived timestamp.

We retain: TRIP_ID, TAXI_ID, CALL_TYPE, DAY_TYPE, MISSING_DATA flag (filtered out if True),
and derived per-point timestamp.

Usage:
    python src/data/porto_expand_trips.py \
        --input data/raw/porto/train.csv \
        --out   data/interim/porto_traj.parquet \
        --chunksize 100000 \
        --sample-secs 15 \
        --keep-cols DAY_TYPE,CALL_TYPE

Notes:
- Input CSV must be comma-delimited, quoted.
- Timestamp column in raw data is UNIX epoch seconds (UTC).
- POLYLINE is a JSON-like string (e.g., "[[-8.6,41.1],[-8.61,41.11],...]").
"""

import argparse, os, sys, json, pathlib, math
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "0")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Porto train.csv")
    ap.add_argument("--out",   required=True, help="Output Parquet path (will be created)")
    ap.add_argument("--chunksize", type=int, default=100_000,
                    help="Rows per CSV chunk (trip rows).")
    ap.add_argument("--sample-secs", type=int, default=15,
                    help="Seconds per POLYLINE sample. Native = 15.")
    ap.add_argument("--keep-cols", default="DAY_TYPE,CALL_TYPE",
                    help="Comma list of categorical cols to retain (optional).")
    ap.add_argument("--skip-missing", action="store_true",
                    help="Drop trips where MISSING_DATA == True.")
    return ap.parse_args()


def robust_json_load(poly_str):
    """Parse the POLYLINE column robustly. Returns list of [lon,lat] or []."""
    if not poly_str or poly_str == "[]":
        return []
    try:
        return json.loads(poly_str)
    except Exception:
        # occasionally malformed spacing/quotes; try fallback
        try:
            poly_str = poly_str.strip().replace("(", "[").replace(")", "]")
            return json.loads(poly_str)
        except Exception:
            return []


def explode_chunk(df, keep_cols, sample_secs, skip_missing):
    """
    Expand one DataFrame chunk of trip rows into point-level rows.
    Returns pyarrow.Table (may be empty).
    """
    out_rows = []
    append = out_rows.append

    for row in df.itertuples(index=False):
        # Extract raw columns (match CSV header order)
        # Header: TRIP_ID,CALL_TYPE,ORIGIN_CALL,ORIGIN_STAND,TAXI_ID,TIMESTAMP,DAY_TYPE,MISSING_DATA,POLYLINE
        trip_id       = row.TRIP_ID
        call_type     = getattr(row, "CALL_TYPE", None)
        taxi_id       = getattr(row, "TAXI_ID", None)
        day_type      = getattr(row, "DAY_TYPE", None)
        missing_data  = getattr(row, "MISSING_DATA", False)
        ts_start      = getattr(row, "TIMESTAMP", None)
        poly_str      = row.POLYLINE

        if skip_missing and str(missing_data).lower() == "true":
            continue

        coords = robust_json_load(poly_str)
        n = len(coords)
        if n == 0:
            continue

        # convert start timestamp
        # raw TIMESTAMP is int epoch seconds; parse to pandas Timestamp (UTC)
        try:
            ts0 = pd.to_datetime(int(ts_start), unit="s", utc=True)
        except Exception:
            # fallback: treat as already datetime convertible
            ts0 = pd.to_datetime(ts_start, utc=True, errors="coerce")

        # downsampling: we step through coords in increments if sample_secs > 15
        # factor = sample_secs / 15; we select every factor-th point
        factor = max(1, int(round(sample_secs / 15)))
        for i in range(0, n, factor):
            lon, lat = coords[i]
            t = ts0 + pd.Timedelta(seconds=i * 15)  # actual underlying spacing always 15s
            # We DO NOT shift timestamps when factor>1; we just skip rows
            out = {
                "trip_id": str(trip_id),
                "taxi_id": int(taxi_id) if pd.notna(taxi_id) else -1,
                "seq": i,
                "timestamp_trip_start": ts0,
                "timestamp": t,
                "lat": float(lat),
                "lon": float(lon),
            }
            if "DAY_TYPE" in keep_cols:
                out["day_type"] = day_type
            if "CALL_TYPE" in keep_cols:
                out["call_type"] = call_type
            append(out)

    if not out_rows:
        return None

    out_df = pd.DataFrame(out_rows)
    # enforce dtypes
    out_df["timestamp_trip_start"] = pd.to_datetime(out_df["timestamp_trip_start"], utc=True)
    out_df["timestamp"]            = pd.to_datetime(out_df["timestamp"], utc=True)

    # arrow table
    return pa.Table.from_pandas(out_df, preserve_index=False)


def main():
    args = parse_args()
    keep_cols = [c.strip().upper() for c in args.keep_cols.split(",") if c.strip()]

    in_path  = pathlib.Path(args.input)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # We will write a multi-batch Parquet via PyArrow writer
    writer = None
    total_trip_rows = 0
    total_point_rows = 0
    chunk_id = 0

    # dtype map for faster read; allow objects for safety
    dtype_map = {
        "TRIP_ID": str,
        "CALL_TYPE": str,
        "ORIGIN_CALL": str,
        "ORIGIN_STAND": str,
        "TAXI_ID": "Int64",  # nullable int
        "TIMESTAMP": "Int64",
        "DAY_TYPE": str,
        "MISSING_DATA": str,
        "POLYLINE": str,
    }

    for df in pd.read_csv(
            in_path,
            chunksize=args.chunksize,
            dtype=dtype_map,
            na_filter=False,
            low_memory=False,
            quotechar='"'):
        chunk_id += 1
        trip_rows = len(df)
        total_trip_rows += trip_rows

        tbl = explode_chunk(df, keep_cols, args.sample_secs, args.skip_missing)
        if tbl is None or tbl.num_rows == 0:
            print(f"[chunk {chunk_id}] no usable rows (all empty/missing)", flush=True)
            continue

        total_point_rows += tbl.num_rows

        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema, compression="snappy")
        writer.write_table(tbl)

        print(f"[chunk {chunk_id}] trips={trip_rows:,} → pts={tbl.num_rows:,} "
              f"(cumulative pts={total_point_rows:,})", flush=True)

    if writer is not None:
        writer.close()

    print("-----------------------------------------------------------")
    print(f"✅ Done. Read {total_trip_rows:,} trip rows.")
    print(f"✅ Wrote ~{total_point_rows:,} point rows to {out_path}")
    print("-----------------------------------------------------------")


if __name__ == "__main__":
    main()