#!/usr/bin/env python3
"""
Filter expanded Porto point-level Parquet to the 1-year study window:
2013-07-01 through 2014-06-30 (inclusive of end date).

Also prints spatial bounding box so we can fetch ERA5 weather.
"""

import argparse, pathlib
import pandas as pd
import pyarrow.parquet as pq

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True,
                    help="Input expanded parquet (porto_traj.parquet).")
    ap.add_argument("--out", dest="outp", required=True,
                    help="Output filtered parquet.")
    ap.add_argument("--start", default="2013-07-01", help="Start date (inclusive, UTC).")
    ap.add_argument("--end",   default="2014-07-01", help="End date (exclusive, UTC). Use 2014-07-01 to include 2014-06-30.")
    return ap.parse_args()

def main():
    args = parse_args()
    IN  = pathlib.Path(args.inp)
    OUT = pathlib.Path(args.outp)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"[filter] reading {IN} ...")
    tbl = pq.read_table(IN)
    df = tbl.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    start = pd.Timestamp(args.start, tz="UTC")
    end   = pd.Timestamp(args.end,   tz="UTC")

    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    df2 = df.loc[mask].reset_index(drop=True)

    print(f"[filter] kept {len(df2):,} of {len(df):,} rows ({len(df2)/len(df):.1%}).")
    df2.to_parquet(OUT, index=False)
    print(f"[filter] wrote {OUT}")

    # bounding box
    lat_min, lat_max = df2.lat.min(), df2.lat.max()
    lon_min, lon_max = df2.lon.min(), df2.lon.max()
    print("-----------------------------------------------------------")
    print("[filter] BOUNDING BOX (no buffer):")
    print(f"   lat_min={lat_min:.6f}, lat_max={lat_max:.6f}")
    print(f"   lon_min={lon_min:.6f}, lon_max={lon_max:.6f}")
    print("[filter] TIME RANGE:")
    print(f"   start={df2.timestamp.min()}  end={df2.timestamp.max()}")
    print("-----------------------------------------------------------")

if __name__ == "__main__":
    main()