#!/usr/bin/env python3
"""
inst.
    python src/data/convert_geolife_to_csv.py \
        --input-root data/raw/geolife_trajectories_1.3/Data \
        --output-root data/processed/geolife_csv \
        --tz Asia/Bangkok
"""
import argparse, pathlib, pandas as pd

def convert_file(src: pathlib.Path, dst: pathlib.Path, tz: str):
    # ข้าม header 6 บรรทัดแรก
    cols = ["lat","lon","zero","alt","date_days","date","time"]
    df = pd.read_csv(src, skiprows=6, header=None, names=cols)
    # รวม date & time + set timezone
    df["timestamp"] = pd.to_datetime(df["date"]+" "+df["time"]
                    ).dt.tz_localize("UTC").dt.tz_convert(tz)
    df = df[["timestamp","lat","lon","alt"]]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)

def main(args):
    in_root = pathlib.Path(args.input_root)
    out_root = pathlib.Path(args.output_root)
    plt_files = list(in_root.rglob("*.plt"))
    for f in plt_files:
        rel = f.relative_to(in_root).with_suffix(".csv")
        convert_file(f, out_root/rel, args.tz)
    print(f"✅ done: {len(plt_files)} trajectories → {out_root}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--tz", default="UTC")
    main(p.parse_args())