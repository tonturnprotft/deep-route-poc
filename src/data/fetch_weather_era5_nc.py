#!/usr/bin/env python3
"""
โหลด ERA5-Land Hourly (.netcdf) → interpolate ใส่ trajectory
"""


import os, argparse, pathlib, tempfile, shutil, zipfile
import pandas as pd, numpy as np, xarray as xr, cdsapi
import calendar
import gc
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# --- variable alias ↔ canonical CDS names -------------------------------
VAR_ALIASES = {
    "t2m": "2m_temperature",
    "tp":  "total_precipitation",
    # add more shorthand names here ifจำเป็น
}

# ----- util ------------------------------------------------------------
def round_grid(x, step=0.1):
    return np.round(x / step) * step

def fetch_month_nc(y, m, area, vars_, cache: pathlib.Path, offline: bool = False) -> pathlib.Path:
    """ดึงไฟล์ .nc มาเก็บใน cache (ถ้ามีแล้วก็ข้าม)"""
    if offline:
        fn = cache / f"era5_{y}_{m:02d}.nc"
        if fn.exists():
            return fn
        raise FileNotFoundError(
            f"{fn} not found in cache and --offline flag is set"
        )
    cache.mkdir(parents=True, exist_ok=True)
    fn = cache / f"era5_{y}_{m:02d}.nc"
    if fn.exists():
        return fn

    print(f"⬇️  ERA-5 Land {y}-{m:02d} …")
    c = cdsapi.Client(quiet=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": vars_,
            "year": str(y),
            "month": f"{m:02d}",
            # number of days actually present in the month (handles Feb + leap years)
            "day": [f"{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,          # N,W,S,E
            "grid": [0.1, 0.1],
            "format": "netcdf",
        },
        tmp,
    )
    # บางครั้ง CDS ใส่มาใน .zip
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            inner = [n for n in z.namelist() if n.endswith(".nc")][0]
            with z.open(inner) as src, open(fn, "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.remove(tmp)
    else:
        shutil.move(tmp, fn)
    return fn
# -----------------------------------------------------------------------

def main(a):
    traj = pd.read_parquet(a.traj, columns=["timestamp", "lat", "lon"])
    traj["timestamp"] = (
        pd.to_datetime(traj["timestamp"], utc=True)
          .dt.floor("h").dt.tz_localize(None)
    )

    result_path = pathlib.Path(a.out)
    if result_path.exists():
        result_path.unlink()           # start fresh
    writer = None

    # ▸ bounding-box (+buffer)
    north, south = traj.lat.max()+a.buffer, traj.lat.min()-a.buffer
    east,  west  = traj.lon.max()+a.buffer, traj.lon.min()-a.buffer
    area = [north, west, south, east]

    ym_list = sorted({(d.year, d.month) for d in traj.timestamp})

    # --- limit to years actually present in GeoLife (Apr‑2007 … Aug‑2012) unless --months provided ---
    # -- optional manual override from --months ---------------------------
    if a.months:
        ym_list = []
        for token in a.months.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                y_str, m_str = token.split("-")
                ym_list.append((int(y_str), int(m_str)))
            except ValueError:  # wrong token format
                raise SystemExit(
                    f"--months entry '{token}' is not in YYYY-MM format"
                )
        ym_list = sorted(set(ym_list))

    ym_list = [
        (y, m) for (y, m) in ym_list
        if (y > 2006 and (y < 2013 or (y == 2012 and m <= 8)))
    ]

    cache = pathlib.Path(a.cache)
    # convert any shorthand to canonical CDS names
    vars_input = [v.strip() for v in a.vars.split(",")]
    vars_ = [VAR_ALIASES.get(v, v) for v in vars_input]

    # step-by-step merge to keep RAM low
    for y, m in tqdm(ym_list, desc="months"):
        nc = fetch_month_nc(y, m, area, vars_, cache, offline=a.offline)
        ds = xr.open_dataset(nc, engine="netcdf4", chunks={"time": 168})  # week-sized
        # spatial subset using whatever coordinate names are present
        lat_dim = "lat" if "lat" in ds.coords else "latitude"
        lon_dim = "lon" if "lon" in ds.coords else "longitude"
        # latitude in ERA5 is usually descending (north -> south). Adjust slice order accordingly
        lat_vals = ds[lat_dim].values
        lat_desc = lat_vals[0] > lat_vals[-1]
        lat_slice = slice(north, south) if lat_desc else slice(south, north)
        # perform spatial subset
        ds = ds.sel(**{lat_dim: lat_slice, lon_dim: slice(west, east)})
        # if the subset returns an empty dataset (e.g. wrong slice direction), skip this month
        if ds.sizes.get(lat_dim, 0) == 0 or ds.sizes.get(lon_dim, 0) == 0:
            ds.close()
            del ds
            gc.collect()
            continue

        var_rename = {short: canon
              for short, canon in VAR_ALIASES.items()
              if short in ds.data_vars and canon not in ds.data_vars}
        # ----- interpolation -------------------------------------------------
        # สร้างตารางเฉพาะเดือนนั้น
        mask = traj.timestamp.dt.to_period("M") == f"{y}-{m:02d}"
        sub = traj.loc[mask]
        # if there are no trajectory points in this month, skip to next iteration
        if sub.empty:
            ds.close()
            del ds
            gc.collect()
            continue
        # --- harmonise coordinate names ----------------------------------
        rename_map = {}
        for old, new in (
            ("longitude", "lon"),
            ("latitude", "lat"),
            ("time", "timestamp"),        # standard ERA5
            ("valid_time", "timestamp"),  # some CDS builds use this name
        ):
            if (old in ds.coords) or (old in ds.dims):
                rename_map[old] = new
        if var_rename:
            ds = ds.rename(var_rename)
        if rename_map:
            ds = ds.rename(rename_map)
        # choose correct time coordinate (some NetCDF builds may keep it as "time" or "valid_time")
        if "timestamp" in ds.coords:
            time_coord = "timestamp"
        elif "time" in ds.coords:
            time_coord = "time"
        elif "valid_time" in ds.coords:
            time_coord = "valid_time"
        else:
            raise ValueError("No usable time coordinate found in NetCDF file.")
        # ----- point‑wise interpolation without full 3‑D broadcast ----------
        out = (
            ds.interp(
                **{
                    time_coord: ("points", sub.timestamp.values),
                    "lon": ("points", sub.lon.values),
                    "lat": ("points", sub.lat.values),
                },
                method="linear",
            )
            .to_dataframe()
            .reset_index()[["points", time_coord, "lat", "lon", *vars_]]
            .drop(columns=["points"])
            .rename(columns={time_coord: "timestamp"})
        )
        out_path = cache / f"wx_{y}_{m:02d}.parquet"
        out.to_parquet(out_path, index=False)
        ds.close()
        del ds
        gc.collect()

    # ---- second pass: stream month‑wise merge straight to parquet ----
    wx_files = sorted(cache.glob("wx_*.parquet"))
    for wp in tqdm(wx_files, desc="merge"):
        wx_month = pd.read_parquet(wp)

        # Kelvin → C
        if "2m_temperature" in wx_month and "t2m_C" not in wx_month:
            wx_month["t2m_C"] = wx_month["2m_temperature"] - 273.15

        # inner merge keeps order of trajectory rows within that month
        merged = traj.merge(wx_month, on=["timestamp", "lat", "lon"], how="left")

        table = pa.Table.from_pandas(merged, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(result_path, table.schema)
        writer.write_table(table)

        # release RAM
        del wx_month, merged, table
        gc.collect()

    if writer is not None:
        writer.close()
    print(f"✅ saved → {result_path}  (stream‑written)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--traj",  required=True)
    p.add_argument("--out",   required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--buffer", type=float, default=0.1)

    p.add_argument(
        "--offline",
        action="store_true",
        help="Do not call the CDS API – fail if a requested month is missing from --cache",
    )

    p.add_argument(
        "--months",
        default="",
        help="Comma‑separated list of months to fetch in YYYY-MM form "
             "(e.g. 2007-05,2007-07). If provided, overrides the "
             "auto‑detected list from the trajectory.",
    )

    p.add_argument("--vars", default="2m_temperature,total_precipitation",
                   help="comma‑separated ERA5‑Land variable names "
                        "(common shorthands like t2m,tp are accepted)")
    main(p.parse_args())