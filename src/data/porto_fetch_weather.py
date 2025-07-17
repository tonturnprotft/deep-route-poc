#!/usr/bin/env python3
"""
porto_fetch_weather.py

Fetch **ERA5-Land hourly** weather for the Porto taxi region and
merge/interpolate it onto a *downsampled, clipped* Porto trajectory parquet
(e.g., the output of `porto_clip_downsample.py`).

This script is a *safe / memory-light* rework of `fetch_weather_era5_nc.py`
tailored to the Porto PoC:

    • Explicit bounding box required (lat/lon args).
    • Explicit date range required (--start, --end  *end-exclusive*).
    • Monthly ERA5 downloads cached under --cache.
    • Chunked interpolation per month + row-group, so we never hold the full
      40M+ trajectory rows in memory.
    • Optional filtering of short trips before interpolation (saves work).
    • Supports offline re-runs (no network if cache files present).
    • Produces a Parquet containing the original trajectory columns
      plus normalized weather columns (°C, mm/h).

Minimal usage (recommended defaults):

```bash
python src/data/porto_fetch_weather.py \
  --traj data/interim/porto_traj_portobox_30s.parquet \
  --out  data/interim/porto_traj_portobox_30s_wx.parquet \
  --cache data/external/era5_porto_cache \
  --lat-min 40.8 --lat-max 41.4 \
  --lon-min -9.1 --lon-max -7.9 \
  --start 2013-07-01 --end 2014-07-01 \
  --vars t2m,tp \
  --min-points 3 \
  --chunksize 2_000_000 \
  --safe \
  --offline-ok
  --interp linear --fallback-nearest
```

-------------------------------------------------------------------------------
Implementation notes
-------------------------------------------------------------------------------
We *subset* ERA5 at 0.1° resolution over the Porto box; that yields a tiny grid
(~7 x 13 points) so loading each monthly NetCDF fully into memory is cheap. We
then vector-interpolate to the taxi points using `xarray.Dataset.interp` with
`method="linear"` in time + space. Because we process in chunks, interpolation
calls are sized ~few 100k–2M points at a time.

If `--safe` is specified (default True), we further break large chunks into
smaller (e.g., 250k) to prevent huge intermediate arrays on laptops.

Weather units are converted:
    • 2m_temperature  (Kelvin → °C) into new column `t2m_C`
    • total_precipitation (m) aggregated hourly → mm (per hour) column `tp_mm`
      NOTE: ERA5-Land `tp` is *accumulated over the hour*. We treat the hourly
      value as an hourly depth. If you need rate, divide by 1hr; since unit is m,
      mm = m * 1000.

-------------------------------------------------------------------------------
"""

import argparse
import calendar
import gc
import os
import pathlib
import shutil
import tempfile
import zipfile
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import xarray as xr
from functools import partial

# We import cdsapi lazily (inside fetch) so that running in offline mode doesn't
# require the package to be installed; but we still type-check import in try/except.


# ------------------------------------------------------------------------------
# Variable alias ↔ canonical CDS names
# ------------------------------------------------------------------------------
VAR_ALIASES = {
    "t2m": "2m_temperature",
    "tp": "total_precipitation",
    # you can add more short names here if needed
}


def _parse_vars(vs: str) -> List[str]:
    out: List[str] = []
    for token in vs.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(VAR_ALIASES.get(token, token))
    return out


# ------------------------------------------------------------------------------
# ERA5 download per-month (cached)
# ------------------------------------------------------------------------------
def fetch_month_nc(
    year: int,
    month: int,
    *,
    area: Sequence[float],
    vars_: Sequence[str],
    cache: pathlib.Path,
    offline_ok: bool,
) -> pathlib.Path:
    """
    Download a single month of ERA5-Land data to cache (if missing).

    Parameters
    ----------
    year, month : int
        Year/month to download.
    area : (N, W, S, E)
        Geographic bounding box for CDS subset request.
    vars_ : sequence of str
        Canonical ERA5 variable names.
    cache : Path
        Directory for monthly NetCDF cache.
    offline_ok : bool
        If True, *do not* attempt download; require file to exist.

    Returns
    -------
    Path to cached NetCDF file.
    """
    cache.mkdir(parents=True, exist_ok=True)
    fn = cache / f"era5_{year}_{month:02d}.nc"
    if fn.exists():
        return fn
    if offline_ok:
        raise FileNotFoundError(
            f"ERA5 cache miss for {year}-{month:02d} at {fn} and offline_ok=True."
        )

    # Lazy import cdsapi only when needed
    try:
        import cdsapi  # type: ignore
    except ImportError as e:  # pragma: no cover - user environment
        raise SystemExit(
            "cdsapi not installed. Install `pip install cdsapi` or rerun in offline mode."
        ) from e

    print(f"[era5] ↓ downloading ERA5-Land {year}-{month:02d} ...")
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name
    c = cdsapi.Client(quiet=True)
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": list(vars_),
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,  # N, W, S, E
            "grid": [0.1, 0.1],
            "format": "netcdf",
        },
        tmp,
    )
    # Some CDS deliveries wrap the NC file inside a zip
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            inner = [n for n in z.namelist() if n.endswith(".nc")][0]
            with z.open(inner) as src, open(fn, "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.remove(tmp)
    else:
        shutil.move(tmp, fn)
    return fn


# ------------------------------------------------------------------------------
# Chunk iteration over a (month-filtered) pandas DataFrame
# ------------------------------------------------------------------------------
def iter_df_chunks(df: pd.DataFrame, chunksize: int) -> Iterator[pd.DataFrame]:
    if chunksize <= 0:
        yield df
        return
    n = len(df)
    for start in range(0, n, chunksize):
        yield df.iloc[start : start + chunksize]


# ------------------------------------------------------------------------------
# ERA5 interpolation helper
# ------------------------------------------------------------------------------
def interp_points(
    ds: xr.Dataset,
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    method: str = "linear",
    return_raw: bool = False,
) -> pd.DataFrame:
    """
    Interpolate dataset `ds` to the provided point vectors.

    Returns a DataFrame with columns ['timestamp','lat','lon', *vars_].
    """
    # NOTE: method may be 'linear' or 'nearest'; we expose this so callers can fallback.
    # --- normalize times for xarray ---
    # xarray interp expects a numpy datetime64 array *without* timezone info.
    # We keep a copy of the original times (for later merge back to taxi rows),
    # then create a timezone-naive numpy datetime64[ns] vector for interpolation.
    times_orig = times  # keep original (may be tz-aware pandas Timestamps / object)
    # Convert to pandas DatetimeIndex for robust tz handling
    times_idx = pd.to_datetime(times_orig)
    if getattr(times_idx, "tz", None) is not None:
        # ensure UTC then drop tz for xarray
        times_idx_xr = times_idx.tz_convert("UTC").tz_localize(None)
    else:
        # already naive; assume UTC
        times_idx_xr = times_idx
    times_xr = times_idx_xr.to_numpy(dtype="datetime64[ns]")

    # rename coords so we can refer generically
    rename_map = {}
    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if "time" in ds.coords:
        rename_map["time"] = "timestamp"
    if "valid_time" in ds.coords:
        rename_map["valid_time"] = "timestamp"
    if rename_map:
        ds = ds.rename(rename_map)

    # Guarantee sorted increasing for xarray interp (lat often descending)
    # We'll not reorder dataset; xarray handles it internally.

    out = (
        ds.interp(
            timestamp=("points", times_xr),
            lon=("points", lons),
            lat=("points", lats),
            method=method,
        )
        .to_dataframe()
        .reset_index(drop=True)
    )
    if return_raw:
        return out  # caller wants raw xarray->DataFrame (coords + raw var names)
    # Reattach / overwrite coord columns. Use direct assignment instead of DataFrame.insert()
    # so we don't error if a column already exists (e.g., when xarray emitted it).
    if getattr(times_idx, "tz", None) is not None:
        times_for_merge = times_idx.tz_convert("UTC")
    else:
        times_for_merge = times_idx.tz_localize("UTC")

    out["timestamp"] = times_for_merge.to_numpy()
    out["lat"] = lats
    out["lon"] = lons

    # Ensure canonical column order: timestamp, lat, lon, then the rest (weather vars)
    base_cols = ["timestamp", "lat", "lon"]
    rest_cols = [c for c in out.columns if c not in base_cols]
    out = out[base_cols + rest_cols]
    return out


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Attach ERA5-Land hourly weather to Porto taxi trajectories.")
    ap.add_argument("--traj", required=True, help="Input parquet (porto_clip_downsample output).")
    ap.add_argument("--out", required=True, help="Output parquet with weather merged.")
    ap.add_argument("--cache", required=True, help="Directory to store / reuse monthly ERA5 NetCDF files.")
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive)")
    ap.add_argument("--vars", default="t2m,tp", help="Comma-separated ERA5-Land variables or shorthands.")
    ap.add_argument("--min-points", type=int, default=0, help="Drop trips with fewer than this many rows *before* interpolation.")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="Rows per interpolation chunk (pre-split further in --safe mode).")
    ap.add_argument("--safe", action="store_true", help="Enable extra safety splitting (recommended on laptops).")
    ap.add_argument("--offline-ok", action="store_true", help="Do not download ERA5 if cache missing; error instead.")
    ap.add_argument("--interp", default="linear", choices=["linear", "nearest"],
                    help="Primary interpolation method for ERA5 -> taxi points.")
    ap.add_argument("--fallback-nearest", action="store_true",
                    help="After primary interpolation, fill NaNs via nearest-neighbor.")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load trajectory columns we need (stream into pandas)
    # We'll use pyarrow.dataset for row-group scanning by month to keep memory low.
    # ------------------------------------------------------------------
    traj_path = pathlib.Path(args.traj)
    ds_arrow = pads.dataset(traj_path, format="parquet")

    cols = [
        "trip_id",
        "taxi_id",
        "seq",
        "timestamp_trip_start",
        "timestamp",
        "lat",
        "lon",
        "day_type",
        "call_type",
    ]
    # Guarantee columns exist
    schema_names = set(ds_arrow.schema.names)
    missing_cols = [c for c in cols if c not in schema_names]
    if missing_cols:
        raise SystemExit(f"Input parquet missing required columns: {missing_cols}")

    # Build month list from requested date window
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    # monthly periods between start (inclusive) and end (exclusive)
    ym_index = pd.period_range(start, end - pd.Timedelta("1D"), freq="M")
    ym_list: List[Tuple[int, int]] = [(p.year, p.month) for p in ym_index]

    # Compute bounding box for ERA5 fetch (north, west, south, east)
    # NOTE: CDS "area" ordering is [N, W, S, E]
    area = [args.lat_max, args.lon_min, args.lat_min, args.lon_max]

    # Parse variables
    vars_ = _parse_vars(args.vars)

    # Prepare output parquet writer (deferred until first write)
    out_path = pathlib.Path(args.out)
    if out_path.exists():
        out_path.unlink()
    writer = None

    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    # Trip length filtering requires we know counts per trip; easiest way is one pass load of just trip_id column lengths.
    # We'll compute counts using a streaming aggregate to avoid loading everything.
    if args.min_points > 0:
        print(f"[filter] computing trip lengths (min_points={args.min_points}) ...")
        lengths = {}
        for frag in ds_arrow.to_batches(columns=["trip_id"]):
            arr = frag.column(0).to_pylist()
            for t in arr:
                lengths[t] = lengths.get(t, 0) + 1
        keep_trips = {t for t, n in lengths.items() if n >= args.min_points}
        print(f"[filter] keeping {len(keep_trips)} trips (>= {args.min_points} points).")
    else:
        keep_trips = None

    # Helper to apply row-level filters (month & trip_id)
    def _get_month_df(year: int, month: int) -> pd.DataFrame:
        # Build arrow filter expressions
        import pyarrow.compute as pc

        month_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        # month_end = start of *next* month (tz-aware already); use < month_end in filter
        month_end = month_start + pd.offsets.MonthBegin(1)
        # Build filter: timestamp >= month_start & timestamp < month_end
        filt = (
            (pc.field("timestamp") >= pa.scalar(month_start.to_pydatetime()))
            & (pc.field("timestamp") < pa.scalar(month_end.to_pydatetime()))
        )
        ds_month = ds_arrow.to_table(columns=cols, filter=filt)
        # Convert to a *plain* pandas DataFrame (avoid Arrow extension dtypes that
        # later confuse pandas merge when joined with weather frames).
        df = ds_month.to_pandas()
        df = df.reset_index(drop=True)
        # ensure pandas datetime64[ns] and strip timezone (we'll treat all as UTC)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if hasattr(df["timestamp"].dt, "tz") and df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        if keep_trips is not None:
            df = df[df["trip_id"].isin(keep_trips)]
        return df

    # ------------------------------------------------------------------
    # Process each month (SAFE positional-attach; no float merge)
    # ------------------------------------------------------------------
    for (y, m) in ym_list:
        print(f"\n[month {y}-{m:02d}] loading taxi rows ...")
        df_month = _get_month_df(y, m)
        if df_month.empty:
            print("  (no data this month; skipping)")
            continue

        # Download / open ERA5 monthly file (cache-aware)
        nc = fetch_month_nc(
            y,
            m,
            area=area,
            vars_=vars_,
            cache=cache,
            offline_ok=args.offline_ok,
        )

        # Open dataset; be flexible wrt backend (netCDF4 optional)
        try:
            ds = xr.open_dataset(nc)  # auto engine
        except ValueError:  # fallback if engine resolution fails
            ds = xr.open_dataset(nc, engine="scipy")

        # ERA5 longitudes are 0..360 (degrees_east). Convert to conventional [-180, 180]
        # so we can reliably clip using the user's negative-degree lon bounds (e.g., Porto ~ -8.x).
        if "longitude" in ds.coords:
            lon_name = "longitude"
        elif "lon" in ds.coords:
            lon_name = "lon"
        else:
            raise RuntimeError("ERA5 dataset missing longitude coordinate.")

        # Generate a new longitude coordinate in [-180, 180] and sort
        lon_vals = ds[lon_name]
        if float(lon_vals.min()) >= 0.0 and float(lon_vals.max()) > 180.0:
            # wrap: (lon + 180) % 360 - 180
            lon_wrapped = ((lon_vals + 180) % 360) - 180
            ds = ds.assign_coords({lon_name: lon_wrapped})
            ds = ds.sortby(lon_name)

        # Now safe to subset region of interest using provided bounds
        # NOTE: ERA5 latitude is descending (N->S); slice handles either order.
        lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
        if lat_name is None:
            raise RuntimeError("ERA5 dataset missing latitude coordinate.")

        ds = ds.sel(
            **{
                lat_name: slice(args.lat_max, args.lat_min),
                lon_name: slice(args.lon_min, args.lon_max),
            }
        )

        ds.load()  # tiny grid after clip; safe

        # Rename to canonical coord names for interpolation (lat/lon/timestamp)
        rename_map = {}
        if "latitude" in ds.coords:
            rename_map["latitude"] = "lat"
        if "longitude" in ds.coords:
            rename_map["longitude"] = "lon"
        if "lat" in ds.coords and "lat" not in rename_map and "latitude" not in ds.coords:
            # already lat; nothing
            pass
        if "lon" in ds.coords and "lon" not in rename_map and "longitude" not in ds.coords:
            pass
        if "time" in ds.coords:
            rename_map["time"] = "timestamp"
        if "valid_time" in ds.coords:
            rename_map["valid_time"] = "timestamp"
        if rename_map:
            ds = ds.rename(rename_map)

        # Harmonize variable names (short → canonical)
        var_rename = {}
        if "t2m" in ds.data_vars and "2m_temperature" not in ds.data_vars:
            var_rename["t2m"] = "2m_temperature"
        if "tp" in ds.data_vars and "total_precipitation" not in ds.data_vars:
            var_rename["tp"] = "total_precipitation"
        if var_rename:
            ds = ds.rename(var_rename)

        # Drop ensemble/expver dims if present (keep first member)
        for dim_name in ("number", "expver"):
            if dim_name in ds.dims:
                ds = ds.isel({dim_name: 0}).drop_vars(dim_name, errors="ignore")
        # squeeze out any length-1 dims
        ds = ds.squeeze(drop=True)

        # Coerce all weather vars to float32 for memory savings
        ds = ds.astype({v: "float32" for v in ds.data_vars})

        # Pick requested variables (fallback keep all)
        keep_vars = [v for v in vars_ if v in ds.data_vars]
        if not keep_vars:
            print(f"[warn] requested vars {vars_} not found in dataset; keeping all variables: {list(ds.data_vars)}")
            keep_vars = list(ds.data_vars)
        ds = ds[keep_vars]

        # Interpolate month in chunks (extra splitting in --safe mode)
        chunk_rows = args.chunksize
        if args.safe and chunk_rows > 250_000:
            chunk_rows = 250_000

        month_chunks = []
        for i, sub in enumerate(iter_df_chunks(df_month, chunk_rows), start=1):
            print(f"  [interp] chunk {i}: rows={len(sub)}")
            # reset index so boolean masks align by position
            sub = sub.reset_index(drop=True)
            wx = interp_points(
                ds,
                lats=sub["lat"].to_numpy(),
                lons=sub["lon"].to_numpy(),
                times=sub["timestamp"].to_numpy(),
                method=args.interp,
                return_raw=True,  # get raw to inspect NaNs
            )
            # Sanity: 1:1 alignment expected
            if len(wx) != len(sub):
                raise RuntimeError(
                    f"interp length mismatch: got {len(wx)} weather rows for {len(sub)} taxi rows"
                )

            # Harmonize weather var columns
            if "t2m" in wx.columns and "2m_temperature" not in wx.columns:
                wx["2m_temperature"] = wx["t2m"]
            if "tp" in wx.columns and "total_precipitation" not in wx.columns:
                wx["total_precipitation"] = wx["tp"]

            # Units: Kelvin -> °C, m -> mm
            t2m_vals = wx.get("2m_temperature")
            tp_vals = wx.get("total_precipitation")

            if t2m_vals is not None:
                t2m_C = t2m_vals.astype("float32") - 273.15
            else:
                t2m_C = pd.Series(np.nan, index=wx.index, dtype="float32")

            if tp_vals is not None:
                tp_mm = tp_vals.astype("float32") * 1000.0
            else:
                tp_mm = pd.Series(np.nan, index=wx.index, dtype="float32")

            # Fallback interpolation for NaNs using nearest-neighbor if requested
            if args.fallback_nearest:
                mask_series = (t2m_C.isna() | tp_mm.isna())
                if mask_series.any():
                    # Use position-based mask to avoid index alignment issues
                    mask_arr = mask_series.to_numpy()
                    sub_mask = sub.iloc[mask_arr]
                    wx_nn = interp_points(
                        ds,
                        lats=sub_mask["lat"].to_numpy(),
                        lons=sub_mask["lon"].to_numpy(),
                        times=sub_mask["timestamp"].to_numpy(),
                        method="nearest",
                        return_raw=True,
                    )
                    # Harmonize weather var columns
                    if "t2m" in wx_nn.columns and "2m_temperature" not in wx_nn.columns:
                        wx_nn["2m_temperature"] = wx_nn["t2m"]
                    if "tp" in wx_nn.columns and "total_precipitation" not in wx_nn.columns:
                        wx_nn["total_precipitation"] = wx_nn["tp"]
                    if "2m_temperature" in wx_nn.columns:
                        t2m_C_nn = wx_nn["2m_temperature"].astype("float32") - 273.15
                    else:
                        t2m_C_nn = pd.Series(np.nan, index=wx_nn.index, dtype="float32")
                    if "total_precipitation" in wx_nn.columns:
                        tp_mm_nn = wx_nn["total_precipitation"].astype("float32") * 1000.0
                    else:
                        tp_mm_nn = pd.Series(np.nan, index=wx_nn.index, dtype="float32")
                    # assign back by position
                    t2m_C.iloc[mask_arr] = t2m_C_nn.to_numpy()
                    tp_mm.iloc[mask_arr] = tp_mm_nn.to_numpy()

            # Attach columns by position (avoid float merge precision bugs)
            sub_out = sub.copy()
            sub_out["t2m_C"] = t2m_C.to_numpy()
            sub_out["tp_mm"] = tp_mm.to_numpy()

            month_chunks.append(sub_out)

        # Release dataset resources
        ds.close()
        del ds

        # month-level coverage diagnostic
        month_df_tmp = pd.concat(month_chunks, ignore_index=True)
        pct_nonnull = month_df_tmp["t2m_C"].notna().mean()*100
        print(f"  [month {y}-{m:02d}] weather non-null: {pct_nonnull:.2f}%")
        month_df = month_df_tmp
        del month_df_tmp
        del month_chunks
        gc.collect()

        # Write month to output parquet (stream append)
        table = pa.Table.from_pandas(month_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

        # cleanup month_df
        del month_df, table
        gc.collect()

    if writer is not None:
        writer.close()
    print(f"\n✅ Done. Weather merged parquet → {out_path}")


if __name__ == "__main__":
    main()