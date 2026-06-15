#!/usr/bin/env python3
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Rechunk ERA5 single-level monthly files from the NCI rt52 archive and
# concatenate them into one yearly file per stream variable.
#
# Source:  /g/data/rt52/era5/single-levels/reanalysis/{stream}/{year}/
# Output:  {output_dir}/{stream}/{stream}_era5_oper_sfc_{YYYYMMDD}-{YYYYMMDD}.nc
#
# The rechunking changes the HDF5 chunk layout from [93, 91, 180] (ERA5 default)
# to [1, 721, 1440] (one full-resolution timestep per chunk), which is optimal
# for the sequential timestep access pattern used by ACCESS-OM3 / DATM.
#
# Raw int16 values are preserved bit-for-bit — no unpack/repack is performed.
# scale_factor, add_offset, _FillValue, units, long_name, standard_name and all
# other variable attributes are carried through unchanged.
#
# After writing, a validation step checks that the int16 values, chunking,
# dtype, time axis and metadata in the output match the rt52 source.
#
# To run a single stream/year:
#   python3 make_era5_yearly_rechunked.py --stream csf --year 1979
#
# To run all 13 streams, 1940-2026, with 24 parallel workers:
#   python3 make_era5_yearly_rechunked.py --workers 24
#
# See --help for all options.
#
# Contact: Ezhilsabareesh Kannadasan <ezhilsabareesh.kannadasan@anu.edu.au>
#
# Dependencies: netCDF4, numpy
# =========================================================================================

"""
Rechunk and concatenate ERA5 single-level monthly files from rt52 into yearly files.

Usage:
    # Single stream and year
    python3 make_era5_yearly_rechunked.py --stream csf --year 1979

    # All 13 streams, 1940-2026, 24 parallel workers
    python3 make_era5_yearly_rechunked.py --workers 24

    # Dry run — print tasks, write nothing
    python3 make_era5_yearly_rechunked.py --dry-run

    # Subset of streams/years
    python3 make_era5_yearly_rechunked.py --streams 10u 2t --years 2000 2001 --workers 4

    # Force-overwrite existing output
    python3 make_era5_yearly_rechunked.py --stream 10u --year 2000 --overwrite
"""

import argparse
import datetime
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import netCDF4
import numpy as np

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata

# ── constants ──────────────────────────────────────────────────────────────────

SOURCE_BASE = "/g/data/rt52/era5/single-levels/reanalysis"
OUTPUT_BASE = "/g/data/tm70/ek4684/era5_rechunked_1h_yearly"

ALL_STREAMS = [
    "10u",
    "10v",
    "2d",
    "2t",
    "ci",
    "cp",
    "csf",
    "lsf",
    "lsp",
    "msl",
    "ssr",
    "ssrd",
    "strd",
]

STREAM_TO_VARNAME = {
    "10u": "u10",
    "10v": "v10",
    "2d": "d2m",
    "2t": "t2m",
    "ci": "siconc",
    "cp": "cp",
    "csf": "csf",
    "lsf": "lsf",
    "lsp": "lsp",
    "msl": "msl",
    "ssr": "ssr",
    "ssrd": "ssrd",
    "strd": "strd",
}

YEAR_FIRST = 1940
YEAR_LAST = 2026
COMPLEVEL = 5
CHUNK_T = 1
CHUNK_LAT = 721
CHUNK_LON = 1440


# ── source file discovery ──────────────────────────────────────────────────────


def find_source_files(stream, year, source_base=SOURCE_BASE):
    """Return sorted list of monthly rt52 NetCDF files for (stream, year)."""
    year_dir = os.path.join(source_base, stream, str(year))
    if not os.path.isdir(year_dir):
        raise FileNotFoundError(f"rt52 source directory not found: {year_dir}")
    files = sorted(
        os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith(".nc")
    )
    if not files:
        raise FileNotFoundError(f"No .nc files in {year_dir}")
    return files


def derive_output_path(stream, year, source_files, output_base=OUTPUT_BASE):
    """
    Derive yearly output path from the date stamps in the first/last source
    filenames, e.g. csf_era5_oper_sfc_19790101-19791231.nc
    """
    m_first = re.search(r"(\d{8})-(\d{8})", os.path.basename(source_files[0]))
    m_last = re.search(r"(\d{8})-(\d{8})", os.path.basename(source_files[-1]))
    start = m_first.group(1) if m_first else f"{year}0101"
    end = m_last.group(2) if m_last else f"{year}1231"
    fname = f"{stream}_era5_oper_sfc_{start}-{end}.nc"
    return os.path.join(output_base, stream, fname)


# ── rechunking ─────────────────────────────────────────────────────────────────


def build_yearly_file(source_files, stream, year, out_path, varname, history_str):
    """
    Read raw int16 from monthly rt52 source files and write a single rechunked
    yearly NetCDF4 file with time as UNLIMITED dimension.

    Chunk layout [1, 721, 1440] — one full-resolution timestep per HDF5 chunk.
    Complevel 5, zlib + shuffle.  No unpack/repack: int16 values are written
    directly to avoid the ~8x memory overhead of float64 conversion.
    """
    with netCDF4.Dataset(source_files[0]) as s0:
        lon_vals = s0.variables["longitude"][:]
        lat_vals = s0.variables["latitude"][:]
        lon_attrs = {
            a: getattr(s0.variables["longitude"], a)
            for a in s0.variables["longitude"].ncattrs()
        }
        lat_attrs = {
            a: getattr(s0.variables["latitude"], a)
            for a in s0.variables["latitude"].ncattrs()
        }
        var_attrs = {
            a: getattr(s0.variables[varname], a)
            for a in s0.variables[varname].ncattrs()
        }
        global_attrs = {a: getattr(s0, a) for a in s0.ncattrs()}
        time_units = s0.variables["time"].units
        time_calendar = s0.variables["time"].calendar

    # Collect raw integer time values across all months
    all_time_raw = []
    for f in source_files:
        with netCDF4.Dataset(f) as ds:
            ds.variables["time"].set_auto_maskandscale(False)
            all_time_raw.extend(ds.variables["time"][:].tolist())
    total_t = len(all_time_raw)

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as dst:

        # Dimensions — time is UNLIMITED
        dst.createDimension("longitude", len(lon_vals))
        dst.createDimension("latitude", len(lat_vals))
        dst.createDimension("time", None)

        # Coordinate: longitude
        v_lon = dst.createVariable(
            "longitude",
            "f4",
            ("longitude",),
            zlib=True,
            shuffle=True,
            complevel=COMPLEVEL,
            chunksizes=(len(lon_vals),),
        )
        for k, v in lon_attrs.items():
            setattr(v_lon, k, v)
        v_lon[:] = lon_vals

        # Coordinate: latitude
        v_lat = dst.createVariable(
            "latitude",
            "f4",
            ("latitude",),
            zlib=True,
            shuffle=True,
            complevel=COMPLEVEL,
            chunksizes=(len(lat_vals),),
        )
        for k, v in lat_attrs.items():
            setattr(v_lat, k, v)
        v_lat[:] = lat_vals

        # Coordinate: time
        v_time = dst.createVariable(
            "time",
            "i4",
            ("time",),
            zlib=True,
            shuffle=True,
            complevel=COMPLEVEL,
            chunksizes=(min(total_t, 744),),
        )
        v_time.units = time_units
        v_time.long_name = "time"
        v_time.calendar = time_calendar
        v_time[:] = np.array(all_time_raw, dtype="int32")

        # Data variable — raw int16; _FillValue must be supplied at creation time
        fill_value = var_attrs.get("_FillValue", netCDF4.default_fillvals["i2"])
        v_data = dst.createVariable(
            varname,
            "i2",
            ("time", "latitude", "longitude"),
            zlib=True,
            shuffle=True,
            complevel=COMPLEVEL,
            chunksizes=(CHUNK_T, CHUNK_LAT, CHUNK_LON),
            fill_value=fill_value,
        )
        for k, v in var_attrs.items():
            if k == "_FillValue":
                continue
            setattr(v_data, k, v)

        # Disable auto-unpack on the output variable: without this, netCDF4
        # converts int16 → float64 internally when scale_factor/add_offset are
        # present, multiplying memory use by ~8.
        v_data.set_auto_maskandscale(False)

        # Copy raw int16 data month by month; sync after each month to flush
        # HDF5 write cache to disk and keep peak memory bounded (~1.5 GB/month)
        t_offset = 0
        for f in source_files:
            with netCDF4.Dataset(f) as src:
                sv = src.variables[varname]
                sv.set_auto_maskandscale(False)
                raw = sv[:]
                n = raw.shape[0]
                v_data[t_offset : t_offset + n, :, :] = raw
                t_offset += n
                logging.info(f"  written {t_offset} / {total_t} timesteps")
            dst.sync()

        # Global attributes
        old_title = global_attrs.get("title", "")
        global_attrs["title"] = re.sub(r"\s+\d{8}-\d{8}$", f" {year}", old_title)

        old_history = global_attrs.get("history", "")
        global_attrs["history"] = (
            f"{history_str}\n{old_history}" if old_history else history_str
        )
        for k, v in global_attrs.items():
            setattr(dst, k, v)

        # Provenance attributes
        dst.rechunked_from = f"{SOURCE_BASE}/{stream}/{year}/"
        dst.rechunked_by = "Ezhilsabareesh Kannadasan (ek4684)"
        dst.rechunked_date = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        dst.original_chunking = "[93, 91, 180]"
        dst.target_chunking = "[1, 721, 1440]"

    return total_t


# ── validation ─────────────────────────────────────────────────────────────────


def validate(out_path, source_files, varname):
    """
    Validate the rechunked yearly output against the rt52 source files.

    Checks performed:
      - time dimension is UNLIMITED
      - total timestep count matches the sum across all monthly source files
      - chunk layout is [1, 721, 1440]
      - dtype is int16
      - scale_factor, add_offset, _FillValue are present on the data variable
      - rechunked_from, rechunked_by, rechunked_date, original_chunking,
        target_chunking are present as global attributes
      - time axis is strictly monotonically increasing
      - raw int16 sample values match the source for the first 3 months
        (first, mid, and last timestep of each month)
      - global raw int16 min/max match the source for the first 3 months

    Returns (n_pass, errors) where errors is a list of failure description strings.
    """
    errors = []

    with netCDF4.Dataset(out_path) as dst:

        if not dst.dimensions["time"].isunlimited():
            errors.append("time dimension is NOT unlimited")

        total_t_out = dst.dimensions["time"].size
        total_t_src = sum(
            netCDF4.Dataset(f).dimensions["time"].size for f in source_files
        )
        if total_t_out != total_t_src:
            errors.append(
                f"timestep count mismatch: output={total_t_out}, source={total_t_src}"
            )

        v_data = dst.variables[varname]
        chunks = v_data.chunking()
        if chunks != [CHUNK_T, CHUNK_LAT, CHUNK_LON]:
            errors.append(
                f"chunking wrong: got {chunks}, expected [{CHUNK_T},{CHUNK_LAT},{CHUNK_LON}]"
            )

        if str(v_data.dtype) != "int16":
            errors.append(f"dtype wrong: got {v_data.dtype}, expected int16")

        for attr in ("scale_factor", "add_offset", "_FillValue"):
            if not hasattr(v_data, attr):
                errors.append(f"variable attribute '{attr}' missing")

        for attr in (
            "rechunked_from",
            "rechunked_by",
            "rechunked_date",
            "original_chunking",
            "target_chunking",
        ):
            if not hasattr(dst, attr):
                errors.append(f"global attribute '{attr}' missing")

        t_vals = dst.variables["time"][:]
        if np.any(np.diff(t_vals.astype("int64")) <= 0):
            errors.append("time axis not strictly monotonically increasing")

        # Sample raw int16 from the first 3 monthly source files
        v_data.set_auto_maskandscale(False)
        t_offset = 0
        for fi, f in enumerate(source_files[:3]):
            with netCDF4.Dataset(f) as src:
                sv = src.variables[varname]
                sv.set_auto_maskandscale(False)
                n = src.dimensions["time"].size
                for ti in [0, n // 2, n - 1]:
                    if not np.array_equal(sv[ti, :, :], v_data[t_offset + ti, :, :]):
                        errors.append(
                            f"raw int16 mismatch: month {fi + 1}, t_local={ti}"
                        )
                src_min, src_max = int(sv[:].min()), int(sv[:].max())
                dst_min = int(v_data[t_offset : t_offset + n].min())
                dst_max = int(v_data[t_offset : t_offset + n].max())
                if src_min != dst_min or src_max != dst_max:
                    errors.append(
                        f"int16 range mismatch: month {fi + 1}: "
                        f"source=[{src_min},{src_max}], output=[{dst_min},{dst_max}]"
                    )
                t_offset += n

    n_checks = 8 + len(source_files[:3]) * 4
    return n_checks - len(errors), errors


# ── single-task processing ─────────────────────────────────────────────────────


def process_one(stream, year, source_base, output_base, overwrite, runcmd):
    """
    Rechunk and concatenate all monthly rt52 files for one (stream, year) into a
    single yearly output file.  Returns True on success, False on failure.
    """
    varname = STREAM_TO_VARNAME[stream]

    try:
        source_files = find_source_files(stream, year, source_base)
    except FileNotFoundError as e:
        logging.error(str(e))
        return False

    out_path = derive_output_path(stream, year, source_files, output_base)
    out_dir = os.path.dirname(out_path)

    if os.path.exists(out_path) and not overwrite:
        logging.info(f"SKIP {stream}/{year}: output already exists — {out_path}")
        return True

    logging.info(
        f"Starting {stream}/{year}: {len(source_files)} monthly files → {os.path.basename(out_path)}"
    )
    for f in source_files:
        logging.info(f"  {os.path.basename(f)}")

    this_file = os.path.normpath(__file__)
    now_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    history_str = (
        f"{now_iso} rechunked from [93,91,180] to [1,721,1440] using netCDF4; "
        f"{len(source_files)} monthly files from rt52 concatenated into one yearly file. "
        + get_provenance_metadata(this_file, runcmd)
    )

    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".tmp.nc", delete=False) as tf:
        tmp_path = tf.name

    try:
        t0 = time.time()
        n_t = build_yearly_file(
            source_files, stream, year, tmp_path, varname, history_str
        )
        elapsed = time.time() - t0
        logging.info(f"{stream}/{year}: wrote {n_t} timesteps in {elapsed:.0f}s")

        n_pass, errors = validate(tmp_path, source_files, varname)
        if errors:
            for e in errors:
                logging.error(f"VALIDATION FAIL [{stream}/{year}]: {e}")
            os.remove(tmp_path)
            return False

        logging.info(f"{stream}/{year}: {n_pass} validation checks passed")
        shutil.move(tmp_path, out_path)
        size_gb = os.path.getsize(out_path) / 1e9
        logging.info(f"{stream}/{year}: DONE — {out_path} ({size_gb:.2f} GB)")
        return True

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        logging.error(f"{stream}/{year}: FAILED — {e}")
        raise


# ── multi-worker support ───────────────────────────────────────────────────────


def _subprocess_task(args_tuple):
    """
    Worker function for parallel execution: invokes this script as a subprocess
    for a single (stream, year) task and returns (stream, year, returncode, elapsed).
    """
    stream, year, source_base, output_base, overwrite = args_tuple
    script = os.path.abspath(__file__)
    cmd = [
        sys.executable,
        script,
        "--stream",
        stream,
        "--year",
        str(year),
        "--source-dir",
        source_base,
        "--output-dir",
        output_base,
    ]
    if overwrite:
        cmd.append("--overwrite")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else "FAIL"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-500:]
        print(f"[{status}] {stream}/{year} ({elapsed:.0f}s)\n  {tail}", flush=True)
    else:
        last_line = (result.stdout.strip().split("\n") or [""])[-1]
        print(f"[{status}] {stream}/{year} ({elapsed:.0f}s)  {last_line}", flush=True)

    return stream, year, result.returncode, elapsed


def _output_exists(stream, year, output_base):
    """Return True if a yearly output file already exists for this (stream, year)."""
    import glob

    return bool(
        glob.glob(
            os.path.join(output_base, stream, f"{stream}_era5_oper_sfc_{year}*-*.nc")
        )
    )


# ── main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Rechunk ERA5 rt52 monthly files into yearly rechunked files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stream",
        choices=ALL_STREAMS,
        help="Process a single stream (combine with --year for a single task).",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Process a single year (combine with --stream for a single task).",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=ALL_STREAMS,
        default=ALL_STREAMS,
        metavar="STREAM",
        help=f"Streams to process in multi-task mode (default: all {len(ALL_STREAMS)}).",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(YEAR_FIRST, YEAR_LAST + 1)),
        metavar="YEAR",
        help=f"Years to process in multi-task mode (default: {YEAR_FIRST}-{YEAR_LAST}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--source-dir",
        default=SOURCE_BASE,
        help=f"Root of the rt52 ERA5 archive (default: {SOURCE_BASE}).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_BASE,
        help=f"Root of the yearly output directory (default: {OUTPUT_BASE}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process and overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tasks that would run without writing any files.",
    )
    args = parser.parse_args()

    # ── single-task mode (also used by each subprocess worker) ────────────────
    if args.stream and args.year:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        this_file = os.path.normpath(__file__)
        runcmd = (
            f"python3 {os.path.basename(this_file)} "
            f"--stream {args.stream} --year {args.year} "
            f"--source-dir {args.source_dir} --output-dir {args.output_dir}"
        )
        ok = process_one(
            args.stream,
            args.year,
            args.source_dir,
            args.output_dir,
            args.overwrite,
            runcmd,
        )
        sys.exit(0 if ok else 1)

    # ── multi-task mode ────────────────────────────────────────────────────────
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    streams = [args.stream] if args.stream else args.streams
    years = [args.year] if args.year else args.years

    tasks = []
    skipped = 0
    for stream in streams:
        for year in years:
            if _output_exists(stream, year, args.output_dir) and not args.overwrite:
                skipped += 1
            else:
                tasks.append(
                    (stream, year, args.source_dir, args.output_dir, args.overwrite)
                )

    print(f"Tasks to run:  {len(tasks)}")
    print(f"Tasks skipped: {skipped}  (output already exists)")
    print(f"Workers:       {args.workers}")
    print(f"Source:        {args.source_dir}")
    print(f"Output:        {args.output_dir}")

    if args.dry_run:
        print("\nDRY RUN — tasks that would run:")
        for stream, year, *_ in tasks:
            print(f"  {stream}/{year}")
        return

    if not tasks:
        print("Nothing to do.")
        return

    t_start = time.time()
    failures = []

    if args.workers == 1:
        for task in tasks:
            stream, year, rc, _ = _subprocess_task(task)
            if rc != 0:
                failures.append((stream, year))
    else:
        with multiprocessing.Pool(args.workers) as pool:
            for stream, year, rc, _ in pool.imap_unordered(_subprocess_task, tasks):
                if rc != 0:
                    failures.append((stream, year))

    total_elapsed = time.time() - t_start
    print(
        f"\nCompleted {len(tasks)} tasks in {total_elapsed:.0f}s ({total_elapsed / 3600:.1f}h)"
    )
    if failures:
        print(f"FAILED ({len(failures)}):")
        for s, y in failures:
            print(f"  {s}/{y}")
        sys.exit(1)
    else:
        print("All tasks succeeded.")


if __name__ == "__main__":
    main()
