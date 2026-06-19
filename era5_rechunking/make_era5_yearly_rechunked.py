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
# Rechunks [93, 91, 180] → [1, 721, 1440] using xarray. Raw int16 packed
# values are preserved (mask_and_scale=False prevents unpack to float64).
#
# To process a single stream and year:
#   python3 make_era5_yearly_rechunked.py --stream csf --year 1979
#
# To process all 13 streams, 1940-2026, with 24 parallel workers:
#   python3 make_era5_yearly_rechunked.py --workers 24
#
# Contact: Ezhilsabareesh Kannadasan <ezhilsabareesh.kannadasan@anu.edu.au>
#
# Dependencies: xarray, netCDF4
# =========================================================================================

"""
Rechunk and concatenate ERA5 single-level monthly files from rt52 into yearly files.

Usage:
    python3 make_era5_yearly_rechunked.py --stream csf --year 1979
    python3 make_era5_yearly_rechunked.py --workers 24
    python3 make_era5_yearly_rechunked.py --stream csf --year 1979 --overwrite
"""

import argparse
import datetime
import glob
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import netCDF4 as nc
import xarray as xr

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

YEAR_FIRST, YEAR_LAST = 1940, 2026
COMPLEVEL = 5
CHUNK_T, CHUNK_LAT, CHUNK_LON = 1, 721, 1440


# ── validation ────────────────────────────────────────────────────────────────


def validate(out_path, varname):
    """
    Sanity-check a freshly written yearly file.  Raises RuntimeError on failure.
    Checks: dtype int16, chunksizes [CHUNK_T,CHUNK_LAT,CHUNK_LON], time UNLIMITED,
    scale_factor/add_offset present, time monotonically increasing.
    """
    with nc.Dataset(str(out_path)) as ds:
        var = ds.variables[varname]

        if var.dtype != "int16":
            raise RuntimeError(f"dtype={var.dtype}, expected int16")

        chunks = var.chunking()
        expected = [CHUNK_T, CHUNK_LAT, CHUNK_LON]
        if chunks != expected:
            raise RuntimeError(f"chunksizes={chunks}, expected {expected}")

        if not ds.dimensions["time"].isunlimited():
            raise RuntimeError("time dimension is not UNLIMITED")

        for attr in ("scale_factor", "add_offset"):
            if attr not in var.ncattrs():
                raise RuntimeError(f"missing variable attribute {attr!r}")

        t = ds.variables["time"][:]
        if len(t) > 1 and not bool((t[1:] > t[:-1]).all()):
            raise RuntimeError("time coordinate is not monotonically increasing")

    logging.info(f"validate: {Path(out_path).name} — OK")


# ── single-task processing ─────────────────────────────────────────────────────


def process_one(stream, year, source_base, output_base, overwrite, runcmd):
    """
    Rechunk and concatenate all monthly rt52 files for one (stream, year) into
    a single yearly output file. Returns True on success, False on failure.
    """
    varname = STREAM_TO_VARNAME[stream]
    source_dir = Path(source_base) / stream / str(year)

    if not source_dir.is_dir():
        logging.error(f"Source directory not found: {source_dir}")
        return False

    source_files = sorted(source_dir.glob("*.nc"))
    if not source_files:
        logging.error(f"No .nc files in {source_dir}")
        return False

    # Derive output filename from first/last source file date stamps
    m0 = re.search(r"(\d{8})-(\d{8})", source_files[0].name)
    ml = re.search(r"(\d{8})-(\d{8})", source_files[-1].name)
    start = m0.group(1) if m0 else f"{year}0101"
    end = ml.group(2) if ml else f"{year}1231"

    out_dir = Path(output_base) / stream
    out_path = out_dir / f"{stream}_era5_oper_sfc_{start}-{end}.nc"

    if out_path.exists() and not overwrite:
        logging.info(f"SKIP {stream}/{year}: output already exists — {out_path}")
        return True

    logging.info(
        f"Processing {stream}/{year}: {len(source_files)} files → {out_path.name}"
    )

    # mask_and_scale=False preserves packed int16 values;
    # without this xarray unpacks to float64 (8x storage, wrong output dtype).
    # Full-spatial Dask chunks (one per monthly file) avoids the 8x8=64 source-chunk
    # reads that would otherwise be needed per output timestep when the source spatial
    # chunks (91x180) differ from the output (721x1440). The encoding dict controls
    # the on-disk HDF5 chunk layout independently of the Dask chunk size.
    ds = xr.open_mfdataset(
        [str(f) for f in source_files],
        combine="by_coords",
        mask_and_scale=False,
        chunks={"time": 744, "latitude": 721, "longitude": 1440},
    )

    # Provenance
    this_file = os.path.normpath(__file__)
    now_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    new_history = (
        f"{now_iso} rechunked from [93,91,180] to [1,721,1440] using xarray; "
        f"{len(source_files)} monthly files from rt52 concatenated into one yearly file. "
        + get_provenance_metadata(this_file, runcmd)
    )
    old_title = ds.attrs.get("title", "")
    old_history = ds.attrs.get("history", "")
    ds.attrs.update(
        {
            "title": re.sub(r"\s+\d{8}-\d{8}$", f" {year}", old_title),
            "history": f"{new_history}\n{old_history}" if old_history else new_history,
            "rechunked_from": f"{SOURCE_BASE}/{stream}/{year}/",
            "rechunked_by": "Ezhilsabareesh Kannadasan (ek4684)",
            "rechunked_date": now_iso,
            "original_chunking": "[93, 91, 180]",
            "target_chunking": "[1, 721, 1440]",
        }
    )

    # _FillValue must be in encoding (not attrs) for netCDF4 writing;
    # with mask_and_scale=False xarray leaves it in attrs
    fill_value = ds[varname].attrs.pop("_FillValue", -32767)

    encoding = {
        varname: {
            "dtype": "int16",
            "_FillValue": fill_value,
            "chunksizes": (CHUNK_T, CHUNK_LAT, CHUNK_LON),
            "zlib": True,
            "shuffle": True,
            "complevel": COMPLEVEL,
        }
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds.to_netcdf(str(out_path), encoding=encoding, unlimited_dims=["time"])
    elapsed = time.time() - t0

    validate(out_path, varname)

    size_gb = out_path.stat().st_size / 1e9
    logging.info(
        f"{stream}/{year}: DONE — {out_path} ({size_gb:.2f} GB, {elapsed:.0f}s)"
    )
    return True


# ── multi-worker support ───────────────────────────────────────────────────────


def _subprocess_task(args_tuple):
    """Invoke this script as a subprocess for one (stream, year) task."""
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
    parser.add_argument("--stream", choices=ALL_STREAMS, help="Stream to process.")
    parser.add_argument("--year", type=int, help="Year to process.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for full run (default: 1).",
    )
    parser.add_argument("--source-dir", default=SOURCE_BASE)
    parser.add_argument("--output-dir", default=OUTPUT_BASE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # ── single-task mode ──────────────────────────────────────────────────────
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

    # ── multi-task mode ───────────────────────────────────────────────────────
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tasks = []
    skipped = 0
    for stream in ALL_STREAMS:
        for year in range(YEAR_FIRST, YEAR_LAST + 1):
            if _output_exists(stream, year, args.output_dir) and not args.overwrite:
                skipped += 1
            else:
                tasks.append(
                    (stream, year, args.source_dir, args.output_dir, args.overwrite)
                )

    print(f"Tasks to run:  {len(tasks)}")
    print(f"Tasks skipped: {skipped}  (output already exists)")
    print(f"Workers:       {args.workers}")

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
