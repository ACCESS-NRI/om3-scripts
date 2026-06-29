#!/usr/bin/env python3
# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Rechunk ERA5 single-level monthly files from the NCI rt52 archive and
# concatenate them into one yearly file per stream variable.
#
# Source:  /g/data/rt52/era5/single-levels/reanalysis/{stream}/{year}/
# Output:  {output_dir}/{stream}/{stream}_era5_oper_sfc_{YYYYMMDD}-{YYYYMMDD}.nc
#
# Rechunks [93, 91, 180] -> [1, 721, 1440] using netCDF4 streaming copies.
# Raw int16 packed values are preserved by disabling netCDF4 auto mask/scale.
#
# To process a single stream and year:
#   python3 make_era5_yearly_rechunked.py --stream csf --year 1979
#
# To process selected streams and years:
#   python3 make_era5_yearly_rechunked.py --stream csf lsp --year 1979 1980-1982
#
# To process all streams, 1940-2026, with a conservative worker count:
#   python3 make_era5_yearly_rechunked.py --workers 6
#
# Contact: Ezhilsabareesh Kannadasan <ezhilsabareesh.kannadasan@anu.edu.au>
#
# Dependencies: netCDF4, numpy
# =========================================================================================

"""
Rechunk and concatenate ERA5 single-level monthly files from rt52 into yearly files.

Usage:
    python3 make_era5_yearly_rechunked.py --stream csf --year 1979
    python3 make_era5_yearly_rechunked.py --stream csf lsp --year 1979 1980-1982
    python3 make_era5_yearly_rechunked.py --workers 6
    python3 make_era5_yearly_rechunked.py --stream csf --year 1979 --overwrite
"""

import argparse
import calendar
import datetime
import logging
import multiprocessing
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

import netCDF4 as nc
import numpy as np

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata

# -- constants -----------------------------------------------------------------

SOURCE_BASE = "/g/data/rt52/era5/single-levels/reanalysis"
DEFAULT_PROJECT = os.environ.get("PROJECT", "tm70")
DEFAULT_USER = os.environ.get("USER") or os.environ.get("LOGNAME", "ek4684")
OUTPUT_BASE = f"/scratch/{DEFAULT_PROJECT}/{DEFAULT_USER}/era5_rechunked_1h_yearly"

STREAM_TO_VARNAME = {
    "10u": "u10",
    "10v": "v10",
    "2d": "d2m",
    "2t": "t2m",
    "cp": "cp",
    "csf": "csf",
    "lsf": "lsf",
    "lsp": "lsp",
    "msl": "msl",
    "ssr": "ssr",
    "ssrd": "ssrd",
    "strd": "strd",
}
ALL_STREAMS = list(STREAM_TO_VARNAME.keys())

YEAR_FIRST, YEAR_LAST = 1940, 2026
COMPLEVEL = 5
CHUNK_T, CHUNK_LAT, CHUNK_LON = 1, 721, 1440
COPY_TIME_BLOCK = 93
PRESERVED_VAR_ATTRS = (
    "scale_factor",
    "add_offset",
    "missing_value",
    "units",
    "long_name",
    "standard_name",
)


# -- validation ----------------------------------------------------------------


def _full_year_hours(year):
    return 8784 if calendar.isleap(year) else 8760


def _local_iso_timestamp():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _attrs_equal(left, right):
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape or right_arr.shape:
        return left_arr.shape == right_arr.shape and bool(np.array_equal(left_arr, right_arr))
    return left == right


def _require_matching_attr(src_obj, dst_obj, attr, label):
    src_has = attr in src_obj.ncattrs()
    dst_has = attr in dst_obj.ncattrs()
    if src_has != dst_has:
        raise RuntimeError(f"{label}: attribute {attr!r} presence differs")
    if src_has and not _attrs_equal(src_obj.getncattr(attr), dst_obj.getncattr(attr)):
        raise RuntimeError(f"{label}: attribute {attr!r} differs")


def validate(
    out_path,
    varname,
    year=None,
    expected_time_count=None,
    allow_partial=False,
    log_success=True,
):
    """
    Sanity-check a yearly file. Raises RuntimeError on failure.

    Checks: variable exists, dtype int16, chunksizes [CHUNK_T, CHUNK_LAT,
    CHUNK_LON], time UNLIMITED, scale_factor/add_offset present, non-empty time,
    expected time count where known, and monotonically increasing time.
    """
    out_path = Path(out_path)
    with nc.Dataset(str(out_path)) as ds:
        if varname not in ds.variables:
            raise RuntimeError(f"missing variable {varname!r}")
        if "time" not in ds.dimensions:
            raise RuntimeError("missing time dimension")
        if "time" not in ds.variables:
            raise RuntimeError("missing time coordinate variable")

        var = ds.variables[varname]
        var.set_auto_maskandscale(False)
        if "time" not in var.dimensions:
            raise RuntimeError(f"variable {varname!r} has no time dimension")
        if str(var.dtype) != "int16":
            raise RuntimeError(f"dtype={var.dtype}, expected int16")

        chunks = var.chunking()
        expected_chunks = [CHUNK_T, CHUNK_LAT, CHUNK_LON]
        if chunks != expected_chunks:
            raise RuntimeError(f"chunksizes={chunks}, expected {expected_chunks}")

        if not ds.dimensions["time"].isunlimited():
            raise RuntimeError("time dimension is not UNLIMITED")

        for attr in ("scale_factor", "add_offset"):
            if attr not in var.ncattrs():
                raise RuntimeError(f"missing variable attribute {attr!r}")

        time_len = len(ds.dimensions["time"])
        if time_len == 0:
            raise RuntimeError("time dimension is empty")

        if expected_time_count is not None:
            if time_len != expected_time_count:
                raise RuntimeError(
                    f"time length={time_len}, expected {expected_time_count}"
                )
        elif year is not None and not allow_partial:
            expected = _full_year_hours(year)
            if time_len != expected:
                raise RuntimeError(f"time length={time_len}, expected {expected}")

        tvar = ds.variables["time"]
        tvar.set_auto_maskandscale(False)
        t = tvar[:]
        if len(t) != time_len:
            raise RuntimeError(
                f"time coordinate length={len(t)}, expected dimension length {time_len}"
            )
        if time_len > 1 and not bool((t[1:] > t[:-1]).all()):
            raise RuntimeError("time coordinate is not monotonically increasing")

    if log_success:
        logging.info("validate: %s - OK", out_path.name)


# -- source and output discovery ------------------------------------------------


def _source_files(stream, year, source_base):
    source_dir = Path(source_base) / stream / str(year)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = sorted(source_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files in {source_dir}")
    return files


def _source_time_count(source_files):
    total = 0
    for source_file in source_files:
        with nc.Dataset(str(source_file)) as ds:
            if "time" not in ds.dimensions:
                raise RuntimeError(f"missing time dimension in {source_file}")
            total += len(ds.dimensions["time"])
    if total == 0:
        raise RuntimeError("source files contain zero time records")
    return total


def _time_stamp(source_file, time_index):
    with nc.Dataset(str(source_file)) as ds:
        time_var = ds.variables["time"]
        time_var.set_auto_maskandscale(False)
        units = time_var.getncattr("units")
        calendar_name = time_var.getncattr("calendar") if "calendar" in time_var.ncattrs() else "standard"
        dt = nc.num2date(
            time_var[time_index],
            units=units,
            calendar=calendar_name,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
    return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"


def _output_candidates(stream, year, output_base):
    out_dir = Path(output_base) / stream
    return sorted(out_dir.glob(f"{stream}_era5_oper_sfc_{year}*-*.nc"))


def _output_valid(
    stream,
    year,
    output_base,
    varname,
    expected_time_count,
    source_files=None,
):
    for candidate in _output_candidates(stream, year, output_base):
        try:
            validate(
                candidate,
                varname,
                year=year,
                expected_time_count=expected_time_count,
                log_success=False,
            )
            if source_files is not None:
                validate_preservation(candidate, source_files, varname)
        except Exception as exc:
            logging.warning("Existing output is invalid: %s (%s)", candidate, exc)
        else:
            return True
    return False


# -- netCDF copying -------------------------------------------------------------


def _copy_attrs(src_obj, dst_obj, skip=()):
    for attr in src_obj.ncattrs():
        if attr not in skip:
            dst_obj.setncattr(attr, src_obj.getncattr(attr))


def _create_output_schema(out_ds, template_ds, varname):
    for dim_name, dim in template_ds.dimensions.items():
        out_ds.createDimension(dim_name, None if dim_name == "time" else len(dim))

    for name, src_var in template_ds.variables.items():
        fill_value = src_var.getncattr("_FillValue") if "_FillValue" in src_var.ncattrs() else None
        kwargs = {"fill_value": fill_value}
        if name == varname:
            kwargs.update(
                {
                    "zlib": True,
                    "shuffle": True,
                    "complevel": COMPLEVEL,
                    "chunksizes": (CHUNK_T, CHUNK_LAT, CHUNK_LON),
                }
            )

        dst_var = out_ds.createVariable(
            name,
            src_var.datatype,
            src_var.dimensions,
            **kwargs,
        )
        _copy_attrs(src_var, dst_var, skip=("_FillValue",))
        src_var.set_auto_maskandscale(False)
        dst_var.set_auto_maskandscale(False)


def _copy_variable_slice(src_var, dst_var, src_start, src_stop, dst_start):
    time_axis = src_var.dimensions.index("time")
    src_key = [slice(None)] * src_var.ndim
    dst_key = [slice(None)] * dst_var.ndim
    count = src_stop - src_start
    src_key[time_axis] = slice(src_start, src_stop)
    dst_key[time_axis] = slice(dst_start, dst_start + count)
    dst_var[tuple(dst_key)] = src_var[tuple(src_key)]


def _copy_static_variables(template_ds, out_ds):
    for name, dst_var in out_ds.variables.items():
        if "time" in dst_var.dimensions:
            continue
        src_var = template_ds.variables[name]
        src_var.set_auto_maskandscale(False)
        dst_var.set_auto_maskandscale(False)
        dst_var[:] = src_var[:]


def _write_yearly_file(source_files, tmp_path, stream, year, varname, runcmd, copy_time_block):
    with nc.Dataset(str(source_files[0])) as template, nc.Dataset(str(tmp_path), "w", format="NETCDF4") as out_ds:
        _create_output_schema(out_ds, template, varname)
        _copy_static_variables(template, out_ds)

        now_iso = _local_iso_timestamp()
        _copy_attrs(template, out_ds)
        old_title = template.getncattr("title") if "title" in template.ncattrs() else ""
        old_history = template.getncattr("history") if "history" in template.ncattrs() else ""
        this_file = os.path.normpath(__file__)
        new_history = (
            f"{now_iso} rechunked from [93,91,180] to [1,721,1440] using netCDF4; "
            f"{len(source_files)} monthly files from rt52 concatenated into one yearly file. "
            + get_provenance_metadata(this_file, runcmd)
        )
        out_ds.setncattr("title", re.sub(r"\s+\d{8}-\d{8}$", f" {year}", old_title))
        out_ds.setncattr("history", f"{new_history}\n{old_history}" if old_history else new_history)
        out_ds.setncattr("rechunked_from", f"{SOURCE_BASE}/{stream}/{year}/")
        out_ds.setncattr("rechunked_by", "Ezhilsabareesh Kannadasan (ek4684)")
        out_ds.setncattr("rechunked_date", now_iso)
        out_ds.setncattr("original_chunking", "[93, 91, 180]")
        out_ds.setncattr("target_chunking", "[1, 721, 1440]")

        out_index = 0
        for source_file in source_files:
            with nc.Dataset(str(source_file)) as src_ds:
                time_len = len(src_ds.dimensions["time"])
                logging.info(
                    "Copying %s/%s from %s (%s time records)",
                    stream,
                    year,
                    source_file.name,
                    time_len,
                )
                for name, dst_var in out_ds.variables.items():
                    if "time" not in dst_var.dimensions:
                        continue
                    src_var = src_ds.variables[name]
                    src_var.set_auto_maskandscale(False)
                    dst_var.set_auto_maskandscale(False)
                    for src_start in range(0, time_len, copy_time_block):
                        src_stop = min(src_start + copy_time_block, time_len)
                        _copy_variable_slice(src_var, dst_var, src_start, src_stop, out_index + src_start)
                out_index += time_len


def _source_for_global_index(source_files, global_index):
    remaining = global_index
    for source_file in source_files:
        with nc.Dataset(str(source_file)) as ds:
            time_len = len(ds.dimensions["time"])
        if remaining < time_len:
            return source_file, remaining
        remaining -= time_len
    raise IndexError(f"global time index out of range: {global_index}")


def validate_preservation(out_path, source_files, varname):
    """Check output metadata and sampled raw packed values against source files."""
    with nc.Dataset(str(source_files[0])) as src0, nc.Dataset(str(out_path)) as out_ds:
        if set(src0.variables) != set(out_ds.variables):
            raise RuntimeError("variable names differ between source template and output")

        out_var = out_ds.variables[varname]
        src_var = src0.variables[varname]
        out_var.set_auto_maskandscale(False)
        src_var.set_auto_maskandscale(False)

        if out_var.dimensions != src_var.dimensions:
            raise RuntimeError(f"{varname}: dimensions differ")
        if out_var.shape[1:] != src_var.shape[1:]:
            raise RuntimeError(f"{varname}: non-time shape differs")
        if str(out_var.dtype) != str(src_var.dtype):
            raise RuntimeError(f"{varname}: dtype differs")

        for attr in PRESERVED_VAR_ATTRS:
            _require_matching_attr(src_var, out_var, attr, varname)
        _require_matching_attr(src_var, out_var, "_FillValue", varname)

        for coord in ("latitude", "longitude"):
            if coord in src0.variables:
                src_coord = src0.variables[coord]
                out_coord = out_ds.variables[coord]
                src_coord.set_auto_maskandscale(False)
                out_coord.set_auto_maskandscale(False)
                if src_coord.dimensions != out_coord.dimensions:
                    raise RuntimeError(f"{coord}: dimensions differ")
                if not np.array_equal(src_coord[:], out_coord[:]):
                    raise RuntimeError(f"{coord}: coordinate values differ")
                for attr in src_coord.ncattrs():
                    _require_matching_attr(src_coord, out_coord, attr, coord)

        out_time = out_ds.variables["time"]
        out_time.set_auto_maskandscale(False)
        for attr in src0.variables["time"].ncattrs():
            _require_matching_attr(src0.variables["time"], out_time, attr, "time")

    time_parts = []
    for source_file in source_files:
        with nc.Dataset(str(source_file)) as src_ds:
            time_var = src_ds.variables["time"]
            time_var.set_auto_maskandscale(False)
            time_parts.append(np.asarray(time_var[:]))
    source_time = np.concatenate(time_parts)

    with nc.Dataset(str(out_path)) as out_ds:
        out_time = out_ds.variables["time"]
        out_time.set_auto_maskandscale(False)
        if not np.array_equal(source_time, np.asarray(out_time[:])):
            raise RuntimeError("time values differ between source files and output")

        total_time = len(out_ds.dimensions["time"])
        sample_indices = sorted({0, total_time // 2, total_time - 1})
        out_var = out_ds.variables[varname]
        out_var.set_auto_maskandscale(False)
        for global_index in sample_indices:
            source_file, local_index = _source_for_global_index(source_files, global_index)
            with nc.Dataset(str(source_file)) as src_ds:
                src_var = src_ds.variables[varname]
                src_var.set_auto_maskandscale(False)
                src_values = np.asarray(src_var[local_index : local_index + 1, :, :])
            out_values = np.asarray(out_var[global_index : global_index + 1, :, :])
            if not np.array_equal(src_values, out_values):
                raise RuntimeError(
                    f"raw packed values differ at sampled time index {global_index}"
                )

    logging.info("preservation: %s raw/metadata checks - OK", Path(out_path).name)


# -- single-task processing -----------------------------------------------------


def process_one(
    stream,
    year,
    source_base,
    output_base,
    overwrite,
    copy_time_block,
    runcmd,
):
    """
    Rechunk and concatenate all monthly rt52 files for one (stream, year) into
    a single yearly output file. Returns True on success, False on recoverable
    setup failure. Unexpected processing failures are raised for useful tracebacks.
    """
    varname = STREAM_TO_VARNAME[stream]

    try:
        source_files = _source_files(stream, year, source_base)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return False

    source_time_count = _source_time_count(source_files)
    full_year_count = _full_year_hours(year)
    source_is_partial = source_time_count != full_year_count
    if source_is_partial:
        logging.warning(
            "Source %s/%s has %s hourly records, not full-year %s; treating as partial source year",
            stream,
            year,
            source_time_count,
            full_year_count,
        )

    start = _time_stamp(source_files[0], 0)
    end = _time_stamp(source_files[-1], -1)

    out_dir = Path(output_base) / stream
    out_path = out_dir / f"{stream}_era5_oper_sfc_{start}-{end}.nc"

    if out_path.exists() and not overwrite:
        try:
            validate(
                out_path,
                varname,
                year=year,
                expected_time_count=source_time_count,
                allow_partial=source_is_partial,
            )
            validate_preservation(out_path, source_files, varname)
        except Exception as exc:
            logging.warning(
                "Existing output is invalid and will be regenerated: %s (%s)",
                out_path,
                exc,
            )
        else:
            logging.info("SKIP %s/%s: output already valid - %s", stream, year, out_path)
            return True

    logging.info(
        "Processing %s/%s: %s files, %s time records -> %s",
        stream,
        year,
        len(source_files),
        source_time_count,
        out_path.name,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        logging.warning("Removing stale temporary output before rewrite: %s", tmp_path)
        tmp_path.unlink()

    t0 = time.time()
    try:
        _write_yearly_file(source_files, tmp_path, stream, year, varname, runcmd, copy_time_block)
        validate(
            tmp_path,
            varname,
            year=year,
            expected_time_count=source_time_count,
            allow_partial=source_is_partial,
        )
        validate_preservation(tmp_path, source_files, varname)
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    elapsed = time.time() - t0

    size_gb = out_path.stat().st_size / 1e9
    logging.info(
        "%s/%s: DONE - %s (%.2f GB, %.0fs)",
        stream,
        year,
        out_path,
        size_gb,
        elapsed,
    )
    return True


# -- task orchestration ---------------------------------------------------------


def _parse_years(year_args):
    if not year_args:
        return list(range(YEAR_FIRST, YEAR_LAST + 1))

    years = []
    for item in year_args:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            sep = "-" if "-" in part else ":" if ":" in part else None
            if sep:
                start_s, end_s = part.split(sep, 1)
                start, end = int(start_s), int(end_s)
                step = 1 if start <= end else -1
                years.extend(range(start, end + step, step))
            else:
                years.append(int(part))

    years = sorted(set(years))
    invalid = [year for year in years if year < YEAR_FIRST or year > YEAR_LAST]
    if invalid:
        raise ValueError(
            f"years outside supported range {YEAR_FIRST}-{YEAR_LAST}: {invalid}"
        )
    return years


def _build_tasks(streams, years, source_base, output_base, overwrite, copy_time_block):
    tasks = []
    skipped = 0
    for stream in streams:
        varname = STREAM_TO_VARNAME[stream]
        for year in years:
            if not overwrite:
                candidates = _output_candidates(stream, year, output_base)
                if candidates:
                    try:
                        source_files = _source_files(stream, year, source_base)
                        expected_time_count = _source_time_count(source_files)
                    except Exception as exc:
                        logging.warning(
                            "Could not validate existing output for %s/%s; task will run and report the setup error: %s",
                            stream,
                            year,
                            exc,
                        )
                    else:
                        if _output_valid(
                            stream,
                            year,
                            output_base,
                            varname,
                            expected_time_count,
                            source_files=source_files,
                        ):
                            skipped += 1
                            continue

            tasks.append(
                (stream, year, source_base, output_base, overwrite, copy_time_block)
            )
    return tasks, skipped


def _task_log_path(output_base, stream, year):
    log_dir = Path(output_base) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{stream}_{year}.log"


def _tail_text(text, limit=4000):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "... <truncated> ...\n" + text[-limit:]


def _subprocess_task(args_tuple):
    """Invoke this script in internal task mode for one (stream, year) task."""
    stream, year, source_base, output_base, overwrite, copy_time_block = args_tuple
    script = os.path.abspath(__file__)
    cmd = [
        sys.executable,
        script,
        "--run-task",
        "--stream",
        stream,
        "--year",
        str(year),
        "--source-dir",
        source_base,
        "--output-dir",
        output_base,
        "--copy-time-block",
        str(copy_time_block),
    ]
    if overwrite:
        cmd.append("--overwrite")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    log_path = _task_log_path(output_base, stream, year)
    log_path.write_text(
        "COMMAND:\n"
        f"{shlex.join(cmd)}\n\n"
        f"RETURN CODE: {result.returncode}\n"
        f"ELAPSED: {elapsed:.0f}s\n\n"
        "STDOUT:\n"
        f"{result.stdout or ''}\n"
        "STDERR:\n"
        f"{result.stderr or ''}\n"
    )

    status = "OK" if result.returncode == 0 else "FAIL"
    if result.returncode != 0:
        print(
            f"[{status}] {stream}/{year} ({elapsed:.0f}s, rc={result.returncode}) log={log_path}\n"
            f"  command: {shlex.join(cmd)}\n"
            f"  stdout tail:\n{_tail_text(result.stdout)}\n"
            f"  stderr tail:\n{_tail_text(result.stderr)}",
            flush=True,
        )
    else:
        last_line = (result.stdout.strip().split("\n") or [""])[-1]
        print(
            f"[{status}] {stream}/{year} ({elapsed:.0f}s) {last_line} log={log_path}",
            flush=True,
        )

    return stream, year, result.returncode, elapsed, str(log_path)


def main():
    parser = argparse.ArgumentParser(
        description="Rechunk ERA5 rt52 monthly files into yearly rechunked files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stream",
        choices=ALL_STREAMS,
        nargs="+",
        metavar="STREAM",
        help="One or more streams to process (default: all non-sea-ice streams).",
    )
    parser.add_argument(
        "--year",
        nargs="+",
        metavar="YEAR_OR_RANGE",
        help="One or more years/ranges, e.g. 1979 1980-1982 (default: 1940-2026).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker tasks (default: 1).",
    )
    parser.add_argument(
        "--copy-time-block",
        "--read-chunk-hours",
        dest="copy_time_block",
        type=int,
        default=COPY_TIME_BLOCK,
        help="Number of time records copied per netCDF4 block (default: 93).",
    )
    parser.add_argument("--source-dir", default=SOURCE_BASE)
    parser.add_argument("--output-dir", default=OUTPUT_BASE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-task", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.copy_time_block < 1:
        parser.error("--copy-time-block must be at least 1")

    streams = args.stream or ALL_STREAMS
    try:
        years = _parse_years(args.year)
    except ValueError as exc:
        parser.error(str(exc))

    if args.run_task:
        if len(streams) != 1 or len(years) != 1:
            parser.error("internal --run-task mode requires exactly one stream and one year")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        try:
            ok = process_one(
                streams[0],
                years[0],
                args.source_dir,
                args.output_dir,
                args.overwrite,
                args.copy_time_block,
                shlex.join(sys.argv),
            )
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0 if ok else 1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tasks, skipped = _build_tasks(
        streams,
        years,
        args.source_dir,
        args.output_dir,
        args.overwrite,
        args.copy_time_block,
    )

    print(f"Streams:      {', '.join(streams)}", flush=True)
    print(f"Years:        {years[0]}-{years[-1]} ({len(years)} total)" if len(years) > 1 else f"Years:        {years[0]}", flush=True)
    print(f"Tasks to run: {len(tasks)}", flush=True)
    if not args.overwrite:
        print(f"Tasks skipped: {skipped}  (valid output already exists)", flush=True)
    print(f"Workers:      {args.workers}", flush=True)
    print(f"Copy block:   time={args.copy_time_block}", flush=True)

    if not tasks:
        print("Nothing to do.", flush=True)
        return

    t_start = time.time()
    failures = []

    if args.workers == 1:
        # Avoid keeping PBS jobs alive after the single child task has completed.
        # This happened on Gadi with Pool(1): the task finished, but the parent
        # process stayed alive until the PBS walltime expired.
        for task in tasks:
            stream, year, rc, _, log_path = _subprocess_task(task)
            if rc != 0:
                failures.append((stream, year, log_path))
    else:
        with multiprocessing.Pool(args.workers) as pool:
            for stream, year, rc, _, log_path in pool.imap_unordered(
                _subprocess_task, tasks
            ):
                if rc != 0:
                    failures.append((stream, year, log_path))

    total_elapsed = time.time() - t_start
    print(
        f"\nCompleted {len(tasks)} tasks in {total_elapsed:.0f}s ({total_elapsed / 3600:.1f}h",
        flush=True,
    )
    if failures:
        print(f"FAILED ({len(failures)}):", flush=True)
        for stream, year, log_path in failures:
            print(f"  {stream}/{year} log={log_path}", flush=True)
        sys.exit(1)

    print("All tasks succeeded.", flush=True)


if __name__ == "__main__":
    main()
