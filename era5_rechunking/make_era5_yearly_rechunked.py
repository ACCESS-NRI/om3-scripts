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
# Monthly values are decoded with their source packing and repacked to one
# year-wide int16 scale_factor and add_offset.
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
import zlib
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

# -- validation ----------------------------------------------------------------


def _full_year_hours(year):
    return 8784 if calendar.isleap(year) else 8760


def _local_iso_timestamp():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _attrs_equal(left, right):
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape or right_arr.shape:
        return left_arr.shape == right_arr.shape and bool(
            np.array_equal(left_arr, right_arr)
        )
    return left == right


def _require_matching_attr(src_obj, dst_obj, attr, label):
    src_has = attr in src_obj.ncattrs()
    dst_has = attr in dst_obj.ncattrs()
    if src_has != dst_has:
        raise RuntimeError(f"{label}: attribute {attr!r} presence differs")
    if src_has and not _attrs_equal(src_obj.getncattr(attr), dst_obj.getncattr(attr)):
        raise RuntimeError(f"{label}: attribute {attr!r} differs")


def _require_matching_attrs(src_obj, dst_obj, label, exclude=()):
    excluded = set(exclude)
    src_attrs = set(src_obj.ncattrs()) - excluded
    dst_attrs = set(dst_obj.ncattrs()) - excluded
    if src_attrs != dst_attrs:
        raise RuntimeError(
            f"{label}: attribute names differ: {sorted(src_attrs)} != {sorted(dst_attrs)}"
        )
    for attr in src_attrs:
        _require_matching_attr(src_obj, dst_obj, attr, label)


def _scalar_attr(var, attr):
    if attr not in var.ncattrs():
        raise RuntimeError(f"{var.name}: missing attribute {attr!r}")
    value = np.asarray(var.getncattr(attr))
    if value.size != 1:
        raise RuntimeError(f"{var.name}: attribute {attr!r} must be scalar")
    result = float(value.reshape(-1)[0])
    if not np.isfinite(result):
        raise RuntimeError(f"{var.name}: attribute {attr!r} is not finite")
    return result


def _packing(var):
    scale_factor = _scalar_attr(var, "scale_factor")
    add_offset = _scalar_attr(var, "add_offset")
    if scale_factor == 0:
        raise RuntimeError(f"{var.name}: scale_factor must be non-zero")
    return scale_factor, add_offset


def _missing_codes(var):
    codes = set()
    for attr in ("_FillValue", "missing_value"):
        if attr not in var.ncattrs():
            continue
        for value in np.asarray(var.getncattr(attr)).reshape(-1):
            numeric = int(value)
            if numeric != value:
                raise RuntimeError(f"{var.name}: {attr}={value!r} is not an integer")
            codes.add(numeric)
    return codes


def _missing_mask(raw_values, var):
    mask = np.zeros(raw_values.shape, dtype=bool)
    for code in _missing_codes(var):
        mask |= raw_values == code
    return mask


def _decode_raw(raw_values, var):
    scale_factor, add_offset = _packing(var)
    decoded = raw_values.astype(np.float64)
    decoded *= scale_factor
    decoded += add_offset
    return decoded


def _valid_code_extrema(var):
    limits = np.iinfo(var.dtype)
    reserved = _missing_codes(var)
    code_min = int(limits.min)
    while code_min in reserved and code_min <= limits.max:
        code_min += 1
    code_max = int(limits.max)
    while code_max in reserved and code_max >= limits.min:
        code_max -= 1
    if code_max <= code_min:
        raise RuntimeError(f"{var.name}: insufficient integer codes for packing")
    return code_min, code_max


def _largest_valid_code_range(var):
    if not np.issubdtype(var.dtype, np.signedinteger):
        raise RuntimeError(
            f"{var.name}: expected a signed integer dtype, got {var.dtype}"
        )

    limits = np.iinfo(var.dtype)
    reserved = sorted(
        code for code in _missing_codes(var) if limits.min <= code <= limits.max
    )
    ranges = []
    start = int(limits.min)
    for code in reserved:
        if start <= code - 1:
            ranges.append((start, code - 1))
        start = code + 1
    if start <= limits.max:
        ranges.append((start, int(limits.max)))
    if not ranges:
        raise RuntimeError(
            f"{var.name}: no integer codes remain after reserving missing values"
        )

    code_min, code_max = max(ranges, key=lambda bounds: bounds[1] - bounds[0])
    if code_max <= code_min:
        raise RuntimeError(f"{var.name}: insufficient integer codes for packing")
    return code_min, code_max


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

        _packing(var)
        filters = var.filters()
        if not filters.get("zlib"):
            raise RuntimeError("main variable is not compressed with zlib")
        if not filters.get("shuffle"):
            raise RuntimeError("main variable does not use the shuffle filter")
        if filters.get("complevel") != COMPLEVEL:
            raise RuntimeError(
                f"compression level={filters.get('complevel')}, expected {COMPLEVEL}"
            )

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


def _validate_source_schema(source_files, varname):
    with nc.Dataset(str(source_files[0])) as template:
        if varname not in template.variables:
            raise RuntimeError(f"missing variable {varname!r} in {source_files[0]}")
        template_var = template.variables[varname]
        template_var.set_auto_maskandscale(False)
        if str(template_var.dtype) != "int16":
            raise RuntimeError(
                f"{varname}: source dtype={template_var.dtype}, expected int16"
            )
        if template_var.dimensions != ("time", "latitude", "longitude"):
            raise RuntimeError(
                f"{varname}: unexpected dimensions {template_var.dimensions!r}"
            )
        if "_FillValue" not in template_var.ncattrs():
            raise RuntimeError(f"{varname}: source has no _FillValue")
        _packing(template_var)

        for source_file in source_files:
            with nc.Dataset(str(source_file)) as source:
                if set(source.variables) != set(template.variables):
                    raise RuntimeError(f"variable names differ in {source_file}")
                source_var = source.variables[varname]
                source_var.set_auto_maskandscale(False)
                if source_var.dimensions != template_var.dimensions:
                    raise RuntimeError(f"{varname}: dimensions differ in {source_file}")
                if source_var.shape[1:] != template_var.shape[1:]:
                    raise RuntimeError(
                        f"{varname}: spatial shape differs in {source_file}"
                    )
                if str(source_var.dtype) != str(template_var.dtype):
                    raise RuntimeError(f"{varname}: dtype differs in {source_file}")
                _packing(source_var)

                _require_matching_attrs(
                    template_var,
                    source_var,
                    varname,
                    exclude=("scale_factor", "add_offset"),
                )

                for coord in ("latitude", "longitude"):
                    template_coord = template.variables[coord]
                    source_coord = source.variables[coord]
                    template_coord.set_auto_maskandscale(False)
                    source_coord.set_auto_maskandscale(False)
                    if template_coord.dimensions != source_coord.dimensions:
                        raise RuntimeError(
                            f"{coord}: dimensions differ in {source_file}"
                        )
                    if not np.array_equal(template_coord[:], source_coord[:]):
                        raise RuntimeError(f"{coord}: values differ in {source_file}")
                    if set(template_coord.ncattrs()) != set(source_coord.ncattrs()):
                        raise RuntimeError(
                            f"{coord}: attributes differ in {source_file}"
                        )
                    for attr in template_coord.ncattrs():
                        _require_matching_attr(
                            template_coord, source_coord, attr, coord
                        )

                template_time = template.variables["time"]
                source_time = source.variables["time"]
                if set(template_time.ncattrs()) != set(source_time.ncattrs()):
                    raise RuntimeError(f"time: attributes differ in {source_file}")
                for attr in template_time.ncattrs():
                    _require_matching_attr(template_time, source_time, attr, "time")


def _calculate_yearly_packing(source_files, varname):
    # Monthly files use independent packing. Their metadata defines a safe
    # physical range without requiring an extra full-data read.
    with nc.Dataset(str(source_files[0])) as template:
        template_var = template.variables[varname]
        template_var.set_auto_maskandscale(False)
        code_min, code_max = _largest_valid_code_range(template_var)
        fill_value = int(template_var.getncattr("_FillValue"))

    physical_min = None
    physical_max = None
    for source_file in source_files:
        with nc.Dataset(str(source_file)) as source:
            source_var = source.variables[varname]
            source_var.set_auto_maskandscale(False)
            scale_factor, add_offset = _packing(source_var)
            source_code_min, source_code_max = _valid_code_extrema(source_var)
            decoded_bounds = (
                source_code_min * scale_factor + add_offset,
                source_code_max * scale_factor + add_offset,
            )
            month_min = min(decoded_bounds)
            month_max = max(decoded_bounds)
            physical_min = (
                month_min if physical_min is None else min(physical_min, month_min)
            )
            physical_max = (
                month_max if physical_max is None else max(physical_max, month_max)
            )

    if physical_min is None or physical_max is None:
        raise RuntimeError(f"{varname}: source contains no valid packing range")
    if physical_min == physical_max:
        scale_factor = 1.0
        add_offset = physical_min
    else:
        scale_factor = (physical_max - physical_min) / (code_max - code_min)
        add_offset = physical_min - code_min * scale_factor

    return {
        "scale_factor": scale_factor,
        "add_offset": add_offset,
        "code_min": code_min,
        "code_max": code_max,
        "fill_value": fill_value,
        "physical_min": physical_min,
        "physical_max": physical_max,
    }


def _time_stamp(source_file, time_index):
    with nc.Dataset(str(source_file)) as ds:
        time_var = ds.variables["time"]
        time_var.set_auto_maskandscale(False)
        units = time_var.getncattr("units")
        calendar_name = (
            time_var.getncattr("calendar")
            if "calendar" in time_var.ncattrs()
            else "standard"
        )
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


def _create_output_schema(out_ds, template_ds, varname, yearly_packing):
    for dim_name, dim in template_ds.dimensions.items():
        out_ds.createDimension(dim_name, None if dim_name == "time" else len(dim))

    for name, src_var in template_ds.variables.items():
        kwargs = {}
        if "_FillValue" in src_var.ncattrs():
            kwargs["fill_value"] = src_var.getncattr("_FillValue")
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
        skip = (
            ("_FillValue", "scale_factor", "add_offset")
            if name == varname
            else ("_FillValue",)
        )
        _copy_attrs(src_var, dst_var, skip=skip)
        if name == varname:
            dst_var.setncattr("scale_factor", yearly_packing["scale_factor"])
            dst_var.setncattr("add_offset", yearly_packing["add_offset"])
        src_var.set_auto_maskandscale(False)
        dst_var.set_auto_maskandscale(False)


def _variable_slice(var, start, stop):
    time_axis = var.dimensions.index("time")
    key = [slice(None)] * var.ndim
    key[time_axis] = slice(start, stop)
    return tuple(key)


def _repack_raw(raw_values, source_var, yearly_packing):
    missing = _missing_mask(raw_values, source_var)
    source_scale, source_offset = _packing(source_var)
    packed_float = raw_values.astype(np.float64)
    packed_float *= source_scale / yearly_packing["scale_factor"]
    packed_float += (source_offset - yearly_packing["add_offset"]) / yearly_packing[
        "scale_factor"
    ]
    np.rint(packed_float, out=packed_float)

    valid = ~missing
    if np.any(valid):
        packed_min = float(np.min(packed_float, where=valid, initial=np.inf))
        packed_max = float(np.max(packed_float, where=valid, initial=-np.inf))
        if (
            packed_min < yearly_packing["code_min"]
            or packed_max > yearly_packing["code_max"]
        ):
            raise RuntimeError(
                f"{source_var.name}: values exceed year-wide packing range "
                f"[{yearly_packing['code_min']}, {yearly_packing['code_max']}]"
            )
        np.clip(
            packed_float,
            yearly_packing["code_min"],
            yearly_packing["code_max"],
            out=packed_float,
        )

    packed = packed_float.astype(source_var.dtype)
    packed[missing] = yearly_packing["fill_value"]
    return packed


def _copy_variable_slice(
    src_var, dst_var, src_start, src_stop, dst_start, yearly_packing=None
):
    src_key = _variable_slice(src_var, src_start, src_stop)
    count = src_stop - src_start
    dst_key = _variable_slice(dst_var, dst_start, dst_start + count)
    raw_values = np.asarray(src_var[src_key])
    if yearly_packing is not None:
        raw_values = _repack_raw(raw_values, src_var, yearly_packing)
    dst_var[dst_key] = raw_values


def _copy_static_variables(template_ds, out_ds):
    for name, dst_var in out_ds.variables.items():
        if "time" in dst_var.dimensions:
            continue
        src_var = template_ds.variables[name]
        src_var.set_auto_maskandscale(False)
        dst_var.set_auto_maskandscale(False)
        dst_var[:] = src_var[:]


def _write_yearly_file(
    source_files,
    tmp_path,
    stream,
    year,
    varname,
    runcmd,
    copy_time_block,
    yearly_packing,
):
    with nc.Dataset(str(source_files[0])) as template, nc.Dataset(
        str(tmp_path), "w", format="NETCDF4"
    ) as out_ds:
        _create_output_schema(out_ds, template, varname, yearly_packing)
        _copy_static_variables(template, out_ds)

        now_iso = _local_iso_timestamp()
        _copy_attrs(template, out_ds)
        old_title = template.getncattr("title") if "title" in template.ncattrs() else ""
        old_history = (
            template.getncattr("history") if "history" in template.ncattrs() else ""
        )
        this_file = os.path.normpath(__file__)
        new_history = (
            f"{now_iso} rechunked from [93,91,180] to [1,721,1440] using netCDF4; "
            f"{len(source_files)} monthly files decoded with their source packing and "
            "repacked to one year-wide int16 encoding. "
            + get_provenance_metadata(this_file, runcmd)
        )
        out_ds.setncattr("title", re.sub(r"\s+\d{8}-\d{8}$", f" {year}", old_title))
        out_ds.setncattr(
            "history", f"{new_history}\n{old_history}" if old_history else new_history
        )
        out_ds.setncattr("rechunked_from", f"{SOURCE_BASE}/{stream}/{year}/")
        out_ds.setncattr("rechunked_by", "Ezhilsabareesh Kannadasan (ek4684)")
        out_ds.setncattr("rechunked_date", now_iso)
        out_ds.setncattr("original_chunking", "[93, 91, 180]")
        out_ds.setncattr("target_chunking", "[1, 721, 1440]")
        out_ds.setncattr(
            "yearly_packing_range",
            f"[{yearly_packing['physical_min']}, {yearly_packing['physical_max']}]",
        )

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
                        _copy_variable_slice(
                            src_var,
                            dst_var,
                            src_start,
                            src_stop,
                            out_index + src_start,
                            yearly_packing if name == varname else None,
                        )
                out_index += time_len


def _decoded_tolerance(output_var, source_values, output_values, valid):
    scale_factor, _ = _packing(output_var)
    if np.any(valid):
        magnitude = max(
            1.0,
            float(np.max(np.abs(source_values), where=valid, initial=0.0)),
            float(np.max(np.abs(output_values), where=valid, initial=0.0)),
        )
    else:
        magnitude = 1.0
    return abs(scale_factor) * 0.500001 + np.finfo(np.float64).eps * magnitude * 16


def validate_preservation(out_path, source_files, varname):
    """Check metadata, coordinates, time, and decoded data against every source month."""
    _validate_source_schema(source_files, varname)

    with nc.Dataset(str(source_files[0])) as source, nc.Dataset(
        str(out_path)
    ) as output:
        if set(source.variables) != set(output.variables):
            raise RuntimeError(
                "variable names differ between source template and output"
            )

        source_var = source.variables[varname]
        output_var = output.variables[varname]
        source_var.set_auto_maskandscale(False)
        output_var.set_auto_maskandscale(False)
        _packing(output_var)

        if output_var.dimensions != source_var.dimensions:
            raise RuntimeError(f"{varname}: dimensions differ")
        if output_var.shape[1:] != source_var.shape[1:]:
            raise RuntimeError(f"{varname}: spatial shape differs")
        if str(output_var.dtype) != str(source_var.dtype):
            raise RuntimeError(f"{varname}: dtype differs")
        if set(output_var.ncattrs()) != set(source_var.ncattrs()):
            raise RuntimeError(f"{varname}: attribute names differ")

        _require_matching_attrs(
            source_var,
            output_var,
            varname,
            exclude=("scale_factor", "add_offset"),
        )

        for coord in ("latitude", "longitude"):
            source_coord = source.variables[coord]
            output_coord = output.variables[coord]
            source_coord.set_auto_maskandscale(False)
            output_coord.set_auto_maskandscale(False)
            if source_coord.dimensions != output_coord.dimensions:
                raise RuntimeError(f"{coord}: dimensions differ")
            if not np.array_equal(source_coord[:], output_coord[:]):
                raise RuntimeError(f"{coord}: coordinate values differ")
            if set(source_coord.ncattrs()) != set(output_coord.ncattrs()):
                raise RuntimeError(f"{coord}: attribute names differ")
            for attr in source_coord.ncattrs():
                _require_matching_attr(source_coord, output_coord, attr, coord)

        source_time = source.variables["time"]
        output_time = output.variables["time"]
        if set(source_time.ncattrs()) != set(output_time.ncattrs()):
            raise RuntimeError("time: attribute names differ")
        for attr in source_time.ncattrs():
            _require_matching_attr(source_time, output_time, attr, "time")

    source_time_parts = []
    for source_file in source_files:
        with nc.Dataset(str(source_file)) as source:
            time_var = source.variables["time"]
            time_var.set_auto_maskandscale(False)
            source_time_parts.append(np.asarray(time_var[:]))
    source_time = np.concatenate(source_time_parts)

    seed = zlib.crc32(f"{Path(out_path).name}:{varname}".encode())
    rng = np.random.default_rng(seed)
    max_error = 0.0
    max_tolerance = 0.0
    output_index = 0

    with nc.Dataset(str(out_path)) as output:
        output_time = output.variables["time"]
        output_time.set_auto_maskandscale(False)
        if not np.array_equal(source_time, np.asarray(output_time[:])):
            raise RuntimeError("time values differ between source files and output")

        output_var = output.variables[varname]
        output_var.set_auto_maskandscale(False)
        # Use a reproducible random timestep from every month. Comparing decoded
        # full fields catches month-specific packing errors at modest I/O cost.
        for source_file in source_files:
            with nc.Dataset(str(source_file)) as source:
                source_var = source.variables[varname]
                source_var.set_auto_maskandscale(False)
                time_len = len(source.dimensions["time"])
                local_index = int(rng.integers(0, time_len))
                source_raw = np.asarray(source_var[local_index, :, :])
                output_raw = np.asarray(output_var[output_index + local_index, :, :])
                source_missing = _missing_mask(source_raw, source_var)
                output_missing = _missing_mask(output_raw, output_var)
                if not np.array_equal(source_missing, output_missing):
                    raise RuntimeError(
                        f"missing-value mask differs for {source_file.name} "
                        f"at time index {local_index}"
                    )

                valid = ~source_missing
                source_values = _decode_raw(source_raw, source_var)
                output_values = _decode_raw(output_raw, output_var)
                tolerance = _decoded_tolerance(
                    output_var, source_values, output_values, valid
                )
                error = float(
                    np.max(
                        np.abs(source_values - output_values),
                        where=valid,
                        initial=0.0,
                    )
                )
                if error > tolerance:
                    raise RuntimeError(
                        f"decoded values differ for {source_file.name} at time index "
                        f"{local_index}: max error {error:.17g} exceeds yearly packing "
                        f"tolerance {tolerance:.17g}"
                    )
                max_error = max(max_error, error)
                max_tolerance = max(max_tolerance, tolerance)
                output_index += time_len

    logging.info(
        "preservation: %s metadata/time and %s monthly decoded fields - OK "
        "(max error %.6g, tolerance %.6g)",
        Path(out_path).name,
        len(source_files),
        max_error,
        max_tolerance,
    )


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
    _validate_source_schema(source_files, varname)
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
            logging.info(
                "SKIP %s/%s: output already valid - %s", stream, year, out_path
            )
            return True

    yearly_packing = _calculate_yearly_packing(source_files, varname)
    logging.info(
        "Processing %s/%s: %s files, %s time records -> %s; "
        "yearly scale_factor=%.17g add_offset=%.17g physical range=[%.17g, %.17g]",
        stream,
        year,
        len(source_files),
        source_time_count,
        out_path.name,
        yearly_packing["scale_factor"],
        yearly_packing["add_offset"],
        yearly_packing["physical_min"],
        yearly_packing["physical_max"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        logging.warning("Removing stale temporary output before rewrite: %s", tmp_path)
        tmp_path.unlink()

    t0 = time.time()
    try:
        _write_yearly_file(
            source_files,
            tmp_path,
            stream,
            year,
            varname,
            runcmd,
            copy_time_block,
            yearly_packing,
        )
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
            parser.error(
                "internal --run-task mode requires exactly one stream and one year"
            )
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

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    tasks, skipped = _build_tasks(
        streams,
        years,
        args.source_dir,
        args.output_dir,
        args.overwrite,
        args.copy_time_block,
    )

    print(f"Streams:      {', '.join(streams)}", flush=True)
    print(
        (
            f"Years:        {years[0]}-{years[-1]} ({len(years)} total)"
            if len(years) > 1
            else f"Years:        {years[0]}"
        ),
        flush=True,
    )
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
        f"\nCompleted {len(tasks)} tasks in {total_elapsed:.0f}s ({total_elapsed / 3600:.1f}h)",
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
