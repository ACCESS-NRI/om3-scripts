#!/usr/bin/env python3
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# Contact: Ezhilsabareesh Kannadasan <ezhilsabareesh.kannadasan@anu.edu.au>

"""
This script processes and smooths sea surface data from the initial conditions
NetCDF files generated using https://github.com/ACCESS-NRI/initial_conditions_access-om3/.
It applies a uniform smoothing filter to the surface layer (0m depth) of the specified
variable for each month and concatenates the smoothed data into a single output NetCDF file.

The input files are 'woa23_ts_<month>_mom.nc' at the resolution of the desired output.

Usage:
    python make_salt_sfc_restore_from_regridded_woa.py --input_path=<input_directory>
    --var=<var_name> --output_file=<output_file>

Example:
    python make_salt_sfc_restore_from_regridded_woa.py --input_path=/path/to/input/dir
    --var=asalt --output_file=/path/to/output/salt_sfc_restore.nc

Command-line arguments:
    - input_path: The directory containing the initial conditions NetCDF files.
    - var: The name of the variable to process.
    - output_file: The path to the output NetCDF file.
"""

import xarray as xr
import numpy as np
from scipy.ndimage import uniform_filter
import argparse
from pathlib import Path
import os
import sys

# Add the root path for the common scripts
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum


def smooth2d(src):
    tmp_src = np.ndarray((src.shape[0] + 6, src.shape[1]))

    # Window size
    ws = 3

    tmp_src[ws:-ws, :] = src[:, :]
    tmp_src[:ws, :] = src[-ws:, :]
    tmp_src[-ws:, :] = src[:3, :]

    dest = uniform_filter(tmp_src, size=ws, mode="nearest")
    return dest[ws:-ws, :]


def main(input_path, variable_to_smooth, output_file):

    file_template = f"{input_path}/woa23_ts_{{:02d}}_mom.nc"

    file_paths = [file_template.format(month) for month in range(1, 13)]

    ds = xr.open_mfdataset(
        file_paths, chunks={"lat": -1, "lon": -1}, decode_times=False
    )

    # Get the sea surface value
    da = ds[variable_to_smooth].isel(depth=0, drop=True)

    # Smooth in x & y (for each month)
    smoothed_da = xr.apply_ufunc(
        smooth2d,
        da,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        vectorize=True,
        dask="parallelized",
    )

    smoothed_da = smoothed_da.assign_attrs(
        {
            "standard_name": da.attrs["standard_name"],
            "long_name": f"{da.attrs['long_name']} at 0m",
            "units": da.attrs["units"],
        }
    )

    out_ds = smoothed_da.to_dataset()
    out_ds["climatology_bounds"] = ds["climatology_bounds"]
    out_ds["time"].attrs["calendar"] = "gregorian"
    # calendar is technically proleptic_gregorian, but FMS doesn't recognise this

    out_ds["time"] = out_ds.time.assign_attrs({"modulo": " "})

    # Obtain metadata
    this_file = sys.argv[0]
    runcmd = f"{sys.executable} {' '.join(sys.argv)}"

    out_ds = out_ds.assign_attrs(
        {
            "history": get_provenance_metadata(this_file, runcmd),
            "input_files": [f"{f}(md5sum:{md5sum(f)})" for f in file_paths],
        }
    )

    # Save
    out_ds[variable_to_smooth].encoding.setdefault(
        "_FillValue", 1e20
    )  #  Set _FillValue if not already set
    out_ds[variable_to_smooth].encoding |= {
        "chunksizes": (1, len(ds.lat), len(ds.lon)),
        "compression": "zlib",
        "complevel": 2,
    }
    out_ds.to_netcdf(
        output_file,
        unlimited_dims="time",
    )

    print(f"Concatenated and smoothed data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and concatenate NetCDF files with smoothing."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the directory containing input NetCDF files.",
    )
    parser.add_argument(
        "--var",
        type=str,
        required=True,
        help="The name of the variable to process.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output NetCDF file.",
    )
    args = parser.parse_args()

    main(args.input_path, args.var, args.output_file)
