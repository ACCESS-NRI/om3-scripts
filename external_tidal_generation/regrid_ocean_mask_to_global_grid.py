#!/usr/bin/env python3
# Copyright 2025 ACCESS-NRI and contributors.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Regrid ocean mask onto a regular grid.
#
# This script regrids an ocean mask from the model grid to a new, regular grid
# using the xESMF regridding library. It loads an ocean mosaic grid (ocean_hgrid.nc),
# and its associated ocean mask (ocean_mask.nc), then creates a target grid
# at a user-specified resolution (--dx (deg) and --dy (deg)). The mask is subsequently
# interpolated onto this new grid using a regridding method (e.g., conservative_normed).
# The final regridded mask is saved as a netcdf file.
#
# Usage:
# python3 regrid_mask.py \
#   --hgrid /path/to/ocean_hgrid.nc \
#   --mask /path/to/ocean_mask.nc \
#   --dx 0.25 \
#   --dy 0.25 \
#   --method conservative_normed \
#   --output regridded_output.nc
#
# Contact:
#    - Minghang Li <Minghang.Li1@anu.edu.au>
#
# Dependencies:
#   - xarray
#   - numpy
#   - xesmf
# =========================================================================================
import os
import sys
from pathlib import Path
import argparse
import xarray as xr
import xesmf as xe

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum


def load_dataset(path: str) -> xr.Dataset:
    """
    Load an input dataset from a netcdf file.
    """
    ds = xr.open_dataset(path)
    return ds


def create_mask_ds(
    ocean_mask: xr.Dataset,
    hgrid_x: xr.DataArray,
    hgrid_y: xr.DataArray,
    hgrid_xc: xr.DataArray,
    hgrid_yc: xr.DataArray,
) -> xr.Dataset:
    """
    Loads the ocean mask and converts it into a Dataset formatted
    for use in xesmf regridding.
    `data` is a dummy variable that contains the mask values.
    """
    ocean_mask_ds = xr.Dataset(
        data_vars={
            "data": (("y", "x"), ocean_mask.mask.values),
            "mask": (("y", "x"), ocean_mask.mask.values),
        },
        coords={
            "lon": (("y", "x"), hgrid_x.values),
            "lat": (("y", "x"), hgrid_y.values),
            "lon_b": (("y_b", "x_b"), hgrid_xc.values),
            "lat_b": (("y_b", "x_b"), hgrid_yc.values),
        },
    )

    return ocean_mask_ds


def main():
    parser = argparse.ArgumentParser(
        description="Regrid ocean mask onto a regular target grid."
    )
    parser.add_argument(
        "--hgrid", type=str, required=True, help="Path to ocean_hgrid.nc"
    )
    parser.add_argument("--mask", type=str, required=True, help="Path to ocean_mask.nc")
    parser.add_argument(
        "--dx",
        type=float,
        default=0.25,
        help="Target grid resolution in longitude (degrees)",
    )
    parser.add_argument(
        "--dy",
        type=float,
        default=0.25,
        help="Target grid resolution in latitude (degrees)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="conservative_normed",
        help="Regridding method (e.g., bilinear, conservative, conservative_normed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="regridded_mask.nc",
        help="Output netcdf file name",
    )

    args = parser.parse_args()

    # Load model grid
    hgrid = load_dataset(args.hgrid)
    hgrid_x = hgrid.x[1::2, 1::2]
    hgrid_y = hgrid.y[1::2, 1::2]
    hgrid_xc = hgrid.x[::2, ::2]
    hgrid_yc = hgrid.y[::2, ::2]

    # Load ocean mask
    ocean_mask = load_dataset(args.mask)

    ocean_mask_ds = create_mask_ds(ocean_mask, hgrid_x, hgrid_y, hgrid_xc, hgrid_yc)

    # Create a regular global grid for regridding
    ds_mask_out = xe.util.grid_global(args.dx, args.dy)

    # Create regridder and regrid
    regridder = xe.Regridder(
        ocean_mask_ds, ds_mask_out, method=args.method, periodic=True
    )
    ds_mask_out[args.method] = regridder(ocean_mask_ds["data"])

    ds_mask_out.attrs["description"] = (
        f"Ocean mask regridded onto a target grid ({args.dy}, {args.dx}) using xesmf"
    )

    this_file = os.path.normpath(__file__)
    runcmd = (
        f"mpirun -n $PBS_NCPUS"
        f"python3 {os.path.basename(this_file)} "
        f"--hgrid={args.hgrid}"
        f"--mask={args.mask}"
        f"--dx={args.dx}"
        f"--dy={args.dy}"
        f"--method={args.method}"
        f"--output={args.output}"
    )

    history = get_provenance_metadata(this_file, runcmd)

    global_attrs = {"history": history}

    # add md5 hashes for input files
    file_hashes = [
        f"{args.hgrid} (md5 hash: {md5sum(args.hgrid)})",
        f"{args.mask} (md5 hash: {md5sum(args.mask)})",
    ]
    global_attrs["inputFile"] = ", ".join(file_hashes)

    ds_mask_out.attrs.update(global_attrs)
    ds_mask_out.to_netcdf(args.output)


if __name__ == "__main__":
    main()
