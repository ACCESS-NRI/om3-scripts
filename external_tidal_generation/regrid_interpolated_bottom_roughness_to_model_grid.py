#!/usr/bin/env python3
# Copyright 2025 ACCESS-NRI and contributors.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Regrid interpolated bottom roughness onto the ocean model grid
#
# This script regrids an interpolated bottom roughness field onto the ocean
# model grid using the xesmf library. It loads the ocean mosaic grid (ocean_hgrid.nc),
# ocean mask (ocean_mask.nc), and the interpolated bottom roughness field (interpolated_h2.nc).
#
# Usage:
#    python3 regrid_bottom_roughness_to_model_grid.py \
#         --hgrid /path/to/ocean_hgrid.nc \
#         --mask /path/to/ocean_mask.nc \
#         --interpolated-file /path/to/interpolated_h2.nc \
#         --method conservative_normed \
#         --output h2.nc
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


def main():
    parser = argparse.ArgumentParser(
        description="Regrid bottom roughness onto a target ocean model grid."
    )
    parser.add_argument(
        "--hgrid", type=str, required=True, help="Path to ocean mosaic ocean_hgrid.nc."
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to ocean_mask.nc.",
    )
    parser.add_argument(
        "--interpolated-file",
        type=str,
        required=True,
        help="Path to an interpolated field, such as interpolated_h2.nc.",
    )
    parser.add_argument(
        "--interpolated-field-name",
        type=str,
        default="hrms_interpolated",
        help="The field name for the interpolated roughness (default: hrms_interpolated).",
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
        default="regridded_output.nc",
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

    # Load interpolated bottom roughness
    interpolated_h2 = load_dataset(args.interpolated_file)

    # Build the target field on the model grid for regridding with mask information
    target_ds = xr.Dataset(
        data_vars={"mask": (("y", "x"), ocean_mask.mask.values)},
        coords={
            "lon": (("y", "x"), hgrid_x.values),
            "lat": (("y", "x"), hgrid_y.values),
            "lon_b": (("y_b", "x_b"), hgrid_xc.values),
            "lat_b": (("y_b", "x_b"), hgrid_yc.values),
        },
    )

    # Create regridder and regrid
    regridder = xe.Regridder(
        interpolated_h2, target_ds, method=args.method, periodic=True
    )

    # Apply the regridder to the interped bottom roughness
    target_ds["h"] = regridder(interpolated_h2[args.interpolated_field_name])

    h2 = xr.Dataset({"h2": target_ds["h"] ** 2})

    h2.attrs["description"] = (
        "Squared bottom roughness amplitude computed using a polynomial fitting method "
        "following Jayne & Laurent (2001)."
    )

    this_file = os.path.normpath(__file__)
    runcmd = (
        f"mpirun -n $PBS_NCPUS"
        f"python3 {os.path.basename(this_file)} "
        f"--hgrid={args.hgrid}"
        f"--mask={args.mask}"
        f"--interpolated-file={args.interpolated_file}"
        f"--method={args.method}"
        f"--output={args.output}"
    )

    history = get_provenance_metadata(this_file, runcmd)

    global_attrs = {"history": history}

    # add md5 hashes for input files
    file_hashes = [
        f"{args.hgrid} (md5 hash: {md5sum(args.hgrid)})",
        f"{args.mask} (md5 hash: {md5sum(args.mask)})",
        f"{args.interpolated_file} (md5 hash: {md5sum(args.interpolated_file)})",
    ]
    global_attrs["inputFile"] = ", ".join(file_hashes)

    h2.attrs.update(global_attrs)
    h2.to_netcdf(args.output)


if __name__ == "__main__":
    main()
