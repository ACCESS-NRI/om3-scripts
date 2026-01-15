#!/usr/bin/env python3
# Copyright 2026 ACCESS-NRI and contributors.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Bottom roughness regridding for internal-tide generation (h^2)
#
# This script regrids a precomputed bottom-roughness field (h^2), generated on the
# regular WOA23 lat-lon grid onto a MOM6 model grid using xesmf.
#
# It is intended to be run after `generate_intermediate_bottom_roughness_intermediate_woa.py`,
# which computes the WOA-based intermediates.
#
# The regridding step is separated into this standalone script to avoid known issues
# when combining xesmf regridding with MPI-based workflows in the analysis environment.
#
# Usage:
#   python3 generate_bottom_roughness_regrid.py \
#       --topog_file /path/to/model_topog.nc \
#       --hgrid_file /path/to/hgrid.nc \
#       --woa_intermediate_file /path/to/woa_intermediates.nc \
#       --method conservative_normed \
#       --periodic \
#       --output_file /path/to/bottom_roughness.nc
#
# Contact:
#    - Minghang Li <Minghang.Li1@anu.edu.au>
#
# Dependencies:
#   - xesmf
#   - xarray
#   - numpy
#
# Modules:
#   module use /g/data/xp65/public/modules
#   module load conda/analysis3-25.05
#   module load openmpi/4.1.7
#   module load git
# =========================================================================================
import argparse
from pathlib import Path
import sys
import os
import numpy as np
import xarray as xr
import xesmf as xe

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from scripts_common import get_provenance_metadata, md5sum
from mesh_generation.generate_mesh import mom6_mask_detection


def src_1d_corners(coord: xr.DataArray, name: str) -> xr.DataArray:
    c = coord.values
    mid = 0.5 * (c[1:] + c[:-1])
    b = np.empty(c.size + 1)
    b[1:-1] = mid
    b[0] = c[0] - (mid[0] - c[0])
    b[-1] = c[-1] + (c[-1] - mid[-1])

    return xr.DataArray(b, dims=(f"{name}_b",), name=f"{name}_b")


def build_source_ds(
    depth_var: xr.DataArray, lon: xr.DataArray, lat: xr.DataArray
) -> xr.Dataset:
    """
    Build xesmf source dataset from depth_var with lon/lat coords.
    """
    lon_b_1d = src_1d_corners(lon, "lon")
    lat_b_1d = src_1d_corners(lat, "lat")
    lat_b2d, lon_b2d = xr.broadcast(lat_b_1d, lon_b_1d)

    source_ds = xr.Dataset(
        data_vars={"mask": (("lat", "lon"), depth_var.values)},
        coords={
            "lon": lon,
            "lat": lat,
            "lon_b": (("lat_b", "lon_b"), lon_b2d.values),
            "lat_b": (("lat_b", "lon_b"), lat_b2d.values),
        },
    )
    return source_ds


def build_target_ds(
    mask: xr.DataArray,
    hgrid_x: xr.DataArray,
    hgrid_y: xr.DataArray,
    hgrid_xc: xr.DataArray,
    hgrid_yc: xr.DataArray,
) -> xr.Dataset:
    """
    Build xesmf target dataset from MOM6 hgrid coords and mask.
    """
    target_ds = xr.Dataset(
        data_vars={"mask": (("y", "x"), mask)},
        coords={
            "lon": (("y", "x"), hgrid_x.values),
            "lat": (("y", "x"), hgrid_y.values),
            "lon_b": (("y_b", "x_b"), hgrid_xc.values),
            "lat_b": (("y_b", "x_b"), hgrid_yc.values),
        },
    )
    return target_ds


def regrid_depth_var_to_mom6(
    depth_var: xr.Dataset,
    topog_file: str,
    hgrid_file: str,
    method: str = "conservative_normed",
    periodic: bool = True,
) -> xr.Dataset:
    """
    Regrid depth_var (on regular WOA lon/lat grid) onto MOM6 grid using xESMF.
    """

    source_lon = depth_var["lon"]
    source_lat = depth_var["lat"]

    # Load model topog file then generate ocean mask ranging from [-280, 80]
    topog = xr.open_dataset(topog_file)
    mask = mom6_mask_detection(topog)

    # model hgrid
    hgrid = xr.open_dataset(hgrid_file)

    # match your slicing exactly
    hgrid_x = hgrid.x[1::2, 1::2]
    hgrid_y = hgrid.y[1::2, 1::2]
    hgrid_xc = hgrid.x[::2, ::2]
    hgrid_yc = hgrid.y[::2, ::2]

    source_ds = build_source_ds(depth_var, lon=source_lon, lat=source_lat)
    target_ds = build_target_ds(mask, hgrid_x, hgrid_y, hgrid_xc, hgrid_yc)

    regridder_kwargs = dict(
        method=method,
        periodic=periodic,
    )

    regridder = xe.Regridder(source_ds, target_ds, **regridder_kwargs)

    target_ds["h2"] = regridder(depth_var)
    target_ds["h2"] = target_ds["h2"].fillna(0.0)

    # tidy up attrs
    target_ds = target_ds.drop_vars(["lon_b", "lat_b", "mask"])

    tmp_attrs = {
        "lon": {
            "long_name": "Longitude",
            "units": "degrees_east",
        },
        "lat": {
            "long_name": "Latitude",
            "units": "degrees_north",
        },
        "h2": {
            "long_name": "Bottom roughness squared (h^2) for internal tide generation",
            "units": "m^2",
            "regrid_method": "conservative_normed",
        },
    }

    for var, attrs in tmp_attrs.items():
        target_ds[var].attrs.update(attrs)

    return target_ds


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean depth based on lambda1 computed from WOA23."
    )
    parser.add_argument(
        "--topog_file",
        type=str,
        required=True,
        help="Path to the model bathymetry file.",
    )
    parser.add_argument(
        "--hgrid_file",
        type=str,
        required=True,
        help="Path to the model hgrid file.",
    )
    parser.add_argument(
        "--woa_intermediate_file",
        type=str,
        default=None,
        help="Intermediate output file including lambda1, mean_depth, depth_var on WOA grid.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="conservative_normed",
        help="Regridding method to use.",
    )
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Whether to use periodic regridding.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="bottom_roughness.nc",
        help="Path to the bottom roughness output file.",
    )
    args = parser.parse_args()

    ds_woa_intermediate = xr.open_dataset(args.woa_intermediate_file)

    # Regridding to model grid
    print("Regridding depth variance to MOM6 grid...")
    regrid_depth_var = regrid_depth_var_to_mom6(
        depth_var=ds_woa_intermediate["depth_var"],
        topog_file=args.topog_file,
        hgrid_file=args.hgrid_file,
        method=args.method,
        periodic=args.periodic,
    )
    print("Regridding done!")

    # Add provenance metadata and MD5 hashes for input files.
    this_file = os.path.normpath(__file__)
    runcmd = (
        f"python3 {os.path.basename(this_file)} "
        f"--topog-file={args.topog_file} "
        f"--hgrid-file={args.hgrid_file} "
        f"--woa_intermediate_file={args.woa_intermediate_file} "
        f"--output_file={args.output_file} "
        f"--method={args.method} "
        f"--periodic={args.periodic}"
    )

    history = get_provenance_metadata(this_file, runcmd)
    global_attrs = {"history": history}
    file_hashes = [
        f"{args.hgrid_file} (md5 hash: {md5sum(args.hgrid_file)})",
        f"{args.topog_file} (md5 hash: {md5sum(args.topog_file)})",
        f"{args.woa_intermediate_file} (md5 hash: {md5sum(args.woa_intermediate_file)})",
    ]
    global_attrs["inputFile"] = ", ".join(file_hashes)
    regrid_depth_var.attrs.update(global_attrs)

    regrid_depth_var.to_netcdf(args.output_file)
    print(f"Output written to {args.output_file}")


if __name__ == "__main__":
    main()
