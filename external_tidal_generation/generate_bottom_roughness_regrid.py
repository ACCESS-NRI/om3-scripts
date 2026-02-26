#!/usr/bin/env python3
# Copyright 2026 ACCESS-NRI and contributors.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Bottom roughness regridding for internal-tide generation (h^2)
#
# This script regrids a precomputed bottom-roughness field (h^2), generated on the
# regular WOA lat-lon grid onto a MOM6 model grid using xesmf.
#
# It is intended to be run after `generate_bottom_roughness_intermediate_woa.py`,
# which computes the WOA-based intermediates.
#
# The regridding step is separated into this standalone script to avoid known issues
# when combining xesmf regridding with MPI-based workflows in the analysis environment.
# https://github.com/ACCESS-NRI/ACCESS-Analysis-Conda/issues/207
#
# Usage:
#   python3 generate_bottom_roughness_regrid.py \
#       --topog_file /path/to/model_topog.nc \
#       --hgrid_file /path/to/hgrid.nc \
#       --woa_intermediate_file /path/to/woa_intermediates.nc \
#       --method conservative_normed \
#       --periodic_regrid \
#       --periodic_lon_laplace \
#       --output_file /path/to/bottom_roughness.nc
#
# Contact:
#    - Minghang Li <Minghang.Li1@anu.edu.au>
#
# Dependencies:
#   - xesmf
#   - xarray
#   - numpy
#   - scipy
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

import scipy.sparse as sp
import scipy.sparse.linalg as spla

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from scripts_common import get_provenance_metadata, md5sum
from mesh_generation.generate_mesh import mom6_mask_detection


def fill_missing_data_laplace(
    field: np.ndarray, mask: np.ndarray, periodic_lon_laplace: bool = True
) -> np.ndarray:
    """
    Fill nans smoothly by solving a discrete Laplace problem over the wet domain.

    This is adapted from:
    https://github.com/ACCESS-NRI/om3-scripts/blob/53c807d/chlorophyll/chl_climatology_and_fill.py,

    which itself was originally derived from:
    https://github.com/adcroft/interp_and_fill/blob/6d8fc06/Interpolate%20and%20fill%20SeaWIFS.ipynb

    This implementation otherwise assumes a regular lat/lon grid (WOA),
    hence tripolar topology is intentionally not handled here.

    Periodic boundary conditions are supported in longitude only (global configuration).

    For regional configurations, set periodic_lon_laplace=False is not implemented yet.
    """
    nj, ni = field.shape
    # Find the missing points to fill (nan in field but mask > 0)
    missing_mask = np.isnan(field) & (mask > 0)
    if not np.any(missing_mask):
        # no missing data to fill but also guarantee nans on dry cells
        return np.where(mask > 0, field, np.nan)

    # change nan to 0 for the sparse matrix construction
    work = np.where(np.isnan(field), 0.0, field)
    missing_j, missing_i = np.where(missing_mask)
    n_missing = missing_j.size
    ind = np.full((nj, ni), -1, dtype=np.int64)
    ind[missing_j, missing_i] = np.arange(n_missing)

    # Sparse matrix
    A = sp.lil_matrix((n_missing, n_missing))
    b = np.zeros(n_missing)
    ld = np.zeros(n_missing)

    def _process_neighbour(n: int, jn: int, in_: int) -> None:
        """Process neighbour at (jn, in_) for row n."""
        if mask[jn, in_] <= 0:
            return

        ld[n] -= 1
        idx = ind[jn, in_]

        if idx >= 0:
            A[n, idx] = 1
        else:
            b[n] -= work[jn, in_]

    for n in range(n_missing):
        j = missing_j[n]
        i = missing_i[n]

        if periodic_lon_laplace:
            im1 = (i - 1) % ni  # west
            ip1 = (i + 1) % ni  # east
            _process_neighbour(n, j, im1)
            _process_neighbour(n, j, ip1)
        else:
            # TODO handle non-periodic case if needed
            raise NotImplementedError(
                "Non-periodic longitude is not implemented yet. "
                "Set periodic_lon_laplace=True for global grids."
            )

        if j > 0:
            _process_neighbour(n, j - 1, i)  # south
        if j < nj - 1:
            _process_neighbour(n, j + 1, i)  # north

    stabilizer = 1e-14  # prevent singular matrix
    A[np.arange(n_missing), np.arange(n_missing)] = ld - stabilizer
    x = spla.spsolve(A.tocsr(), b)
    work[missing_j, missing_i] = x
    work = np.where(mask > 0, work, np.nan)
    return work


def compute_needed_woa_source_cells(regridder, mom6_mask, source_shape):
    """
    Return a boolean mask on the woa grid (source grid).

    When True -> the source woa cell contributes via regridding weights,
    when False -> the source woa cell does not contribute to any wet MOM6 cell.
    """
    # W (n_target, n_source) sparse matrix mapping source to target cells
    # target_flat = W source_flat
    W = regridder.weights.data.tocsr()

    target_flat = mom6_mask.ravel()

    wet_target = np.where(target_flat)

    W_wet = W[wet_target]  # keep only rows corresponding to wet target cells

    needed_source_indices = np.unique(W_wet.indices)

    needed_mask = np.zeros(source_shape[0] * source_shape[1], dtype=bool)
    needed_mask[needed_source_indices] = True

    needed_mask = needed_mask.reshape(source_shape)
    return needed_mask


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
    depth_var: xr.DataArray,
    lambda1: xr.DataArray,
    topog_file: str,
    hgrid_file: str,
    method: str = "conservative_normed",
    periodic_regrid: bool = True,
    periodic_lon_laplace: bool = True,
) -> xr.Dataset:
    """
    Regrid depth_var (on regular WOA lon/lat grid) onto MOM6 grid using xESMF.
    """

    source_lon = depth_var["lon"]
    source_lat = depth_var["lat"]

    # Load model topog file then generate ocean mask ranging from [-280, 80]
    topog = xr.open_dataset(topog_file)
    mom6_mask = mom6_mask_detection(topog)

    # model hgrid
    hgrid = xr.open_dataset(hgrid_file)

    # match your slicing exactly
    hgrid_x = hgrid.x[1::2, 1::2]
    hgrid_y = hgrid.y[1::2, 1::2]
    hgrid_xc = hgrid.x[::2, ::2]
    hgrid_yc = hgrid.y[::2, ::2]

    source_ds = build_source_ds(depth_var, lon=source_lon, lat=source_lat)
    target_ds = build_target_ds(mom6_mask, hgrid_x, hgrid_y, hgrid_xc, hgrid_yc)

    regridder_kwargs = dict(
        method=method,
        periodic=periodic_regrid,
    )

    regridder = xe.Regridder(source_ds, target_ds, **regridder_kwargs)

    # compute which woa cells are needed
    needed_woa_source_cells = compute_needed_woa_source_cells(
        regridder, mom6_mask, depth_var.shape
    )

    # Allow filling in woa or needed cells
    woa_ocean = lambda1.values > 0
    fill_mask = woa_ocean | needed_woa_source_cells
    depth_var_filled_np = fill_missing_data_laplace(
        depth_var.values,
        fill_mask,
        periodic_lon_laplace=periodic_lon_laplace,
    )

    depth_var_filled = xr.DataArray(
        depth_var_filled_np,
        dims=depth_var.dims,
        coords=depth_var.coords,
        attrs=depth_var.attrs,
    )

    target_ds["h2"] = regridder(depth_var_filled)

    # check MOM wet should not be nan
    bad_cells = np.isnan(target_ds["h2"].values) & (mom6_mask > 0)
    n_bad = np.sum(bad_cells)
    if n_bad > 0:
        raise ValueError(
            f"Regridding resulted in {n_bad} NaN values in wet MOM cells."
            f"Check source data coverage and regridding method."
        )

    # MOM does not like nans input, so fill any remaining nans (e.g. in dry cells) with 0.0
    target_ds["h2"] = target_ds["h2"].fillna(0.0)

    # tidy up vars
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
            "regrid_method": method,
        },
    }

    for var, attrs in tmp_attrs.items():
        target_ds[var].attrs.update(attrs)

    return target_ds


def main():
    parser = argparse.ArgumentParser(
        description="Regridding the MOM6 input bottom roughness squared (h^2) field from the WOA grid to the target model grid."
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
        choices=[
            "nearest_s2d",
            "nearest_d2s",
            "bilinear",
            "conservative",
            "conservative_normed",
            "patch",
        ],
        help=(
            "Regridding method to use: "
            "Supported xESMF methods include: conservative, conservative_normed, "
            "bilinear, nearest_s2d, nearest_d2s, and patch. "
            "Default is conservative_normed."
        ),
    )
    parser.add_argument(
        "--periodic_regrid",
        action="store_true",
        help=(
            "Whether to use periodic regridding in x direction (longitude)."
            "Only useful for global grids with non-conservative regridding."
            "Will be forced to False for conservative regridding."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="bottom_roughness.nc",
        help="Path to the bottom roughness output file.",
    )
    parser.add_argument(
        "--periodic_lon_laplace",
        action="store_true",
        help="Whether to use periodic longitude when smooth-filling nans on the WOA grid with the laplace solver.",
    )
    args = parser.parse_args()

    ds_woa_intermediate = xr.open_dataset(args.woa_intermediate_file)

    # Regridding to model grid
    print("Regridding depth variance to MOM6 grid...")
    regrid_depth_var = regrid_depth_var_to_mom6(
        depth_var=ds_woa_intermediate["depth_var"],
        lambda1=ds_woa_intermediate["lambda1"],
        periodic_lon_laplace=args.periodic_lon_laplace,
        topog_file=args.topog_file,
        hgrid_file=args.hgrid_file,
        method=args.method,
        periodic_regrid=args.periodic_regrid,
    )
    print("Regridding done!")

    # Add provenance metadata and MD5 hashes for input files.
    this_file = os.path.normpath(__file__)
    runcmd = (
        f"python3 {os.path.basename(this_file)} "
        f"--topog_file={args.topog_file} "
        f"--hgrid_file={args.hgrid_file} "
        f"--woa_intermediate_file={args.woa_intermediate_file} "
        f"--output_file={args.output_file} "
        f"--method={args.method} "
        f"--periodic_regrid={args.periodic_regrid} "
        f"--periodic_lon_laplace={args.periodic_lon_laplace}"
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

    output_path = Path(args.output_file)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    # ensure tmp does not exist
    if tmp_path.exists():
        tmp_path.unlink()

    regrid_depth_var.to_netcdf(tmp_path)
    tmp_path.replace(output_path)

    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
