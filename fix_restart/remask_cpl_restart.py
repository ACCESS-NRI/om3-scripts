# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import argparse
import numpy as np
import netCDF4 as nc
from scipy import ndimage as nd
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum

"""
This script updates ACCESS-OM3 coupler restart files by infilling missing (masked) values
based on the surrounding ocean cells and reapplying a land mask using a provided mask file
(e.g., kmt.nc).

Note: This process only modifies the **surface-level cells** (i.e., the topmost levels) 
and does not fill vertical profiles for newly wet columns introduced by bathymetry changes.

Example usage:
python remask_cpl_restart.py --input_file /path/to/access-om3.cpl.r.0000-01-01-00000.nc --mask_file /path/to/kmt.nc --mask_var kmt
"""


def unmask_2d(var, mask, missing_value):
    if mask is None:
        if missing_value is None:
            raise ValueError("missing_value must be provided when mask is None.")
        mask = np.zeros_like(var.data)
        mask[np.where(var.data == missing_value)] = 1

    # Find indices of the nearest valid (unmasked) points for each grid cell
    ind = nd.distance_transform_edt(
        mask[:, :], return_distances=False, return_indices=True
    )
    var[:, :] = var[tuple(ind)]


def unmask_3d(v, mask, missing_value):
    for t in range(v.shape[0]):
        unmask_2d(v[t, :], mask, missing_value)


def unmask_4d(v, mask, missing_value):
    for t in range(v.shape[0]):
        unmask_3d(v[t, :], mask, missing_value)


def unmask_file(ncfile, mask=None, missing_value=None, skip_vars=[]):
    for v in ncfile.variables:
        if v in skip_vars or v.startswith("atm"):
            continue
        var = ncfile.variables[v][:]
        if mask is None and missing_value is None:
            missing_value = var.fill_value

        if len(var.shape) == 4:
            unmask_4d(var, mask, missing_value)
        elif len(var.shape) == 3:
            unmask_3d(var, mask, missing_value)
        elif len(var.shape) == 2:
            unmask_2d(var, mask, missing_value)
        else:
            print(f"WARNING: not unmasking {v} because it is 1D")
        ncfile.variables[v][:] = var[:]


def apply_mask_2d(v, landmask, mask_val):
    v[np.where(landmask)] = mask_val


def apply_mask_3d(v, landmask, mask_val):
    for d in range(v.shape[0]):
        apply_mask_2d(v[d, :], landmask, mask_val)


def apply_mask_4d(v, landmask, mask_val):
    for t in range(v.shape[0]):
        apply_mask_3d(v[t, :], landmask, mask_val)


def apply_mask_file(ncfile, mask, mask_val=0.0, skip_vars=[]):
    for v in ncfile.variables:
        if v in skip_vars or v.startswith("atm"):
            continue

        var = ncfile.variables[v][:]

        if len(var.shape) == 4:
            apply_mask_4d(var, mask, mask_val)
        elif len(var.shape) == 3:
            apply_mask_3d(var, mask, mask_val)
        elif len(var.shape) == 2:
            apply_mask_2d(var, mask, mask_val)
        else:
            print(f"WARNING: not applying mask {v} because it is 1D")
        ncfile.variables[v][:] = var[:]


def main():
    parser = argparse.ArgumentParser(
        description="Fix missing values in restart files and apply land mask using kmt"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the NetCDF restart file to be fixed",
    )
    parser.add_argument(
        "--mask_file",
        required=True,
        help="Path to the NetCDF file containing the land mask or kmt",
    )
    parser.add_argument(
        "--mask_var",
        default="kmt",
        help="Name of the mask variable in the mask file (default: kmt)",
    )
    args = parser.parse_args()

    # all atmosphere variables plus the list below are skipped
    skip_vars = [
        "time",
        "time_bnds",
        "start_ymd",
        "start_tod",
        "curr_ymd",
        "curr_tod",
        "ocnExpAccum_cnt",
    ]
    missing_value = 1e30

    # Load mask
    with nc.Dataset(args.mask_file) as f:
        mask = np.array(f.variables[args.mask_var][:], dtype=bool)
        mask = ~mask  # dry = True

    with nc.Dataset(args.input_file, "r+") as f:

        unmask_file(f, mask, missing_value, skip_vars=skip_vars)
        apply_mask_file(f, mask, skip_vars=skip_vars)

        this_file = os.path.normpath(__file__)
        runcmd = f"python3 {os.path.basename(this_file)} --input_file {args.input_file} --mask_file {args.mask_file} --mask_var {args.mask_var}"

        # Add metadata
        f.setncattr(
            "title",
            "Coupler restart fields updated with land mask of modified bathymetry",
        )
        f.setncattr("history", get_provenance_metadata(this_file, runcmd))
        f.setncattr(
            "mask_file",
            f"{os.path.abspath(args.mask_file)} (md5 hash: {md5sum(args.mask_file)})",
        )
        f.setncattr(
            "input_file",
            f"{os.path.abspath(args.input_file)} (md5 hash: {md5sum(args.input_file)})",
        )


if __name__ == "__main__":
    sys.exit(main())
