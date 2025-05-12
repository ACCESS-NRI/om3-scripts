# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import argparse
import numpy as np
import netCDF4 as nc
from scipy import ndimage as nd
from datetime import datetime

"""
This script updates coupler restart files by unmasking the restart and then applying a updated land masks.

Example usage:
python fix_cpl_restart.py --input_file /path/to/access-om3.cpl.r.*.nc --mask_file /path/to/kmt.nc --mask_var kmt
"""


def unmask_2d(var, mask, missing_value):
    if mask is None:
        if missing_value is None:
            raise ValueError("missing_value must be provided when mask is None.")
        mask = np.zeros_like(var.data)
        mask[np.where(var.data == missing_value)] = 1

    ind = nd.distance_transform_edt(
        mask[:, :], return_distances=False, return_indices=True
    )
    var[:, :] = var[tuple(ind)]
    print("2d done", flush=True)


def unmask_3d(v, mask, missing_value):
    for t in range(v.shape[0]):
        unmask_2d(v[t, :], mask, missing_value)


def unmask_4d(v, mask, missing_value):
    for t in range(v.shape[0]):
        unmask_3d(v[t, :], mask, missing_value)


def unmask_file(filename, mask=None, missing_value=None, skip_vars=[]):
    with nc.Dataset(filename, "r+") as f:
        for v in f.variables:
            if v in skip_vars or v.startswith("atm"):
                continue
            print(f"Unmasking variable: {v}")
            var = f.variables[v][:]
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
            f.variables[v][:] = var[:]

def apply_mask_2d(v, landmask, mask_val):
    v[np.where(landmask)] = mask_val


def apply_mask_3d(v, landmask, mask_val):
    for d in range(v.shape[0]):
        apply_mask_2d(v[d, :], landmask, mask_val)


def apply_mask_4d(v, landmask, mask_val):
    for t in range(v.shape[0]):
        apply_mask_3d(v[t, :], landmask, mask_val)


def apply_mask_file(filename, mask, mask_val=0.0, skip_vars=[]):
    with nc.Dataset(filename, "r+") as f:
        for v in f.variables:
            if v in skip_vars or v.startswith("atm"):
                continue

            var = f.variables[v][:]

            if len(var.shape) == 4:
                apply_mask_4d(var, mask, mask_val)
            elif len(var.shape) == 3:
                apply_mask_3d(var, mask, mask_val)
            elif len(var.shape) == 2:
                apply_mask_2d(var, mask, mask_val)
            else:
                print(f"WARNING: not applying mask {v} because it is 1D")
            f.variables[v][:] = var[:]
        
        # Add metadata
        f.setncattr(
            "title",
            "Coupler restart fields updated with land mask of modified bathymetry",
        )
        f.setncattr("history", f"Updated on {datetime.now().strftime('%Y-%m-%d')}")
        f.setncattr("source", "fix_cpl_restart.py")
        f.setncattr("run_command", " ".join(sys.argv))


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

    unmask_file(args.input_file, mask, missing_value, skip_vars=skip_vars)
    apply_mask_file(args.input_file, mask, mask_val=0.0, skip_vars=skip_vars)


if __name__ == "__main__":
    sys.exit(main())
