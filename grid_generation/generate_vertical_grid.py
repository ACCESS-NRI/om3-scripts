# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate a vertical grid for MOM using a hyperbolic tangent function.
#
# This script creates a vertical grid and writes it to a NetCDF file.
# This script relates to the article "Stewart, K.D., Hogg, A.M., Griffies, S.M., Heerdegen, A.P.,
# Ward, M.L., Spence, P. and England, M.H., 2017. Vertical resolution of baroclinic modes in
# global ocean models. Ocean Modelling, 113, pp.50-65. http://dx.doi.org/10.1016/j.ocemod.2017.03.012"
#
# Usage:
# Run the script with command-line arguments to specify:
#   - H: Maximum ocean depth (meters)
#   - dzd: Maximum grid spacing at depth (meters)
#   - min_dz: Minimum grid spacing at the surface (meters)
#   - depfac: Tuning parameter for grid sharpness
#
# Example command to generate KDS75 grid or 025-degree OM3 vertical grid with 75 levels:
# python generate_vertical_grid.py --H 6000 --dzd 200 --min_dz 1 --depfac 1.101
#
# Contact:
#    Ezhilsabareesh Kannadasan <ezhilsabareesh.kannadasan@anu.edu.au>
#    This script was originally developed by Kial Stewart.
# Dependencies:
#   - netCDF4
#   - numpy
# =========================================================================================

import os
from datetime import datetime
import argparse
import netCDF4 as nc
import numpy as np
from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata

# Define a small constant to initialize the iteration and prevent numerical issues
epsilon = 0.001


# Define the mathematical function that determines the vertical grid spacing
# based on depth, tuning parameter, and maximum grid spacing at depth.
def f_all(kk, H, depfac, dzd):
    return np.tanh(np.pi * ((kk) / (H * depfac))) * dzd + epsilon


def generate_vertical_grid(H, dzd, min_dz, depfac, output_filename):

    # Initialize the first two entries of the vertical grid with 0 and epsilon
    # for both depth (z) and grid spacing (dz).
    delta_z = [0, epsilon]
    prop_z = [0, epsilon]

    # Iteratively compute the next depth level by stepping from the current deepest
    # point along the defined function to generate the vertical grid.

    while prop_z[-1] + delta_z[-1] < 1.2 * H:
        aa = np.linspace(1.0, 1.5, 10000)
        bb = np.zeros(len(aa))
        loopkill = 1.0
        ii = 0
        while loopkill > 0:
            bb[ii] = f_all(prop_z[-1] + (delta_z[-1] * aa[ii]), H, depfac, dzd) - (
                delta_z[-1] * aa[ii]
            )
            loopkill = bb[ii]
            ii += 1
        aa_bb = np.polyfit(aa[: ii - 1], bb[: ii - 1], 1)
        dznew = delta_z[-1] * (np.abs(aa_bb[1] / aa_bb[0]))
        delta_z.append(dznew)
        prop_z.append(prop_z[-1] + delta_z[-1])

    # Adjust the grid vertically so that the surface spacing matches min_dz
    new_surf = np.max(
        np.where(np.array(delta_z) < min_dz)
    )  # Identify the level where spacing first reaches min_dz
    real_prop_z = (
        np.array(prop_z[new_surf:]) - prop_z[new_surf]
    )  # Shift the grid so that this level becomes the new surface
    real_delta_z = np.array(delta_z[new_surf:])  # Update the grid spacing accordingly
    real_prop_z = real_prop_z[
        real_prop_z < H
    ]  # Trim the grid to ensure it does not exceed the maximum depth H
    real_delta_z = real_delta_z[
        : len(real_prop_z)
    ]  # Trim the spacing values to match the adjusted depth levels

    # Write to NetCDF file
    write_netcdf_file(output_filename, real_prop_z, this_file)

    print(
        f"SUCCESS! A vertical grid with {len(real_prop_z) - 1} levels has been generated. "
        f"Grid spacing ranges from {real_delta_z[0]:.2f} m at the surface to {real_delta_z[-1]:.2f} m at depth. "
        f"Output written to: {output_filename}"
    )


def write_netcdf_file(output_filename, real_prop_z, this_file):
    """Function to write vertical grid data to a NetCDF file."""
    # Convert to float32 (single precision) to ensure values are exactly representable in single precision,
    # then convert back to float64 (double precision) for storage in NetCDF.
    real_prop_z_float32 = real_prop_z.astype(np.float32)
    real_prop_z_float64 = real_prop_z_float32.astype(np.float64)

    eddyfile = nc.Dataset(output_filename, "w", format="NETCDF4")
    eddyfile.createDimension("nzv", len(real_prop_z))
    zeta = eddyfile.createVariable("zeta", "f8", ("nzv",))
    zeta.units = "meters"
    zeta.standard_name = "depth"
    zeta.long_name = "vertical grid depth at top and bottom of each cell"
    eddyfile.variables["zeta"][:] = real_prop_z_float64
    eddyfile.setncatts(get_provenance_metadata())  # Obtain metadata
    eddyfile.close()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate a vertical grid for ocean models."
    )
    parser.add_argument(
        "--H", type=float, required=True, help="Maximum ocean depth (meters)"
    )
    parser.add_argument(
        "--dzd",
        type=float,
        required=True,
        help="Maximum grid spacing at depth (meters)",
    )
    parser.add_argument(
        "--min_dz",
        type=float,
        required=True,
        help="Minimum grid spacing at the surface (meters)",
    )
    parser.add_argument(
        "--depfac",
        type=float,
        required=True,
        help="Tuning parameter for grid sharpness",
    )
    parser.add_argument(
        "--output", type=str, default="ocean_vgrid.nc", help="Output NetCDF filename"
    )
    args = parser.parse_args()

    # Call the function to generate the vertical grid
    generate_vertical_grid(args.H, args.dzd, args.min_dz, args.depfac, args.output)


if __name__ == "__main__":
    main()
