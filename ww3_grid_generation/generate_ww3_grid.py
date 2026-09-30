#!/usr/bin/env python3
# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate the WAVEWATCH III (WW3) grid definition files from a MOM supergrid
# (ocean_hgrid.nc) and topography (topog.nc).
#
# WW3 reads its grid as a set of plain-text arrays referenced from ww3_grid.nml. This
# script writes the five files that ACCESS-OM3 uses:
#   <gridname>.Lat    cell-centre latitudes  (from the h-points of the supergrid)
#   <gridname>.Lon    cell-centre longitudes (from the h-points of the supergrid)
#   <gridname>.Dpt    bottom depths, negative in the ocean and 0 on land
#   <gridname>.Mask   WW3 status map, 1 for wet points and 0 for land
#   <gridname>.Obstr  sub-grid obstruction, written as zeros for both x and y
#
# To run:
#   python generate_ww3_grid.py --grid-filename=<ocean_hgrid.nc> \
#     --topog-filename=<topog.nc> --gridname=<gridname> --output-dir=<dir>
# For more information, run `python generate_ww3_grid.py -h`
#
# For example, for the ACCESS-OM3 global 100km configuration:
#   python generate_ww3_grid.py \
#     --grid-filename=/g/data/vk83/prerelease/configurations/inputs/access-om3/mom/grids/mosaic/global.100km/2026.03.13/ocean_hgrid.nc \
#     --topog-filename=/g/data/vk83/prerelease/configurations/inputs/access-om3/share/grids/global.100km/2026.04.07/topog.nc \
#     --gridname=OM3_100km --output-dir=.
#
# The generated files are read by ww3_grid, which produces the mod_def.ww3 used by the
# configuration. --gridname must match the file names given in ww3_grid.nml.
#
# The h-point locations are taken as every second point of the MOM supergrid
# (Lat[1::2, 1::2]). topog.nc stores land as a fill value, so land points are masked out
# and their depth set to zero; the depth sign convention follows the WW3 grid reader:
# https://github.com/ACCESS-NRI/WW3/blob/1845b8c17321e8625829f8edad763f44722cfeac/model/src/w3gridmd.F90#L4696-L4723
#
# The outputs are plain text and cannot carry embedded metadata, so the run command, the
# full github url of the current version of this script and the md5 hashes of the inputs
# are written to a README.md alongside them. This is to uniquely identify the script and
# inputs used to generate the grid. To produce grid files for sharing, ensure you are
# using a version of this script which is committed and pushed to github. For grid files
# intended for released configurations, use the latest version checked in to the main
# branch of the github repository.
#
# Contact:
#   Ezhilsabareesh Kannadasan <ezhilsabareesh.kannadasan@anu.edu.au>
#   This script was adapted from a configuration by Shou Li, and modified to work with
#   the version of WW3 used in ACCESS-OM3.
#
# Dependencies:
#   argparse, netCDF4 and numpy
# =========================================================================================

import argparse
import os
import sys
from pathlib import Path

import netCDF4 as NC
import numpy as np

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata


def generate_ww3_grid(grid_filename, topog_filename, gridname, output_dir):
    """
    Write the WW3 .Lat, .Lon, .Dpt, .Mask and .Obstr files for gridname into output_dir.

    Returns the list of files written.
    """

    grid_handle = NC.Dataset(grid_filename, mode="r")
    topog_handle = NC.Dataset(topog_filename, mode="r")

    # Read in grid
    Lon = grid_handle.variables["x"][:, :]
    Lat = grid_handle.variables["y"][:, :]
    H_DPT = topog_handle.variables["depth"][:, :]

    # Convert the topography to a regular array and build the WW3 wet mask.
    # topog.nc uses a large negative fill value on land; WW3 needs those points
    # masked out and their depth set to zero.
    H_DPT = np.ma.filled(H_DPT, np.nan)
    H_MSK6 = np.where(np.isfinite(H_DPT) & (H_DPT > 0.0), 1, 0)
    H_DPT = np.where(H_MSK6 == 1, -np.abs(H_DPT), 0.0)

    # The h-points (cell centres) are every second point of the MOM supergrid
    H_LAT = Lat[1::2, 1::2]
    H_LON = Lon[1::2, 1::2]

    if H_LAT.shape != H_DPT.shape:
        raise ValueError(
            f"The h-point grid from {grid_filename} has shape {H_LAT.shape}, "
            f"which does not match the topography in {topog_filename} with shape "
            f"{H_DPT.shape}. Check that the supergrid and topography are for the "
            "same configuration."
        )

    LLAT = len(H_LAT[:, 0])
    LLON = len(H_LAT[0, :])

    mask_file = os.path.join(output_dir, gridname + ".Mask")
    dpt_file = os.path.join(output_dir, gridname + ".Dpt")
    obstr_file = os.path.join(output_dir, gridname + ".Obstr")
    lat_file = os.path.join(output_dir, gridname + ".Lat")
    lon_file = os.path.join(output_dir, gridname + ".Lon")

    with open(mask_file, "w") as f6, open(dpt_file, "w") as f7, open(
        obstr_file, "w"
    ) as f9, open(lat_file, "w") as f10, open(lon_file, "w") as f11:
        for ii in np.arange(0, LLAT):
            for jj in np.arange(0, LLON):
                f6.write(str(int(H_MSK6[ii, jj])) + " ")
                f7.write(str(H_DPT[ii, jj]) + " ")
                f9.write(str(0) + " ")
                f10.write(format(H_LAT[ii, jj], ".7e") + "   ")
                f11.write(format(H_LON[ii, jj], ".7e") + "   ")
            f6.write("\n")
            f7.write("\n")
            f9.write("\n")
            f10.write("\n")
            f10.write("   ")
            f11.write("\n")
            f11.write("   ")

        # WW3 expects the obstruction map as two blocks, x followed by y. Both are
        # written as zeros here, i.e. no sub-grid blocking.
        for ii in np.arange(0, LLAT):
            for jj in np.arange(0, LLON):
                f9.write(str(0) + " ")
            f9.write("\n")

    grid_handle.close()
    topog_handle.close()

    return [lat_file, lon_file, dpt_file, mask_file, obstr_file]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the WAVEWATCH III grid text files (.Lat, .Lon, .Dpt, .Mask, "
            ".Obstr) from a MOM supergrid and topography."
        )
    )
    parser.add_argument(
        "--grid-filename",
        required=True,
        type=str,
        help="The path to the MOM supergrid (ocean_hgrid.nc).",
    )
    parser.add_argument(
        "--topog-filename",
        required=True,
        type=str,
        help="The path to the topography file (topog.nc).",
    )
    parser.add_argument(
        "--gridname",
        required=True,
        type=str,
        help=(
            "The base name of the generated files, e.g. 'OM3_100km' writes "
            "OM3_100km.Lat, OM3_100km.Lon etc. This must match the file names given "
            "in ww3_grid.nml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="The directory to write the grid files into. Defaults to the current directory.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_files = generate_ww3_grid(
        args.grid_filename, args.topog_filename, args.gridname, args.output_dir
    )

    # The outputs are plain text, so record the provenance in an accompanying README
    get_provenance_metadata(
        input_files=[args.grid_filename, args.topog_filename],
        output_dir=args.output_dir,
        output_filename=output_files,
    )

    print(
        f"Wrote {len(output_files)} WW3 grid files to {os.path.abspath(args.output_dir)}"
    )


if __name__ == "__main__":
    main()
