# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate an spreading pattern to spread iceberge runoff fluxes, based on regridding
# an input dataset and ensuring all target cells are ocean cells
#
# To run:
#   python generate_rofi_pattern.py --mesh_filename=<input_file> --weights_filename=<output_file>
#
# This script currently supports mesh files in the ESMF unstructed mesh format.
#
# There is not enough memory on the gadi login node to run this, its simplest to run in
# a terminal through are.nci.org.au
#
# The run command and full github url of the current version of this script is added to the
# metadata of the generated weights file. This is to uniquely identify the script and inputs used
# to generate the mesh file. To produce weights files for sharing, ensure you are using a version
# of this script which is committed and pushed to github. For mesh files intended for released
# configurations, use the latest version checked in to the main branch of the github repository.
#
# Contact:
#   Anton Steketee <anton.steketee@anu.edu.au>
#
# Dependencies:
#   esmpy, xarray and scipy
# =========================================================================================


import os
import sys
from copy import copy
from sklearn.neighbors import BallTree
from xesmf.util import cell_area
import xarray as xr
import numpy as np

from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from regrid_common import Regrid_Common
from scripts_common import get_provenance_metadata, md5sum
from mesh_generation.generate_mesh import mom6_mask_detection

# in a climatology, with 365 day calendar, whats the day of the middle of each month
DAY_IN_MONTH = [15.5, 45, 74.5, 105, 135.5, 166, 196.5, 227.5, 258, 288.5, 319, 349.5]


def main():

    regrid = Regrid_Common()

    # regrid.parser.description = (
    #     regrid.parser.description +
    #     " For use in the MOM data_table."
    # )

    regrid.parser.add_argument(
        "--topog-file",
        type=str,
        required=True,
        help="Path to the model topography file, which is used to generate the model mask.",
    )

    regrid.parse_cli()

    regrid.open_datasets()

    # stash some metadata for later
    global_attrs = {
        k: regrid.forcing_src.attrs[k]
        for k in [
            "description",
            "original_data",
            "title",
            "history",
            "original_data_source",
        ]
        if k in regrid.forcing_src.attrs
    }

    global_attrs["Makoff_doi"] = regrid.forcing_src.attrs["DOI"]

    # combine forcing for all regions
    regrid.forcing_src = (
        regrid.forcing_src.drop_vars(["region_map", "region_map_expanded"])
        .sum("region")
        .melt
    )

    regrid.regrid_forcing()

    # load topog.nc
    topo = xr.open_dataset(regrid.args.topog_file)
    mask = mom6_mask_detection(topo)

    grid_dest = regrid.grid_dest

    grid_dest["mask"] = xr.DataArray(mask, dims=["ny", "nx"])

    nx = len(grid_dest.nx)
    ny = len(grid_dest.ny)

    # convert destination grid into 1d (lat, lon)
    # source cells are all grid cells
    # target cells are ocean cells only
    grid_dest["i"] = xr.DataArray(
        np.arange(0, ny * nx).reshape(ny, nx), dims=["ny", "nx"]
    )
    grid_stacked = grid_dest[["lat", "lon", "i"]].stack(points=["ny", "nx"])
    source_index = grid_stacked.i.values
    source_cells = list(zip(grid_stacked["lat"].values, grid_stacked["lon"].values))
    source_cells_rad = np.deg2rad(source_cells)

    mask_stacked = grid_stacked.where(
        grid_dest["mask"].stack(points=["ny", "nx"]), drop=True
    )
    target_cells = list(zip(mask_stacked["lat"].values, mask_stacked["lon"].values))
    target_index = mask_stacked["i"].values
    target_cells_rad = np.deg2rad(target_cells)

    # create a BallTree (nearest neighbour) and query for all source cells
    mask_tree = BallTree(target_cells_rad, metric="haversine")
    ii = mask_tree.query(source_cells_rad, return_distance=False)
    # result is index in target_cells, convert to grid index
    new_index = mask_stacked.i.values[ii[:, 0]].astype(np.int64)

    # use esmf grid areas for consistency with CMEPS
    area = cell_area(grid_dest)

    # adjustment for area of old area vs new area
    area_1d = area.values.flatten()
    area_frac = area_1d / area_1d[new_index]

    # convert regrid result into 1d, move to nearest neighbour where needed
    weights = np.reshape(regrid.forcing_regrid.values, (12, nx * ny))
    weights_adj = copy(weights)
    for i, new_i in enumerate(new_index):
        if i != new_i:
            weights_adj[:, i] = 0
            weights_adj[:, new_i] += weights[:, i] * area_frac[i]

    # new output
    weights_da = xr.DataArray(
        np.reshape(weights_adj, (12, 300, 360)), dims=["time", "rofi_ny", "rofi_nx"]
    )

    weights_ds = weights_da.to_dataset(name="pattern_Forr_rofi")

    weights_ds["time"] = DAY_IN_MONTH

    weights_ds.time.attrs = {
        "standard_name": "time",
        "calendar": "proleptic_gregorian",
        "units": "days since 0001-01-01 00:00:00",
    }

    # Add some info about how the file was generated
    this_file = os.path.normpath(__file__)

    runcmd = (
        f"python3 {os.path.basename(this_file)} --topog-file={regrid.args.topog_file} "
        f"{regrid.runcmd_args}"
    )

    # add md5 hashes for input files
    # file_hashes = [
    #     f"{args.topog_file} (md5 hash: {md5sum(args.topog_file)})",
    # ]
    # global_attrs["inputFile"] = ", ".join(file_hashes)
    # tideamp.attrs.update(global_attrs)

    global_attrs |= {
        "description": "Mankoff 2025 iceberg spreading climatology remapped onto an ACCESS-OM3 grid",
        "history": get_provenance_metadata(this_file, runcmd),
    }

    weights_ds.attrs = weights_ds.attrs | global_attrs

    regrid.forcing_regrid = weights_ds

    regrid.save_output()


if __name__ == "__main__":

    main()
