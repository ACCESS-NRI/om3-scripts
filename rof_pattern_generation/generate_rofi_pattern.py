# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate a spreading pattern to spread iceberge runoff fluxes, based on regridding
# the Mankoff 2025 spreading climatology dataset and moving any results on land to the nearest ocean cell
#
# To run:
#   python generate_rofi_pattern.py --hgrid-filename=<path-to-supergrid-file>
# --output-filename=<path-to-output-file> --topog-file=<path-to-bathymetry-file>
#
# This script currently supports using the hgrid file and topog file in MOM6 formats
# for the ocean grid and mask
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
#   xesmf, xarray and sklearn
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

# source data to regrid
AQ_MELT_PATTERN = "/g/data/av17/access-nri/OM3/Mankoff_2025_V9/AQ_iceberg_melt.nc"
GL_MELT_PATTERN = "/g/data/av17/access-nri/OM3/Mankoff_2025_V9/GL_iceberg_melt.nc"

# attributes to copy from source data
ATTRS = [
    "title",
    "history",
    "original_data_source",
]


def move_runoff_on_land(grid_dest_in, mask, forcing_regrid_glob):
    """
    For a provided grid, land mask, and DataArray, move any data which is on land
    into ocean, usign a nearest neighbour algorithm
    """

    grid_dest = grid_dest_in.copy()
    grid_dest["mask"] = xr.DataArray(mask, dims=["ny", "nx"])

    nx = len(grid_dest.nx)
    ny = len(grid_dest.ny)

    # convert destination grid into 1d (lat, lon)
    # source cells are all grid cells, target cells are ocean cells only
    grid_dest["i"] = xr.DataArray(
        np.arange(0, ny * nx).reshape(ny, nx), dims=["ny", "nx"]
    )
    grid_stacked = grid_dest[["lat", "lon", "i"]].stack(points=["ny", "nx"])
    source_cells = list(zip(grid_stacked["lat"].values, grid_stacked["lon"].values))
    source_cells_rad = np.deg2rad(source_cells)

    mask_stacked = grid_stacked.where(
        grid_dest["mask"].stack(points=["ny", "nx"]), drop=True
    )
    target_cells = list(zip(mask_stacked["lat"].values, mask_stacked["lon"].values))
    target_cells_rad = np.deg2rad(target_cells)

    # create a BallTree (nearest neighbour) and query for all source cells
    mask_tree = BallTree(target_cells_rad, metric="haversine")
    ii = mask_tree.query(source_cells_rad, return_distance=False)
    # ii is index in target_cells, convert to grid index
    new_index = mask_stacked.i.values[ii[:, 0]].astype(np.int64)

    # adjustment for fraction of old_area/new_area, use esmf grid areas for consistency with CMEPS
    area = cell_area(grid_dest)
    area_1d = area.values.flatten()
    area_frac = area_1d / area_1d[new_index]

    # convert regrid result into 1d, move to nearest neighbour where needed
    weights = np.reshape(forcing_regrid_glob.values, (12, nx * ny))
    weights_adj = copy(weights)
    for i, new_i in enumerate(new_index):
        if i != new_i:
            weights_adj[:, i] = 0
            weights_adj[:, new_i] += weights[:, i] * area_frac[i]

    # new output
    weights_da = xr.DataArray(
        np.reshape(weights_adj, (12, ny, nx)), dims=["time", "rof_ny", "rof_nx"]
    )

    return weights_da


def main():

    regrid_aq = Regrid_Common()

    regrid_aq.parser.description = (
        "Interpolate Mankoff 2025 climatology of iceberg spreading to to a grid "
        "specified by a provided MOM supergrid file and landmask from a MOM topog "
        " file. For use in CMEPS."
    )

    regrid_aq.parser.add_argument(
        "--topog-file",
        type=str,
        required=True,
        help="Path to the model topography file, which is used to generate the model mask.",
    )

    regrid_aq.parse_cli()

    #  We need one regrid instance for each hemisphere:
    regrid_gl = copy(regrid_aq)
    regrid_aq.forcing_filename = AQ_MELT_PATTERN
    regrid_gl.forcing_filename = GL_MELT_PATTERN

    # stash some metadata for later and then regrid each hemsiphere
    global_attrs = {}
    for regrid in [regrid_aq, regrid_gl]:
        regrid.open_datasets()
        for k in ATTRS:
            val = regrid.forcing_src.attrs.get(k)
            if val is not None:
                if k in global_attrs:
                    global_attrs[k] += f"\n{val}"
                else:
                    global_attrs[k] = val

        global_attrs["Mankoff_doi"] = regrid.forcing_src.attrs["DOI"]

        # combine forcing for all regions
        regrid.forcing_src = regrid.forcing_src.drop_vars(
            ["region_map", "region_map_expanded"]
        ).sum("region")[["melt"]]
        regrid.regrid_forcing()

    # combine two regridding results
    regrid = regrid_aq
    forcing_regrid_glob = regrid_aq.forcing_regrid + regrid_gl.forcing_regrid

    # find the ocean mask using the bathymetry
    topo = xr.open_dataset(regrid.args.topog_file)
    mask = mom6_mask_detection(topo)

    # After doing the regridding, map any runoff on land cells into the ocean
    weights_da = move_runoff_on_land(
        regrid.grid_dest, mask, forcing_regrid_glob["melt"]
    )

    weights_ds = weights_da.to_dataset(name="pattern_Forr_rofi")

    # Set calendar
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

    # Info about input data used
    file_hashes = [
        f"{AQ_MELT_PATTERN} (md5 hash: {md5sum(AQ_MELT_PATTERN)})",
        f"{GL_MELT_PATTERN} (md5 hash: {md5sum(GL_MELT_PATTERN)})",
        f"{regrid.hgrid_filename} (md5 hash: {md5sum(regrid.hgrid_filename)})",
        f"{regrid.args.topog_file} (md5 hash: {md5sum(regrid.args.topog_file)})",
    ]
    if regrid.mask_filename:
        file_hashes.append(
            f"{regrid.mask_filename} (md5 hash: {md5sum(regrid.mask_filename)})"
        )

    global_attrs |= {
        "description": "Mankoff 2025 iceberg spreading climatology remapped onto an ACCESS-OM3 grid",
        "history": get_provenance_metadata(this_file, runcmd),
        "inputFile": ", ".join(file_hashes),
    }

    weights_ds.attrs = weights_ds.attrs | global_attrs

    # Save output
    var_encoding = dict(zlib=True, complevel=1, _FillValue=-1.0e36)
    for var in weights_ds.data_vars:
        weights_ds[var].encoding |= var_encoding
    # Coordinates should not have _FillValue
    coord_encoding = dict(_FillValue=None)
    for coord in weights_ds.coords:
        weights_ds[coord].encoding |= coord_encoding

    unlimited_dims = "time" if "time" in weights_ds.dims else None
    weights_ds.to_netcdf(regrid.output_filename, unlimited_dims=unlimited_dims)


if __name__ == "__main__":

    main()
