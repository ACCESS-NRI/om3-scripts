# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate remapping weights from an ESMF mesh file for remapping a runoff field
# from an unmasked mesh to a masked mesh without losing any water volume. Each field is mapped to
# - nearby ocean cells, with weights generated with exponential decay if source cell is near a large
# volume river, otherwise
# - the nearest ocean cell.
#
# To run:
#   python generate_rof_weights.py --mesh_filename=<input_file> --weights_filename=<output_file>
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

import xarray as xr
import esmpy
from sklearn.neighbors import BallTree
from scipy.ndimage import binary_dilation
import numpy as np
from copy import copy

from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from scripts_common import get_provenance_metadata

# these lat, lon pairs represent where high volumes on runoff enter the ocean
# they represent where maximum flow in JRA55do > 0.15 kg/m^2/s, plus rivers whose mouths are
# land in the 8km mask and would otherwise be moved to a single ocean cell
# An optional third value is a bearing (degrees clockwise from north). If set, runoff is only
# spread into ocean cells within 45 degrees of that bearing from the point. Use it when
# the nearest ocean cells include a different basin.
SPREAD_POINTS = [
    (-0.125, 310.625),  # Amazon
    (8.625, 299.375),  # Tigre
    (66.125, 196.375),  # Kobuk/Noatak
    (71.875, 82.875),  # Yenisey
    (72.125, 128.875),  # Lena
    (73.375, 106.625),  # Khatanga
    (66.375, 70.625, 45),  # Ob, spread NE (Gulf of Ob, not west of Yamal)
    (68.125, 54.125),  # Pechora
    (67.125, 264.625, 0),  # Back (Chantrey Inlet), spread N (not Hudson Bay)
]

# Buffer (degrees) around spread_points to spread from
SRC_DIST = 1

# Number of cells to spread to
SPREAD_N = 75

# Half width (radians) of the cone of allowed spreading targets, for SPREAD_POINTS with a bearing
BEARING_HALF_WIDTH = np.deg2rad(45)

# Decay length (radians) when calculating spreading weights
# spreading should decrease by a factor of e (2.718) every FOLD distance
FOLD = np.deg2rad(2)

# netcdf compression settings to use
COMP_ENCODING = {"complevel": 1, "compression": "zlib"}


def initial_bearing(lat1, lon1, lat2, lon2):
    """
    Initial great-circle bearing from point 1 to point(s) 2, clockwise from north.
    All arguments and the result are in radians.
    """
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x) % (2 * np.pi)


def angle_diff(a, b):
    """Smallest absolute difference between two angles (radians)"""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def mesh_area(mesh_filename):
    """
    Calculate area for each cell in a mesh.
    per CMEPS we are using the internally calculated areas, not the mesh provided ones.
    """

    model_mesh = esmpy.Mesh(
        filename=mesh_filename,
        filetype=esmpy.FileFormat.ESMFMESH,
    )

    fld = esmpy.Field(model_mesh, meshloc=esmpy.MeshLoc.ELEMENT)

    fld.get_area()

    return copy(fld.data)


class Rof_Remapping_Weights:

    def __init__(
        self, mesh_filename, weights_filename, nx, ny, spread, input_mesh_filename=None
    ):
        # We need to generate remapping weights for use in the mediator, such that the overall volume of runoff is conserved
        # and no runoff is mapped onto land cells. For OM3, inside the mediator, the grid doesn't change as we run the mediator with
        # the ocean grid (the DROF component does the remapping from JRA grid to mediator grid). Therefore input_mesh_filename
        # defaults to mesh_filename. For CM3, input_mesh_filename can be set to a different mesh if the source (unmasked,
        # runoff-providing) grid (aka the UM) differs from the destination (masked, ocean) grid.

        self.mesh_filename = mesh_filename
        self.mesh_ds = xr.open_dataset(mesh_filename)

        if len(self.mesh_ds.elementMask) != ny * nx:
            raise RuntimeError("ny * nx does not equal number of elements in mesh file")

        self.input_mesh_filename = input_mesh_filename or mesh_filename
        self.input_mesh_ds = xr.open_dataset(self.input_mesh_filename)

        self.ny = ny
        self.nx = nx
        self.weights_f = weights_filename
        self.spread = spread

    def target_masks(self):
        """
        create masks for where runoff can be placed when being placed at coast (nospread)
        or being spread
        """
        # make new mask of land plus one adjacent cell of ocean
        ocn_mask = self.mesh_ds.elementMask.values == 1
        land_neighbours = binary_dilation(np.reshape(ocn_mask, (self.ny, self.nx)) == 0)
        land_neighbours = land_neighbours.flatten()

        # target for no spread runoff is ocean cells which are adjacent land
        self.target_mask_nospread = xr.DataArray(
            (land_neighbours & ocn_mask) == 1, dims="elementCount"
        )

        # target for spread runoff is whole ocean
        self.target_mask_spread = self.mesh_ds.elementMask == 1

    def source_mask(self):
        """
        create mask of cells within a small range of points to be spread
        these are cells which should be spread
        """
        # for each cell in the input mesh, find the nearest point in SPREAD_POINTS (if any)
        # and mark it (with the index in SPREAD_POINTS) for spreading
        # if source regions overlap, the later point in SPREAD_POINTS takes precedence
        self.spread_source = np.full(
            self.input_mesh_ds.sizes["elementCount"], -1, dtype=int
        )

        if self.spread:

            lon = self.input_mesh_ds.centerCoords[:, 0].values
            lat = self.input_mesh_ds.centerCoords[:, 1].values

            for ipoint, (plat, plon, *_) in enumerate(SPREAD_POINTS):

                # Handle longitude wraparound correctly
                dlon = ((lon - plon + 180) % 360) - 180

                point_mask = (abs(lat - plat) <= SRC_DIST) & (abs(dlon) <= SRC_DIST)

                self.spread_source[point_mask] = ipoint

        self.spread_source_da = xr.DataArray(self.spread_source, dims="elementCount")

    def spread_weights(self):
        """
        For each of the spread and nospread target masks, use a ball tree to find nearest ocean cells.
        For cells in the spread source mask, spread with exponential decay away from the cell
        For cells ourside this mask, map to the closest ocean cell only.
        """

        # Haversine distances expect lat first, lon second, so index coordDim backwards
        center_coords_rad = np.deg2rad(self.mesh_ds.centerCoords.isel(coordDim=[1, 0]))

        # Make a BallTree from the target cells for no spread
        mask_tree_nospread = BallTree(
            center_coords_rad.where(self.target_mask_nospread, drop=True),
            metric="haversine",
        )
        target_mask_nospread_i = self.mesh_ds.elementCount.where(
            self.target_mask_nospread, other=0, drop=True
        )

        # Using the Trees, look up the nearest ocean cell to every source grid cell.
        # input is all cells of the input/source mesh
        in_coords_rad = np.deg2rad(
            self.input_mesh_ds.centerCoords.isel(coordDim=[1, 0])
        )

        # nearest 1 - no spread
        i_1 = mask_tree_nospread.query(
            in_coords_rad,
            k=1,
            return_distance=False,
        )

        # From https://earthsystemmodeling.org/docs/release/ESMF_5_2_0rp3/ESMF_refdoc/node3.html :
        # " The indices and weights generated by ESMF_FieldRegridStore() are stored in the output file as variables col, row
        # and S. Where col and row are the indices to the source and the destination grid cells. These are a one-dimension
        # array with length defined by dimension n_s. S is the weight which is multiplied by the source value indicated by
        # col and then summed with the destination value indicated by row to build the final interpolated value of the destination.

        # make 2d array of indices and weights, then fill with spread/no_spread depending on the row
        n = len(in_coords_rad)
        indices = np.full((n, SPREAD_N), -1, dtype=int)
        weights = np.zeros((n, SPREAD_N), dtype=float)

        # we made i_1 for all source cells, use it for cells which are not spread

        # Nospread points
        # when using where on integers, set other=0 to avoid converting to floats
        nospread_i = self.input_mesh_ds.elementCount.where(
            self.spread_source_da < 0, other=0, drop=True
        )

        row = target_mask_nospread_i[i_1[nospread_i, 0]]

        # Get the mesh element areas and adjust for area change between the
        # source cell (i) and destination (row)
        area = mesh_area(self.mesh_filename)
        source_area = mesh_area(self.input_mesh_filename)

        indices[nospread_i, 0] = row
        weights[nospread_i, 0] = source_area[nospread_i] / area[row]

        # Spread points: for each point in SPREAD_POINTS, find the SPREAD_N nearest
        # target cells to each of its source cells and spread with exponential decay
        if self.spread:

            target_i = self.mesh_ds.elementCount.where(
                self.target_mask_spread, other=0, drop=True
            ).values  # Index for the target_mask (other=0 preserves integer dtype)
            target_coords_rad = center_coords_rad.where(
                self.target_mask_spread, drop=True
            ).values

            # Ball tree targeting all ocean, can be usef for points without a bearing
            ocean_tree = BallTree(target_coords_rad, metric="haversine")

            for ipoint in np.unique(self.spread_source[self.spread_source >= 0]):
                plat, plon, *bearing = SPREAD_POINTS[ipoint]
                # Get the indices of the source cells which are to be spread from this point
                point_spread_i = np.where(self.spread_source == ipoint)[0]

                if bearing:
                    # only target cells within BEARING_HALF_WIDTH of the bearing
                    allowed = np.where(
                        angle_diff(
                            initial_bearing(
                                np.deg2rad(plat),
                                np.deg2rad(plon),
                                target_coords_rad[:, 0],
                                target_coords_rad[:, 1],
                            ),
                            np.deg2rad(bearing[0]),
                        )
                        <= BEARING_HALF_WIDTH
                    )[0]
                    tree = BallTree(target_coords_rad[allowed], metric="haversine")
                else:
                    allowed = np.arange(len(target_i))
                    tree = ocean_tree

                dist, i_n = tree.query(
                    in_coords_rad.values[point_spread_i],
                    k=SPREAD_N,
                    return_distance=True,
                )
                row = target_i[allowed[i_n]]

                # Exponential weights, normalised for the area change between the
                # source cell and each destination so volume is conserved
                w = np.exp(-dist / FOLD)
                area_frac = source_area[point_spread_i, None] / area[row]
                w /= np.sum(w / area_frac, axis=1, keepdims=True)

                indices[point_spread_i] = row
                weights[point_spread_i] = w

        # Flatten sparse structure
        col = np.repeat(self.input_mesh_ds.elementCount, SPREAD_N)
        row = indices.ravel()
        S = weights.ravel()

        self.create_weights_file(row, col, S)

    def create_weights_file(self, row, col, S):
        """ "
        For provided data (row, col, S) save it to a newtcdf weights file
        """

        # Convert to ESMF 1-based indexing
        row += 1
        col += 1

        weights_ds = xr.Dataset()
        weights_ds["col"] = xr.DataArray(data=col, dims="n_s")
        weights_ds["row"] = xr.DataArray(data=row, dims="n_s")
        weights_ds["S"] = xr.DataArray(data=S, dims="n_s")

        # Remove zero weights
        weights_ds = weights_ds.where(weights_ds["S"] != 0, other=0, drop=True)

        # add global attributes
        weights_ds.attrs = {
            "gridType": "unstructured mesh",
        }

        # add git info to history
        input_files = [self.mesh_filename]
        if self.input_mesh_filename != self.mesh_filename:
            input_files.append(self.input_mesh_filename)

        weights_ds.attrs |= get_provenance_metadata(
            input_files, output_filename=self.weights_f, licence="CC BY 4.0"
        )

        # save (compressed)
        encoding = {}
        for iVar in weights_ds.data_vars:
            encoding[iVar] = COMP_ENCODING
        weights_ds.to_netcdf(self.weights_f, encoding=encoding)


def gen_rof_weights(
    mesh_filename, weights_filename, nx, ny, spread=True, input_mesh_filename=None
):
    """
    Conveniance function for doing the whole weights generation
    """
    rof_weights = Rof_Remapping_Weights(
        mesh_filename, weights_filename, nx, ny, spread, input_mesh_filename
    )
    rof_weights.target_masks()
    rof_weights.source_mask()
    rof_weights.spread_weights()


def main():
    parser = argparse.ArgumentParser(
        description="Create an remapping weights to transfer runoff from unmasked mesh to masked mesh using ESMF mesh file."
    )

    parser.add_argument(
        "--mesh_filename",
        type=str,
        required=True,
        help="The path to the mesh file specifying the model (destination) grid "
        "and land mask.",
    )
    parser.add_argument(
        "--input_mesh_filename",
        type=str,
        required=False,
        default=None,
        help="The path to the mesh file specifying the source (runoff-providing) grid, "
        "if different to mesh_filename. Defaults to mesh_filename.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        required=True,
        help="Number of global ocean cells in x directon",
    )
    parser.add_argument(
        "--ny",
        type=int,
        required=True,
        help="Number of global ocean cells in y directon",
    )
    parser.add_argument(
        "--weights_filename",
        type=str,
        required=True,
        help="The path to the weights file to output (netcdf).",
    )
    parser.add_argument(
        "--spread",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=True,
        help="Spread large rivers",
    )

    args = parser.parse_args()

    return gen_rof_weights(
        args.mesh_filename,
        args.weights_filename,
        args.nx,
        args.ny,
        args.spread,
        args.input_mesh_filename,
    )


if __name__ == "__main__":
    import argparse

    main()
