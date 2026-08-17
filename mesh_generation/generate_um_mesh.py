# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate an ESMF mesh file for a UM (Unified Model) atmosphere grid.
#
# To run:
#   python generate_um_mesh.py --atm=<resolution> --mesh-filename=<output-file>
#
# <resolution> is a UM grid resolution string, e.g. n96 (New Dynamics) or n96e (End Game).
#
# The mesh has triangular elements at the poles (fanning out from a single pole node)
# and quadrilateral elements everywhere else.
#
# This is adapted from create_um_mesh() in
# https://github.com/ACCESS-NRI/land-fraction-ancillary-suite/blob/7c3f54084673eb0d0513058c4c6282ab74ead3e4/bin/create_landfrac.py#L41
#
# The run command and full github url of the current version of this script is added to the
# metadata of the generated mesh file. This is to uniquely identify the script and inputs used
# to generate the mesh file. To produce mesh files for sharing, ensure you are using a version
# of this script which is committed and pushed to github. For mesh files intended for released
# configurations, use the latest version checked in to the main branch of the github repository.
#
# Contact:
#   Anton Steketee <anton.steketee@anu.edu.au>
#
# Dependencies:
#   esmpy, xarray and numpy
# =========================================================================================

import os
import re
from datetime import datetime

import esmpy
import numpy as np
import xarray as xr

from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata


def parse_um_resolution(atm):
    """
    Parse a UM grid resolution string (e.g. "n96" or "n96e") into (nlat, nlon).
    """
    m = re.match(r"n(\d+)e?$", atm)
    if not m:
        raise ValueError(f"Could not parse atmosphere resolution string: {atm}")
    atm_res = int(m.group(1))

    nlon = 2 * atm_res
    if nlon % 4 != 0:
        raise ValueError(
            f"Invalid atmosphere resolution {atm}: 2*n must be divisible by 4"
        )
    nlat = 3 * nlon // 4

    return nlat, nlon


def create_um_mesh(nlat, nlon, output_filename):
    """
    Create and save an ESMF mesh file for a UM lat/lon grid with nlat x nlon cells.

    The poles are represented by a fan of triangular elements around a single pole
    node; all other elements are quadrilaterals.

    Adapted from create_um_mesh() in
    https://github.com/ACCESS-NRI/land-fraction-ancillary-suite/blob/7c3f54084673eb0d0513058c4c6282ab74ead3e4/bin/create_landfrac.py#L41

    Parameters
    ----------
    nlat: int
        Number of latitude cells
    nlon: int
        Number of longitude cells
    output_filename: str
        The path to the mesh file to output (netcdf)

    Returns
    -------
    esmpy.Mesh
        The mesh, loaded back in from output_filename
    """
    num_elements = nlon * nlat
    element_ids = np.arange(1, num_elements + 1)
    element_types = np.array(
        [esmpy.MeshElemType.TRI] * nlon
        + [esmpy.MeshElemType.QUAD] * ((nlat - 2) * nlon)
        + [esmpy.MeshElemType.TRI] * nlon
    )
    element_lon = np.zeros(num_elements)
    element_lon[:] = ((element_ids - 1) % nlon + 0.5) * (360 / nlon)
    element_lat = np.zeros(num_elements)
    element_lat[:] = ((element_ids - 1) // nlon + 0.5) * (180 / nlat) - 90
    element_coords = np.zeros((num_elements, 2))
    element_coords[:, 0] = element_lon
    element_coords[:, 1] = element_lat

    dx = 360 / nlon
    dy = 180 / nlat
    pi_over_180 = np.pi / 180
    element_areas = (
        dx
        * pi_over_180
        * (
            np.sin((element_lat + 0.5 * dy) * pi_over_180)
            - np.sin((element_lat - 0.5 * dy) * pi_over_180)
        )
    )

    num_nodes = nlon * (nlat - 1) + 2
    node_lon = np.zeros(num_nodes)
    node_lon[0] = 0.0
    node_lon[-1] = 0.0
    node_lon[1:-1] = element_lon[:-nlon] - 0.5 * (360 / nlon)
    node_lat = np.zeros(num_nodes)
    node_lat[0] = -90.0
    node_lat[-1] = 90.0
    node_lat[1:-1] = element_lat[:-nlon] + 0.5 * (180 / nlat)
    node_coords = np.zeros((num_nodes, 2))
    node_coords[:, 0] = node_lon
    node_coords[:, 1] = node_lat

    element_conn = []
    # south pole row: triangles fanning from node 1 (south pole)
    iy = 0
    for ix in range(nlon):
        south_pole = 1
        north_west = iy * nlon + 2 + ix
        north_east = iy * nlon + 2 + (ix + 1) % nlon
        conn = [north_west, south_pole, north_east]
        element_conn.extend(conn)

    # interior rows: quads
    for iy in range(1, nlat - 1):
        for ix in range(nlon):
            north_west = iy * nlon + 2 + ix
            north_east = iy * nlon + 2 + (ix + 1) % nlon
            south_west = (iy - 1) * nlon + 2 + ix
            south_east = (iy - 1) * nlon + 2 + (ix + 1) % nlon
            conn = [north_west, south_west, south_east, north_east]
            element_conn.extend(conn)

    # north pole row: triangles fanning from last node (north pole)
    iy = nlat - 1
    for ix in range(nlon):
        north_pole = num_nodes
        south_west = (iy - 1) * nlon + 2 + ix
        south_east = (iy - 1) * nlon + 2 + (ix + 1) % nlon
        conn = [north_pole, south_west, south_east]
        element_conn.extend(conn)

    element_conn = np.array(element_conn)

    nodeCoords = xr.DataArray(
        dims=("nodeCount", "coordDim"),
        data=node_coords,
        attrs=dict(units="degrees"),
    )
    elementConn = xr.DataArray(
        dims=("connectionCount"),
        data=element_conn.astype(np.int32),
        attrs=dict(long_name="Node indices that define the element connectivity"),
    )
    numElementConn = xr.DataArray(
        dims=("elementCount"),
        data=element_types.astype(np.int32),
        attrs=dict(long_name="Number of nodes per element"),
    )
    centerCoords = xr.DataArray(
        dims=("elementCount", "coordDim"),
        data=element_coords,
        attrs=dict(units="degrees"),
    )
    elementArea = xr.DataArray(
        dims=("elementCount"),
        data=element_areas,
        attrs=dict(units="radians^2", long_name="area weights"),
    )

    um_mesh_ds = xr.Dataset(
        dict(
            nodeCoords=nodeCoords,
            elementConn=elementConn,
            numElementConn=numElementConn,
            centerCoords=centerCoords,
            elementArea=elementArea,
        ),
        attrs=dict(
            gridType="unstructured mesh",
            timeGenerated=f"{datetime.now()}",
            created_by=f"{os.environ.get('USER')}",
        ),
    )

    # force no _FillValue (for now)
    for v in um_mesh_ds.variables:
        if "_FillValue" not in um_mesh_ds[v].encoding:
            um_mesh_ds[v].encoding["_FillValue"] = None

    um_mesh_ds.attrs |= get_provenance_metadata()

    um_mesh_ds.to_netcdf(output_filename)

    return esmpy.Mesh(filename=output_filename, filetype=esmpy.FileFormat.ESMFMESH)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate an ESMF mesh file for a UM atmosphere grid."
    )
    parser.add_argument(
        "--atm",
        required=True,
        help="Atmosphere grid horizontal resolution, e.g. n96 or n96e",
    )
    parser.add_argument(
        "--mesh-filename",
        required=True,
        help="The path to the mesh file to output (netcdf).",
    )
    args = parser.parse_args()

    nlat, nlon = parse_um_resolution(args.atm)
    create_um_mesh(nlat, nlon, args.mesh_filename)


if __name__ == "__main__":
    main()
