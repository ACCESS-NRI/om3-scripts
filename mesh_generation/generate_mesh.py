# Copyright 2023 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate an ESMF mesh file from an input grid.
#
# To run:
#   python generate_mesh.py --grid-type=<grid-type> --grid-filename=<grid-file> \
#     --mask-filename=<mask-file> --mesh-filename=<output-file> --lon-name=<lon-name> \
#     --lat-name=<lat-name> --area-name=<area-name> --wrap-lons
#
# This script currently supports two grid-types:
#   - "mom" to generate a mesh representation of h-cells from a MOM supergrid
#   - "latlon" to generate a mesh representation from lat/lon locations
# For more information, run `python generate_mesh.py -h`
#
# The run command and full github url of the current version of this script is added to the
# metadata of the generated mesh file. This is to uniquely identify the script and inputs used
# to generate the mesh file. To produce mesh files for sharing, ensure you are using a version
# of this script which is committed and pushed to github. For mesh files intended for released
# configurations, use the latest version checked in to the main branch of the github repository.
#
# Contact:
#   Dougie Squire <dougie.squire@anu.edu.au>
#
# Dependencies:
#   argparse, xarray, numpy and pandas
# =========================================================================================

import os
from datetime import datetime

import numpy as np
import xarray as xr
import cf_xarray as cfxr
import pandas as pd

from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum

EARTH_R = 6.37122e6


class BaseGrid:

    def __init__(
        self,
        x_centres,
        y_centres,
        x_corners,
        y_corners,
        area=None,
        mask=None,
        inputs=None,
    ):
        """
        Initialise a mesh object

        Parameters
        ----------
        x_centres: len(elementCount) array-like
            Longitudinal positions of the element centre coords
        y_centres: len(elementCount) array-like
            Latitudinal positions of the element centre coords
        x_corners: (elementCount x 4) array-like
            Longitudinal positions of the corner nodes of each element, ordered ll, lr, ur, ul
        y_corners: (elementCount x 4) array-like
            LongitLatitudinaludinal positions of the corner nodes of each element, ordered ll, lr, ur, ul
        area: len(elementCount) array-like, optional
            Areas of each element
        mask: len(elementCount) array-like
            Mask values for each element, optional
        inputs: str or list of str, optional
            Paths to the files used to create the grid
        """

        self.x_centres = x_centres
        self.y_centres = y_centres

        self.x_corners = x_corners.flatten()
        self.y_corners = y_corners.flatten()

        self.area = area
        self.mask = mask

        if isinstance(inputs, str):
            inputs = [inputs]
        self.inputs = inputs

        self.mesh = None

    def create_mesh(self, wrap_lons=False, global_attrs=None):
        """
        Create the mesh as an xarray Dataset

        Parameters
        ----------
        wrap_lons: boolean, optional
            If True, wrap longitude values into the range between 0 and 360
        global_attrs: dict
            Global attributes to the mesh object
        """

        if wrap_lons:
            self.x_centres = (self.x_centres + 360) % 360
            self.x_corners = (self.x_corners + 360) % 360

        centres = np.stack((self.x_centres, self.y_centres), axis=1)
        corners_df = pd.DataFrame({"x": self.x_corners, "y": self.y_corners})

        # calculate indexes of corner nodes per element
        elem_conn = (
            (corners_df.groupby(["x", "y"], sort=False).ngroup() + 1)
            .to_numpy()
            .reshape((-1, 4))
        )

        # calculate corner nodes
        nodes = corners_df.drop_duplicates().to_numpy()

        # create mask if we don't have one
        if self.mask is None:
            self.mask = np.ones_like(self.x_centres, dtype=np.int8)

        # create a new dataset for the mesh
        ds = xr.Dataset()
        ds["nodeCoords"] = xr.DataArray(
            nodes.astype(np.float64),
            dims=("nodeCount", "coordDim"),
            attrs={"units": "degrees"},
        )
        ds["elementConn"] = xr.DataArray(
            elem_conn.astype(np.int32),
            dims=("elementCount", "maxNodePElement"),
            attrs={"long_name": "Node indices that define the element connectivity"},
        )
        ds["numElementConn"] = xr.DataArray(
            4 * np.ones_like(self.x_centres, dtype=np.int32),
            dims=("elementCount"),
            attrs={"long_name": "Number of nodes per element"},
        )
        ds["centerCoords"] = xr.DataArray(
            centres.astype(np.float64),
            dims=("elementCount", "coordDim"),
            attrs={"units": "degrees"},
        )

        ds["elementMask"] = xr.DataArray(
            self.mask.astype(np.int8),
            dims=("elementCount"),
        )

        if self.area is not None:
            ds["elementArea"] = xr.DataArray(
                (self.area / (EARTH_R * EARTH_R)).astype(np.float64),
                dims=("elementCount"),
                attrs={"units": "radians^2", "long_name": "area weights"},
            )

        # force no _FillValue (for now)
        for v in ds.variables:
            if "_FillValue" not in ds[v].encoding:
                ds[v].encoding["_FillValue"] = None

        # add global attributes
        ds.attrs = {
            "gridType": "unstructured mesh",
            "timeGenerated": f"{datetime.now()}",
            "created_by": f"{os.environ.get('USER')}",
        }
        if self.inputs:
            file_hashes = []
            for input in self.inputs:
                file_hashes.append(f"{input} (md5 hash: {md5sum(input)})")
            ds.attrs["inputFile"] = ", ".join(file_hashes)

        # add git info to history
        if global_attrs:
            ds.attrs |= global_attrs

        self.mesh = ds

        return self

    def write(self, filename):
        """
        Save the mesh to a file
        """

        if self.mesh is None:
            raise ValueError(
                "Before writing, you must first create the mesh object using self.create_mesh()"
            )

        self.mesh.to_netcdf(filename)


class MomSuperGrid(BaseGrid):

    def __init__(
        self,
        hgrid_filename,
        mask_filename=None,
        lon_name=None,
        lat_name=None,
        area_name=None,
    ):
        """
        Initialise a mesh representation of h-cells from a MOM supergrid

        Parameters
        ----------
        hgrid_filename: str
            Path to the MOM hgrid netcdf file
        mask_filename: str, optional
            Path to a netcdf file containing a mask corresponding to the MOM hgrid
        lon_name: str, optional
            The name of the longitude variable. Default is "x"
        lat_name: str, optional
            The name of the latitude variable. Default is "y"
        area_name: str, optional
            The name of the area variable if one exists. Default is "area"
        """

        # Set default values for lon_name, lat_name and area_name
        lon_name = lon_name or "x"
        lat_name = lat_name or "y"
        area_name = area_name or "area"

        grid = xr.open_dataset(hgrid_filename)
        inputs = [hgrid_filename]

        if mask_filename:
            mask = xr.open_dataset(mask_filename).mask.values.flatten()
            inputs += [mask_filename]
        else:
            mask = None

        # sum areas in elements
        area = grid[area_name].values
        area = (
            area[::2, ::2] + area[1::2, ::2] + area[1::2, 1::2] + area[::2, 1::2]
        ).flatten()

        x = grid[lon_name].values
        y = grid[lat_name].values

        # prep x corners
        ll = x[:-2:2, :-2:2]
        lr = x[:-2:2, 2::2]
        ul = x[2::2, :-2:2]
        ur = x[2::2, 2::2]
        x_corners = np.stack(
            (ll.flatten(), lr.flatten(), ur.flatten(), ul.flatten()), axis=1
        )
        x_centres = x[1:-1:2, 1:-1:2].flatten()

        # prep y corners
        ll = y[:-2:2, :-2:2]
        lr = y[:-2:2, 2::2]
        ul = y[2::2, :-2:2]
        ur = y[2::2, 2::2]
        y_corners = np.stack(
            (ll.flatten(), lr.flatten(), ur.flatten(), ul.flatten()), axis=1
        )
        y_centres = y[1:-1:2, 1:-1:2].flatten()

        super().__init__(
            x_centres=x_centres,
            y_centres=y_centres,
            x_corners=x_corners,
            y_corners=y_corners,
            area=area,
            mask=mask,
            inputs=inputs,
        )


class LatLonGrid(BaseGrid):

    def __init__(
        self,
        grid_filename,
        mask_filename=None,
        lon_name=None,
        lat_name=None,
        area_name=None,
    ):
        """
        Initialise a mesh representation from lat/lon locations

        Parameters
        ----------
        grid_filename: str
            Path to a netcdf file containing a lat/lon grid
        mask_filename: str, optional
            Path to a netcdf file containing a mask corresponding to the lat/lon grid
        lon_name: str, optional
            The name of the longitude variable. An attempt will be made to guess the name if not passed.
        lat_name: str, optional
            The name of the latitude variable. An attempt will be made to guess the name if not passed.
        area_name: str, optional
            The name of the area variable if one exists. An attempt will be made to guess the name if not passed.
        """

        grid = xr.open_dataset(grid_filename, chunks=-1)
        inputs = [grid_filename]

        longitude, longitude_bounds = _get_longitude(grid, lon_name)
        latitude, latitude_bounds = _get_latitude(grid, lat_name)

        if mask_filename:
            mask = xr.open_dataset(mask_filename).values.flatten()
            inputs += [mask_filename]
        else:
            mask = None

        if area_name in grid:
            area = _get_area(grid, area_name)
        else:
            area = None

        x_centres = longitude
        y_centres = latitude
        # flip and concat for ll, lr, ur, ul
        x_corners = np.concatenate(
            [longitude_bounds, longitude_bounds[..., ::-1]], axis=-1
        )
        # repeat for ll, lr, ur, ul
        y_corners = np.repeat(latitude_bounds, 2, axis=1)

        # broadcast corners
        x_corners, y_corners = np.broadcast_arrays(
            np.expand_dims(x_corners, axis=0), np.expand_dims(y_corners, axis=1)
        )
        x_corners = x_corners.reshape(-1, 4)
        y_corners = y_corners.reshape(-1, 4)

        # broadcast centres
        x_centres, y_centres = np.broadcast_arrays(
            np.expand_dims(x_centres, axis=0), np.expand_dims(y_centres, axis=1)
        )
        x_centres = x_centres.flatten()
        y_centres = y_centres.flatten()

        super().__init__(
            x_centres=x_centres,
            y_centres=y_centres,
            x_corners=x_corners,
            y_corners=y_corners,
            area=area,
            mask=mask,
            inputs=inputs,
        )


def _get_longitude(ds, lon_name):
    """
    Return the longitude variable and bounds
    """
    if lon_name:
        longitude = ds[lon_name]
    else:
        try:
            longitude = ds.cf["longitude"]
        except KeyError:
            if "lon" in ds:
                longitude = ds["lon"]
            else:
                raise KeyError(
                    "Cannot automatically determine the name of the longitude variable. Please pass --lon-name"
                )

    # Get/calculate the bounds
    if "longitude" in ds.cf.bounds:
        longitude_bounds = ds.cf.get_bounds("longitude")
    else:
        ds_bounds = ds.cf.add_bounds(longitude.name)
        longitude_bounds = ds_bounds.cf.get_bounds(longitude.name)

    return longitude.values, longitude_bounds.values


def _get_latitude(ds, lon_name):
    """
    Return the latitude variable and bounds
    """
    if lon_name:
        latitude = ds[lon_name]
    else:
        try:
            latitude = ds.cf["latitude"]
        except KeyError:
            if "lon" in ds:
                latitude = ds["lat"]
            else:
                raise KeyError(
                    "Cannot automatically determine the name of the latitude variable. Please pass --lon-name"
                )

    # Get/calculate the bounds
    if "latitude" in ds.cf.bounds:
        latitude_bounds = ds.cf.get_bounds("latitude")
    else:
        ds_bounds = ds.cf.add_bounds(latitude.name)
        latitude_bounds = ds_bounds.cf.get_bounds(latitude.name)

    return latitude.values, latitude_bounds.values


def _get_area(ds, area_name):
    """
    Return the area variable
    """
    if area_name:
        return ds[area_name]
    else:
        try:
            return ds.cf["area"]
        except KeyError:
            raise KeyError(
                "Cannot automatically determine the name of the area variable. Please pass --area-name"
            )


gridtype_dispatch = {
    "latlon": LatLonGrid,
    "mom": MomSuperGrid,
}


def main():
    parser = argparse.ArgumentParser(
        description="Create an ESMF mesh file from a grid in a netcdf file."
    )

    parser.add_argument(
        "--grid-type",
        choices=gridtype_dispatch.keys(),
        required=True,
        help="The type of grid in the netcdf file.",
    )
    parser.add_argument(
        "--wrap-lons",
        default=False,
        action="store_true",
        help="Wrap longitude values into the range between 0 and 360.",
    )
    parser.add_argument(
        "--grid-filename",
        type=str,
        required=True,
        help="The path to the netcdf file specifying the grid.",
    )
    parser.add_argument(
        "--mask-filename",
        type=str,
        default=None,
        help="The path to a netcdf file specifying the mask. If not passed, no mask is set in the mesh.",
    )
    parser.add_argument(
        "--mesh-filename",
        type=str,
        required=True,
        help="The path to the mesh file to create.",
    )
    parser.add_argument(
        "--lon-name",
        type=str,
        default=None,
        help="The name of the longitude variable in the input grid. If not passed, an attempt will be made to guess the name.",
    )
    parser.add_argument(
        "--lat-name",
        type=str,
        default=None,
        help="The name of the latitude variable in the input grid. If not passed, an attempt will be made to guess the name.",
    )
    parser.add_argument(
        "--area-name",
        type=str,
        default=None,
        help="The name of the area variable in the input grid. If not passed, an attempt will be made to guess the name.",
    )

    args = parser.parse_args()
    grid_type = args.grid_type
    wrap_lons = args.wrap_lons
    grid_filename = os.path.abspath(args.grid_filename)
    mesh_filename = os.path.abspath(args.mesh_filename)
    mask_filename = args.mask_filename
    lon_name = args.lon_name
    lat_name = args.lat_name
    area_name = args.area_name

    if mask_filename:
        mask_filename = os.path.abspath(mask_filename)

    this_file = os.path.normpath(__file__)

    # Add some info about how the file was generated
    runcmd = (
        f"python3 {os.path.basename(this_file)} --grid-type={grid_type} --grid-filename={grid_filename} "
        f"--mesh-filename={mesh_filename}"
    )
    if mask_filename:
        runcmd += f" --mask-filename={mask_filename}"
    if wrap_lons:
        runcmd += f" --wrap-lons"
    if lon_name:
        runcmd += f" --lon-name={lon_name}"
    if lat_name:
        runcmd += f" --lat-name={lat_name}"
    if area_name:
        runcmd += f" --area-name={area_name}"

    global_attrs = {"history": get_provenance_metadata(this_file, runcmd)}

    mesh = gridtype_dispatch[grid_type](
        grid_filename, mask_filename, lon_name, lat_name, area_name
    )

    mesh.create_mesh(wrap_lons=wrap_lons, global_attrs=global_attrs).write(
        mesh_filename
    )


if __name__ == "__main__":
    import argparse

    main()
