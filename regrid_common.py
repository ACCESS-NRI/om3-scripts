# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# These are common functions/classes which assist with regridding
# =========================================================================================

import os
import xesmf as xe
import xarray as xr
import argparse

from scripts_common import md5sum


def _guess_longitude_name(ds):
    coords = ds.cf.coordinates
    if "longitude" in coords:
        if len(coords["longitude"]) == 1:
            return coords["longitude"][0]
        else:
            raise KeyError("Multiple longitude variables exist. Please pass --lon-name")
    elif "lon" in ds:
        return "lon"
    else:
        raise KeyError(
            "Cannot automatically determine the name of the longitude variable. Please pass --lon-name"
        )


def _guess_latitude_name(ds):
    coords = ds.cf.coordinates
    if "latitude" in coords:
        if len(coords["latitude"]) == 1:
            return coords["latitude"][0]
        else:
            raise KeyError("Multiple latitude variables exist. Please pass --lat-name")
    elif "lat" in ds:
        return "lat"
    else:
        raise KeyError(
            "Cannot automatically determine the name of the latitude variable. Please pass --lat-name"
        )


class Regrid_Common:

    def __init__(self):
        self.method = "bilinear"
        self.extrap = "nearest_s2d"
        self.periodic = True
        self.parser = argparse.ArgumentParser(
            description=(
                "Interpolate a provided forcing file to a grid specified by a provided MOM supergrid file"
            )
        )

    def parse_cli(self):

        parser = self.parser

        parser.add_argument(
            "--hgrid-filename",
            type=str,
            required=True,
            help="The path to the MOM supergrid file to use as a grid for interpolation.",
        )

        parser.add_argument(
            "--output-filename",
            type=str,
            required=True,
            help="The path to the file to be outputted.",
        )

        parser.add_argument(
            "--mask-filename",
            type=str,
            default=None,
            help=(
                "The path to a land-sea (0-1) mask for the forcing file. Cells that fall on land in "
                "the destination grid will take their values from the nearest source grid cell."
            ),
        )

        parser.add_argument(
            "--lon-name",
            type=str,
            default=None,
            help=(
                "The name of the longitude variable in the input grid. If not passed, an attempt will "
                "be made to guess the name."
            ),
        )

        parser.add_argument(
            "--lat-name",
            type=str,
            default=None,
            help=(
                "The name of the latitude variable in the input grid. If not passed, an attempt will "
                "be made to guess the name."
            ),
        )

        args = parser.parse_args()
        self.args = args
        self.hgrid_filename = os.path.abspath(args.hgrid_filename)
        self.output_filename = os.path.abspath(args.output_filename)
        self.mask_filename = args.mask_filename
        self.lon_name = args.lon_name
        self.lat_name = args.lat_name

        if self.mask_filename:
            self.mask_filename = os.path.abspath(self.mask_filename)

        # some info about how the file was generated

        runcmd_args = f"--hgrid-filename={self.hgrid_filename} --output-filename={self.output_filename}"

        if self.mask_filename:
            runcmd_args += f" --mask-filename={self.mask_filename}"
        if self.lon_name:
            runcmd_args += f" --lon-name={self.lon_name}"
        if self.lat_name:
            runcmd_args += f" --lat-name={self.lat_name}"

        self.runcmd_args = runcmd_args

    ## NOTE: it's implied that forcing_filename is set outside this class

    def open_datasets(self):

        forcing_filename = self.forcing_filename
        hgrid_filename = self.hgrid_filename
        mask_filename = self.mask_filename
        lon_name = self.lon_name
        lat_name = self.lat_name

        # Load the input data
        forcing_src = xr.open_dataset(forcing_filename).compute()
        hgrid = xr.open_dataset(hgrid_filename).compute()
        if mask_filename:
            forcing_mask = xr.open_dataset(mask_filename).compute()

        # Drop "mask" variable from forcing_src if it exists
        if "mask" in forcing_src:
            forcing_src = forcing_src.drop_vars("mask")

        # Standardise lon/lat coordinate names
        if not lon_name:
            lon_name = _guess_longitude_name(forcing_src)

        if not lat_name:
            lat_name = _guess_latitude_name(forcing_src)

        forcing_src = forcing_src.rename({lon_name: "lon", lat_name: "lat"})

        # Get source and destination grid
        grid_src = forcing_src[["lon", "lat"]]
        if mask_filename:
            # Add the mask to the source grid so that land values are extrapolated
            if "mask" in forcing_mask:
                grid_src = grid_src.assign(mask=forcing_mask["mask"])
            else:
                raise ValueError(
                    f"Input mask-filename must contain a variable named 'mask'"
                )

        # Destination grid is tracer cell centres
        # hgrid = xr.open_dataset(hgrid_path)
        hgrid_x = hgrid.x[1::2, 1::2]
        hgrid_y = hgrid.y[1::2, 1::2]
        hgrid_xc = hgrid.x[::2, ::2]
        hgrid_yc = hgrid.y[::2, ::2]
        grid_dest = xr.Dataset(
            coords={
                "lon": (("ny", "nx"), hgrid_x.values),
                "lat": (("ny", "nx"), hgrid_y.values),
                "lon_b": (("nyp", "nxp"), hgrid_xc.values),
                "lat_b": (("nyp", "nxp"), hgrid_yc.values),
            },
        )

        self.forcing_src = forcing_src
        if mask_filename:
            self.forcing_mask = forcing_mask
        self.grid_src = grid_src
        self.grid_dest = grid_dest

    def regrid_forcing(self):

        # Regrid using bilinear interpolation with nearest neighbour extrapolation
        # NOTE: This will not conserve global quantities
        regridder = xe.Regridder(
            self.grid_src,
            self.grid_dest,
            method=self.method,
            extrap_method=self.extrap,
            periodic=self.periodic,
        )
        forcing_regrid = regridder(self.forcing_src, keep_attrs=True)

        # Add coodinates (required by data_table)
        forcing_regrid = forcing_regrid.assign_coords(
            lon=self.grid_dest["lon"], lat=self.grid_dest["lat"]
        )

        forcing_regrid["lat"].attrs = dict(
            long_name="Latitude of T-cell center",
            standard_name="latitude",
            units="degree_north",
        )
        forcing_regrid["lon"].attrs = dict(
            long_name="Longitude of T-cell center",
            standard_name="longitude",
            units="degrees_east",
        )

        self.forcing_regrid = forcing_regrid

    def save_output(self):

        forcing_regrid = self.forcing_regrid

        # Info about input data used
        file_hashes = [
            f"{self.forcing_filename} (md5 hash: {md5sum(self.forcing_filename)})",
            f"{self.hgrid_filename} (md5 hash: {md5sum(self.hgrid_filename)})",
        ]
        if self.mask_filename:
            file_hashes.append(
                f"{self.mask_filename} (md5 hash: {md5sum(self.mask_filename)})"
            )

        global_attrs = {
            "inputFile": ", ".join(file_hashes),
        }
        forcing_regrid.attrs = forcing_regrid.attrs | global_attrs

        # Save output
        # _FillValue is required by older (MOM5-era) versions of FMS
        var_encoding = dict(zlib=True, complevel=4, _FillValue=-1.0e10)
        for var in forcing_regrid.data_vars:
            forcing_regrid[var].encoding |= var_encoding
        # Coordinates should not have _FillValue
        coord_encoding = dict(_FillValue=None)
        for coord in forcing_regrid.coords:
            forcing_regrid[coord].encoding |= coord_encoding

        unlimited_dims = "time" if "time" in forcing_regrid.dims else None
        forcing_regrid.to_netcdf(self.output_filename, unlimited_dims=unlimited_dims)
