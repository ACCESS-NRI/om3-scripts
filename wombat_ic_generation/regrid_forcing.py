# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Interpolate a provided forcing file to a grid specified by a provided MOM supergrid file for
# use in the MOM data_table.
#
# To run:
#   python regrid_forcing.py --forcing-filename=<path-to-forcing-file>
#     --hgrid-filename=<path-to-supergrid-file> --output-filename=<path-to-output-file>
# these extra arguments are available if required:
#     --mask-filename=<path-to-mask-file>--lon-name=<lon-name> --lat-name=<lat-name>
#
# For more information, run `python regrid_forcing.py -h`
#
# The run command and full github url of the current version of this script is added to the
# metadata of the generated file. This is to uniquely identify the script and inputs used to
# generate the file. To produce files for sharing, ensure you are using a version of this script
# which is committed and pushed to github. For files intended for released configurations, use the
# latest version checked in to the main branch of the github repository.
#
# Contact:
#   Dougie Squire <dougie.squire@anu.edu.au>
#
# Dependencies:
#   argparse, xarray, cf_xarray, xesmf
# =========================================================================================

import os
import sys
from pathlib import Path

import xarray as xr
import xesmf as xe

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate a provided forcing file to a grid specified by a provided MOM supergrid"
            "file for use in the MOM data_table."
        )
    )

    parser.add_argument(
        "--forcing-filename",
        type=str,
        required=True,
        help="The path to the forcing file to interpolate.",
    )

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
    forcing_filename = os.path.abspath(args.forcing_filename)
    hgrid_filename = os.path.abspath(args.hgrid_filename)
    output_filename = os.path.abspath(args.output_filename)
    mask_filename = args.mask_filename
    lon_name = args.lon_name
    lat_name = args.lat_name

    if mask_filename:
        mask_filename = os.path.abspath(mask_filename)

    this_file = os.path.normpath(__file__)

    # Add some info about how the file was generated
    runcmd = (
        f"python3 {os.path.basename(this_file)} --forcing-filename={forcing_filename} "
        f"--hgrid-filename={hgrid_filename} --output-filename={output_filename}"
    )
    if mask_filename:
        runcmd += f" --mask-filename={mask_filename}"
    if lon_name:
        runcmd += f" --lon-name={lon_name}"
    if lat_name:
        runcmd += f" --lat-name={lat_name}"

    file_hashes = [
        f"{forcing_filename} (md5 hash: {md5sum(forcing_filename)})",
        f"{hgrid_filename} (md5 hash: {md5sum(hgrid_filename)})",
    ]
    if mask_filename:
        file_hashes.append(f"{mask_filename} (md5 hash: {md5sum(mask_filename)})")
    global_attrs = {
        "history": get_provenance_metadata(this_file, runcmd),
        "inputFile": ", ".join(file_hashes),
    }

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
    lon_dest = hgrid["x"][1:-1:2, 1:-1:2].to_dataset(name="lon")
    lat_dest = hgrid["y"][1:-1:2, 1:-1:2].to_dataset(name="lat")
    grid_dest = xr.merge((lon_dest, lat_dest))

    # Regrid using bilinear interpolation with nearest neighbour extrapolation
    # NOTE: This will not conserve global quantities
    regridder = xe.Regridder(
        grid_src,
        grid_dest,
        method="bilinear",
        extrap_method="nearest_s2d",
        periodic=True,
    )
    forcing_regrid = regridder(forcing_src, keep_attrs=True)

    # Add coodinates and metadata required by data_table
    forcing_regrid = forcing_regrid.assign_coords(
        lon=lon_dest["lon"], lat=lat_dest["lat"]
    )
    forcing_regrid = forcing_regrid.rename({"nyp": "ny", "nxp": "nx"})
    forcing_regrid = forcing_regrid.assign_coords(
        {
            "ny": ("ny", range(forcing_regrid.sizes["ny"])),
            "nx": ("nx", range(forcing_regrid.sizes["nx"])),
        }
    )
    forcing_regrid["time"].attrs = dict(axis="T", standard_name="time", modulo="y")
    # Both axis and cartesian_axis attributes are required to work with recent (MOM6-era) and older
    # (MOM5-era) versions of FMS
    forcing_regrid["ny"].attrs = dict(axis="Y", cartesian_axis="Y")
    forcing_regrid["nx"].attrs = dict(axis="X", cartesian_axis="X")
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
    forcing_regrid.attrs = forcing_regrid.attrs | global_attrs

    # Save output
    # _FillValue is required by older (MOM5-era) versions of FMS
    var_encoding = dict(zlib=True, complevel=4, _FillValue=-1.0e36)
    for var in forcing_regrid.data_vars:
        forcing_regrid[var].encoding |= var_encoding
    # Coordinates should not have _FillValue
    coord_encoding = dict(_FillValue=None)
    for coord in forcing_regrid.coords:
        forcing_regrid[coord].encoding |= coord_encoding
    # Older (MOM5-era) versions of FMS can't handle integer type dimensions
    forcing_regrid["nx"].encoding |= {"dtype": "float32"}
    forcing_regrid["ny"].encoding |= {"dtype": "float32"}
    unlimited_dims = "time" if "time" in forcing_regrid.dims else None
    forcing_regrid.to_netcdf(output_filename, unlimited_dims=unlimited_dims)


if __name__ == "__main__":
    import argparse

    main()
