# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Create CO2 forcing file for IAF configurations using WOMBAT from global mean annual CO2 data.
# The script uses CMIP7 CO2 gm forcing data (https://doi.org/10.5281/zenodo.14892947) for the
# historical period up to 2022, then extends this using NOAA global mean data
# (https://doi.org/10.15138/9N0H-ZH07) for 2023 onward.
#
# To run:
#   python co2_iaf.py --co2-cmip-filename=<path-to-cmip-co2-file> \
#       --co2-noaa-filename=<path-to-noaa-co2-file> \
#       --hgrid-filename=<path-to-supergrid-file> --output-filename=<path-to-output-file>
#
# For more information, run `python co2_iaf.py -h`
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
#   argparse, xarray, numpy, pandas
# =========================================================================================

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum

xr.set_options(keep_attrs=True)


def change_year(cftime_obj, new_year):
    return cftime_obj.replace(year=new_year)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Broadcast a 1D timeseries global-average CO2 concentration onto a provided grid."
        )
    )

    parser.add_argument(
        "--co2-cmip-filename",
        type=str,
        required=True,
        help="Path to NetCDF file containing the CMIP7 CO2 gm forcing data (https://doi.org/10.5281/zenodo.14892947).",
    )

    parser.add_argument(
        "--co2-noaa-filename",
        type=str,
        required=True,
        help="Path to text file containing the NOAA global mean CO2 data (https://doi.org/10.15138/9N0H-ZH07).",
    )

    parser.add_argument(
        "--hgrid-filename",
        type=str,
        required=True,
        help="The path to the MOM supergrid file to use as the target grid.",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        required=True,
        help="The path to the file to be outputted.",
    )

    args = parser.parse_args()

    co2_cmip_filename = os.path.abspath(args.co2_cmip_filename)
    co2_noaa_filename = os.path.abspath(args.co2_noaa_filename)
    hgrid_filename = os.path.abspath(args.hgrid_filename)
    output_filename = os.path.abspath(args.output_filename)

    this_file = os.path.normpath(__file__)

    # provenance metadata
    runcmd = (
        f"python3 {os.path.basename(this_file)} --co2-cmip-filename={co2_cmip_filename} "
        f"--co2-noaa-filename={co2_noaa_filename} --hgrid-filename={hgrid_filename} "
        f"--output-filename={output_filename}"
    )

    file_hashes = [
        f"{co2_cmip_filename} (md5 hash: {md5sum(co2_cmip_filename)})",
        f"{co2_noaa_filename} (md5 hash: {md5sum(co2_noaa_filename)})",
        f"{hgrid_filename} (md5 hash: {md5sum(hgrid_filename)})",
    ]

    global_attrs = {
        "history": get_provenance_metadata(this_file, runcmd),
        "inputFile": ", ".join(file_hashes),
    }

    # Load the input data
    co2_cmip = xr.open_dataset(co2_cmip_filename, decode_cf=False).compute()
    hgrid = xr.open_dataset(hgrid_filename).compute()
    co2_noaa = pd.read_csv(
        co2_noaa_filename,
        comment="#",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["year", "conc"],
    )

    # Reset calendar from proleptic_gregorian to gregorian because FMS doesn't support the former.
    # They are equivalent for dates after 1582.
    co2_cmip.time.attrs["calendar"] = "gregorian"
    co2_cmip.time_bnds.attrs["calendar"] = "gregorian"
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    co2_cmip = xr.decode_cf(co2_cmip, decode_times=time_coder)

    # Destination grid is tracer cell centres
    lon_dest = hgrid["x"][1:-1:2, 1:-1:2].to_dataset(name="lon")
    lat_dest = hgrid["y"][1:-1:2, 1:-1:2].to_dataset(name="lat")
    grid_dest = xr.merge((lon_dest, lat_dest)).set_coords(["lon", "lat"])

    # Broadcast the CO2 data onto the destination grid
    co2_bc = xr.merge(xr.broadcast(co2_cmip["co2"], grid_dest))
    co2_cmip = xr.merge((co2_cmip.drop_vars("co2"), co2_bc))
    co2_cmip = co2_cmip.set_coords("time_bnds")

    # Extend the CO2 data using NOAA data
    template = co2_cmip.isel(time=0)
    co2_noaa_extend = co2_noaa[co2_noaa["year"] > 2022]
    co2_extend = []
    for _, row in co2_noaa_extend.iterrows():
        time = change_year(template.time.item(), row["year"])
        time_bnds = np.array(
            [
                change_year(template.time_bnds.values[0], row["year"]),
                change_year(template.time_bnds.values[1], row["year"] + 1),
            ]
        )
        co2 = row["conc"] + 0 * template
        co2_extend.append(
            co2.assign_coords({"time": time, "time_bnds": (("bnds",), time_bnds)})
        )

    co2_extend.insert(0, co2_cmip)
    co2_cmip = xr.concat(co2_extend, dim="time")

    # Add coodinates and metadata required by data_table
    co2_cmip = co2_cmip.rename({"nyp": "ny", "nxp": "nx"})
    co2_cmip = co2_cmip.assign_coords(
        {
            "ny": ("ny", range(co2_cmip.sizes["ny"])),
            "nx": ("nx", range(co2_cmip.sizes["nx"])),
        }
    )
    # Both axis and cartesian_axis attributes are required to work with recent (MOM6-era) and older
    # (MOM5-era) versions of FMS
    co2_cmip["ny"].attrs = dict(axis="Y", cartesian_axis="Y")
    co2_cmip["nx"].attrs = dict(axis="X", cartesian_axis="X")
    co2_cmip["lat"].attrs = dict(
        long_name="Latitude of T-cell center",
        standard_name="latitude",
        units="degree_north",
    )
    co2_cmip["lon"].attrs = dict(
        long_name="Longitude of T-cell center",
        standard_name="longitude",
        units="degrees_east",
    )
    co2_cmip.attrs = co2_cmip.attrs | global_attrs
    co2_cmip.attrs["comment"] = (
        co2_cmip.attrs.get("comment")
        + " Extended beyond 2022 by ACCESS-NRI using NOAA data (https://doi.org/10.15138/9N0H-ZH07)."
    )

    # Save output
    # FMS doesn't like negative times
    units = "days since 1750-01-01"
    co2_cmip["time"].encoding |= {"units": units}
    co2_cmip["time_bnds"].encoding |= {"units": units}
    # _FillValue is required by older (MOM5-era) versions of FMS
    var_encoding = dict(zlib=True, complevel=4, _FillValue=-1.0e36)
    for var in co2_cmip.data_vars:
        co2_cmip[var].encoding |= var_encoding
    # Coordinates should not have _FillValue
    coord_encoding = dict(_FillValue=None)
    for coord in co2_cmip.coords:
        co2_cmip[coord].encoding |= coord_encoding
    # Older (MOM5-era) versions of FMS can't handle integer type dimensions
    co2_cmip["nx"].encoding |= {"dtype": "float32"}
    co2_cmip["ny"].encoding |= {"dtype": "float32"}
    unlimited_dims = "time" if "time" in co2_cmip.dims else None
    co2_cmip.to_netcdf(output_filename, unlimited_dims=unlimited_dims)


if __name__ == "__main__":
    main()
