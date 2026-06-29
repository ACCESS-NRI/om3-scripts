# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Calculate the climatological average of GlobColour monthly CHL (10.48670/moi-00281) and fill
# missing regions using harmonic interpolation.
#
# This script take approximately 30 minutes to run on a single Sapphire Rapids node.
#
# To run:
#   python chl_climatology_and_fill.py --input-directory=<path-to-input-directory>
#      --output-filename=<path-to-output-file>
#
# For more information, run `python chl_climatology_and_fill.py -h`
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
#   argparse, xarray, scipy, numpy, regionmask, dask
# =========================================================================================

import os
import sys
import glob
from pathlib import Path

import numpy as np
import xarray as xr
from distributed import Client

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from regrid_common import fill_ocean_horiz
from scripts_common import get_provenance_metadata

xr.set_options(keep_attrs=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the monthly climatological average of GlobColour CHL (10.48670/moi-00281) "
            "and fill missing regions using harmonic interpolation."
        )
    )

    parser.add_argument(
        "--input-directory",
        type=str,
        required=True,
        help="The path directory containing the monthly GlobColour CHL files.",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        required=True,
        help="The path to the file to be outputted.",
    )

    args = parser.parse_args()
    input_directory = os.path.abspath(args.input_directory)
    output_filename = os.path.abspath(args.output_filename)

    # Obtain metadata
    this_file = sys.argv[0]
    runcmd = f"{sys.executable} {' '.join(sys.argv)}"

    history_attrs = {
        "history": get_provenance_metadata(this_file, runcmd),
    }

    # Load the input data and compute the monthly climatology
    input_files = sorted(glob.glob(f"{input_directory}/*.nc"))

    print("Calculating the monthly climatology...")

    with Client(threads_per_worker=1):
        ds = xr.open_mfdataset(
            input_files,
            chunks={"lat": 1024, "lon": 1024},
            parallel=True,
            use_cftime=True,
        )
        chl = ds[["CHL"]].groupby("time.month").mean("time").compute()

    # Fill missing data for each month
    print("Filling missing data...")
    chl_filled = []
    for month in range(1, 13):
        print(f"  Filling month {month}...")
        chl_filled.append(
            fill_ocean_horiz(
                chl["CHL"].sel(month=month),
                top_bound="regular",
                n_erode=250,
                erode_first=False,
            )
        )

    chl["CHL"] = xr.concat(chl_filled, dim="month")

    # Add time array
    calendar = "gregorian"
    times = xr.date_range(
        start=str(int(ds.time.dt.year.median())),
        periods=13,
        freq="MS",
        calendar=calendar,
        use_cftime=True,
    )
    times = times[:-1] + times.diff()[1:] / 2  # Middle of the month
    chl = chl.rename({"month": "time"}).assign_coords({"time": times})

    # Add climatology_bounds
    chl.CHL.attrs["cell_methods"] = "time: mean within years time: mean over years"
    chl.time.attrs["climatology"] = "climatology_bounds"
    start_times = xr.date_range(
        start=ds.time.min().item(),
        periods=12,
        freq="MS",
        calendar=calendar,
        use_cftime=True,
    )
    end_times = xr.date_range(
        end=ds.time.max().item(),
        periods=12,
        freq="MS",
        calendar=calendar,
        use_cftime=True,
    ).shift(1, "MS")
    climatology_bounds = xr.DataArray(
        np.vstack(
            (
                start_times.sort_values(key=lambda x: x.month),
                end_times.sort_values(key=lambda x: x.month),
            )
        ).T,
        dims=["time", "nv"],
    )
    chl = chl.assign_coords({"climatology_bounds": climatology_bounds})

    # Add attrs and save
    del chl.CHL.attrs["ancillary_variables"]
    chl.attrs["title"] = (
        "Global ocean surface Chlorophyll-a concentration filled monthly climatology"
    )
    chl.time.attrs["long_name"] = "Time"
    chl.time.attrs["standard_name"] = "time"
    chl.time.attrs["axis"] = "T"
    chl.attrs |= history_attrs
    comp = dict(zlib=True, complevel=4)
    encoding = {var: comp for var in chl.data_vars}
    # Time coords should be double type according for CF conventions
    time_encoding = {
        "dtype": "float64",
        "units": "days since 0001-01-01 00:00:00.000000",
        "calendar": calendar,
        "_FillValue": None,
    }
    encoding |= {
        "time": time_encoding,
        "climatology_bounds": time_encoding,
        "lat": {"_FillValue": None},
        "lon": {"_FillValue": None},
    }

    chl.to_netcdf(output_filename, unlimited_dims="time", encoding=encoding)


if __name__ == "__main__":
    import argparse

    main()
