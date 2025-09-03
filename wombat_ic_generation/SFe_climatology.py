# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Calculate the climatological average of the SFe variable from
# CESM-MIMI_1980-2015_CAM4-6MEAN_MonthlyDep_Hamiltonetal2020.nc (https://doi.org/10.7298/xqqj-qk90)
#
# To run:
#   python SFe_climatology.py --input-filename=<path-to-input-file>
#      --output-filename=<path-to-output-file>
#
# For more information, run `python SFe_climatology.py -h`
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
#   argparse, xarray
# =========================================================================================

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum

xr.set_options(keep_attrs=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the climatological average of the SFe variable from "
            "CESM-MIMI_1980-2015_CAM4-6MEAN_MonthlyDep_Hamiltonetal2020.nc (https://doi.org/10.7298/xqqj-qk90)"
        )
    )

    parser.add_argument(
        "--input-filename",
        type=str,
        required=True,
        help="The path to the input file.",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        required=True,
        help="The path to the file to be outputted.",
    )

    args = parser.parse_args()
    input_filename = os.path.abspath(args.input_filename)
    output_filename = os.path.abspath(args.output_filename)

    this_file = os.path.normpath(__file__)

    # Add some info about how the file was generated
    runcmd = (
        f"python3 {os.path.basename(this_file)} --input-filename={input_filename} "
        f"--output-filename={output_filename}"
    )

    history_attrs = {
        "history": get_provenance_metadata(this_file, runcmd),
        "inputFile": f"{input_filename} (md5 hash: {md5sum(input_filename)})",
    }

    # Load the input data
    ds = xr.open_dataset(input_filename).compute()

    # Convert from kg/m2/s to mol/m2/s
    g_per_kg = 1000
    g_per_mol = 55.845
    SFe = g_per_kg / g_per_mol * ds[["SFe"]]
    SFe.SFe.attrs["unit"] = "mol/m2/s"

    # Create time array at middle of month
    # FMS requires calendar to be one of: noleap, 365_day, 365_days, 360_day,
    # julian, no_calendar, thirty_day_months, gregorian
    calendar = "gregorian"
    times = xr.date_range(
        start="0001",
        periods=13,
        freq="MS",
        calendar=calendar,
        use_cftime=True,
    )
    times = times[:-1] + times.diff()[1:] / 2  # Middle of the month

    # Average all years (1980-2014)
    SFe_clim = SFe.mean("yrs").rename({"mon": "time"}).assign_coords({"time": times})
    SFe_clim.SFe.attrs["cell_methods"] = (
        "time: mean within months time: mean over years"
    )

    # Update attributes
    SFe_clim.attrs["Title"] = (
        "Monthly climatological (1980-2014) soluble iron and dust deposition"
    )
    SFe_clim.attrs |= {
        "DOI": "https://doi.org/10.7298/xqqj-qk90 (file: CESM-MIMI_1980-2015_CAM4-6MEAN_MonthlyDep_Hamiltonetal2020.nc)",
    }
    SFe_clim.attrs |= history_attrs

    # Add climatology_bounds
    SFe_clim.time.attrs["climatology"] = "climatology_bounds"
    start_times = xr.date_range(
        start=str(int(SFe.yrs.min().item())),
        periods=12,
        freq="MS",
        calendar=calendar,
        use_cftime=True,
    )
    end_times = xr.date_range(
        start=str(int(SFe.yrs.max().item())),
        periods=13,
        freq="MS",
        calendar=calendar,
        use_cftime=True,
    )[1:]
    climatology_bounds = xr.DataArray(
        np.vstack((start_times, end_times)), dims=["nv", "time"]
    )
    SFe_clim = SFe_clim.assign_coords({"climatology_bounds": climatology_bounds})

    comp = dict(zlib=True, complevel=4)
    encoding = {var: comp for var in SFe_clim.data_vars}
    encoding |= {
        "time": {
            "units": "days since 0001-01-01 00:00:00.000000",
            "calendar": calendar,
        },
        "climatology_bounds": {
            "units": "days since 0001-01-01 00:00:00.000000",
            "calendar": calendar,
        },
    }
    unlimited_dims = "time" if "time" in SFe_clim.dims else None
    SFe_clim.to_netcdf(
        output_filename, unlimited_dims=unlimited_dims, encoding=encoding
    )


if __name__ == "__main__":
    import argparse

    main()
