#!/usr/bin/env python3
# Copyright ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""
This script combines WOA23 monthly (upper-ocean, above 1500m) and seasonal
(whole-depth) temperature and salinity climatologies, and uses TEOS-10 (via the
GSW library) to compute absolute salinity, conservative temperature and
potential temperature. Data is on the native WOA23 grid.

To run:
    python make_woa23_ts_initial_conditions.py --output-dir=<path-to-output-dir>

For more information, run `python make_woa23_ts_initial_conditions.py -h`

The run command and full github url of the current version of this script is added to
the metadata of the generated files. This is to uniquely identify the script and inputs
used to generate them. To produce output for sharing, ensure you are using a version of
this script which is committed and pushed to github.

Dependencies:
    argparse, xarray, gsw, python-dateutil
"""

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path
from glob import glob

import gsw
import xarray as xr
from dateutil.relativedelta import relativedelta

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata

WOA23_PATH = "/g/data/av17/access-nri/OM3/woa23"

# WOA23 uses a strange "months since" calendar unit, so there's a few functions
# to convert to "days since" and handle the calendar


def _day_of_midmonth(month):
    """
    Day-of-year of the middle of `month`, on a climatological (non-leap) 365-day
    year. An arbitrary non-leap year is used only to get correct month lengths.
    """
    year = 2001
    start_of_year = dt.date(year, 1, 1)
    start_of_month = dt.date(year, month, 1)
    start_of_next_month = dt.date(year + (month == 12), month % 12 + 1, 1)
    days_in_month = (start_of_next_month - start_of_month).days
    return (start_of_month - start_of_year).days + days_in_month / 2


def _parse_time_origin(units):
    """
    Parse the reference date out of a CF time `units` string, e.g.
    "months since 1955-01-01 00:00:00".
    """
    match = re.search(r"\d{4}-\d{1,2}-\d{1,2}", units)
    if match is None:
        raise ValueError(f"Could not parse a reference date from time units: {units!r}")
    return dt.datetime.strptime(match.group(0), "%Y-%m-%d").date()


def _climatology_bounds_in_days(bounds_months, time_origin):
    """
    Convert a pair of climatology_bounds values (in months since time_origin) to
    days since time_origin.
    """
    bounds_dates = [time_origin + relativedelta(months=int(m)) for m in bounds_months]
    return [(date - time_origin).days for date in bounds_dates]


class _MonthlyWoaTS:
    """
    Loads, converts and builds the combined WOA23 temperature/salinity initial
    condition dataset for one month, kept lazy/dask-backed throughout so that
    save_all() can compute and write every month together in one pass. Use as:
    _MonthlyWoaTS(month, output_dir).load().convert().to_dataset()
    """

    def __init__(self, month, output_dir):
        self.month = month
        # WOA23 file-name suffixes: 01-12 are monthly means, 13-16 are seasonal
        self.season = str(13 + (month - 1) // 3)
        self.mon_str = f"{month:02d}"
        self.output_file = f"{output_dir}/woa23_decav_ts_{self.mon_str}_04.nc"

    def load(self):
        """
        Open the source files and combine upper-ocean monthly data with deep-ocean
        seasonal data below 1500m.
        """
        upper_files = glob(f"{WOA23_PATH}/woa23_decav_[st]{self.mon_str}_04.nc")
        deep_files = glob(f"{WOA23_PATH}/woa23_decav_[st]{self.season}_04.nc")

        self.source_files = upper_files + deep_files

        print(
            f"Processing month {self.mon_str} (deep ocean filled from season {self.season})"
        )

        upper_ds = xr.open_mfdataset(upper_files, decode_times=False)
        deep_ds = xr.open_mfdataset(deep_files, decode_times=False)

        # Dask arrays can't be assigned into positionally like numpy arrays, so the
        # upper/deep depth ranges are joined with a concat instead, on both t_an and
        # s_an at once.
        n_upper = upper_ds.sizes["depth"]
        combined = xr.concat(
            [
                upper_ds[["t_an", "s_an"]].isel(time=0, drop=True),
                deep_ds[["t_an", "s_an"]].isel(
                    time=0, drop=True, depth=slice(n_upper, None)
                ),
            ],
            dim="depth",
        )
        self.t_in_situ = combined["t_an"]
        self.s_practical = combined["s_an"]

        self.time_origin = _parse_time_origin(upper_ds["time"].attrs["units"])
        self.climatology_bounds = _climatology_bounds_in_days(
            deep_ds["climatology_bounds"].isel(time=0).values, self.time_origin
        )
        return self

    def convert(self):
        """
        Compute absolute salinity, conservative temperature and potential
        temperature from the combined in-situ temperature/practical salinity,
        using TEOS-10, and assemble them into a Dataset with CF attributes.
        """
        depth = self.t_in_situ["depth"]
        lat = self.t_in_situ["lat"]
        lon = self.t_in_situ["lon"]

        pressure = gsw.p_from_z(-depth, lat)

        s_absolute = gsw.SA_from_SP(self.s_practical, pressure, lon, lat)
        t_conservative = gsw.CT_from_t(s_absolute, self.t_in_situ, pressure)
        t_potential = gsw.pt0_from_t(s_absolute, self.t_in_situ, pressure)

        self.ds = self.s_practical.to_dataset(name="practical_salinity")
        # set attributes for new variables
        self.ds["absolute_salinity"] = s_absolute.assign_attrs(
            standard_name="sea_water_absolute_salinity",
            long_name="absolute salinity calculated using teos10 from practical salinity",
            units="g kg-1",
            cell_methods=self.s_practical.attrs["cell_methods"],
        )
        self.ds["conservative_temperature"] = t_conservative.assign_attrs(
            standard_name="sea_water_conservative_temperature",
            long_name=(
                "conservative temperature calculated using teos10 from objectively "
                "analysed mean fields for sea_water_temperature"
            ),
            units="degrees celsius",
            cell_methods=self.t_in_situ.attrs["cell_methods"],
        )
        self.ds["potential_temperature"] = t_potential.assign_attrs(
            standard_name="sea_water_potential_temperature",
            long_name=(
                "potential temperature with reference pressure, p_ref = 0 dbar calculated "
                "using teos10 from objectively analysed mean fields for sea_water_temperature"
            ),
            units="degrees celsius",
            cell_methods=self.t_in_situ.attrs["cell_methods"],
        )
        return self

    def to_dataset(self):
        """Assemble the output Dataset, with CF metadata, provenance and encoding attached."""
        time_value = _day_of_midmonth(self.month)

        # fix up time dimension
        ds_out = self.ds.expand_dims(time=[time_value])
        ds_out["climatology_bounds"] = xr.DataArray(
            [self.climatology_bounds],
            dims=("time", "nbounds"),
            coords={"time": [time_value]},
        )

        ds_out["time"].attrs = {
            "standard_name": "time",
            "long_name": "time",
            "axis": "T",
            "climatology": "climatology_bounds",
            "units": f"days since {self.time_origin.isoformat()} 00:00:00",
        }

        ds_out.attrs = get_provenance_metadata(self.source_files)
        ds_out.attrs["Conventions"] = "CF-1.10"
        ds_out.attrs["title"] = (
            "WOA23-derived temperature and salinity fields with conservative temperature"
        )
        ds_out.attrs["summary"] = (
            "Conservative temperature computed from in-situ temperature and practical salinity "
            "using TEOS-10 via the GSW library"
        )

        base_encoding = {"zlib": True, "complevel": 4, "dtype": "float32"}
        fill_value_by_var = {
            "practical_salinity": self.s_practical.encoding["_FillValue"],
            "absolute_salinity": self.s_practical.encoding["_FillValue"],
            "conservative_temperature": self.t_in_situ.encoding["_FillValue"],
            "potential_temperature": self.t_in_situ.encoding["_FillValue"],
        }
        for var, fill_value in fill_value_by_var.items():
            ds_out[var].encoding |= base_encoding | {"_FillValue": fill_value}
        for coord in ["time", "depth", "lat", "lon"]:
            ds_out[coord].encoding["_FillValue"] = None

        return ds_out


def save_all(output_dir):
    """
    Build every month's dataset lazily, then compute and write them all in a single
    dask-scheduled pass (rather than looping through months one at a time).
    """
    months = [
        _MonthlyWoaTS(month, output_dir).load().convert() for month in range(1, 13)
    ]
    datasets = [m.to_dataset() for m in months]
    paths = [m.output_file for m in months]

    print(f"Writing {len(paths)} files to {output_dir}")
    xr.save_mfdataset(datasets, paths, unlimited_dims="time")
    print("Done")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine WOA23 monthly (upper-ocean) and seasonal (deep-ocean) temperature "
            "and salinity climatologies, and compute absolute salinity, conservative "
            "temperature and potential temperature using TEOS-10 (via the GSW library)."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the monthly output files to.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_all(args.output_dir)


if __name__ == "__main__":
    main()
