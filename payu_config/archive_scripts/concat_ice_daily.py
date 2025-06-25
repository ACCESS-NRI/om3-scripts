#!python3
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
# dependencies: module use /g/data/xp65/public/modules ; module load conda/analysis3

import xarray as xr
import numpy as np
import glob
import dask
from dask.distributed import Client
import cftime
import datetime

CICE_DAILY_FN = "access-om3.cice*.????-??-??-?????.nc"  # "access-om3.cice.1day.mean.????-??-??-?????.nc"


def start_of_month(dt):
    """
    For a cftime.datetime object, return the start of the month
    """
    # Get the class of the input datetime (e.g., DatetimeNoLeap, Datetime360Day, etc.)
    dt_class = type(dt)
    first_day = dt_class(dt.year, dt.month, 1)

    return dt_class(first_day.year, first_day.month, first_day.day)


def end_of_month(dt):
    """
    For a cftime.datetime object, return the end of the month
    """
    # Get the class of the input datetime (e.g., DatetimeNoLeap, Datetime360Day, etc.)
    dt_class = type(dt)

    # Handle December by rolling over to January of next year
    if dt.month == 12:
        next_month = dt_class(dt.year + 1, 1, 1)
    else:
        next_month = dt_class(dt.year, dt.month + 1, 1)

    # Subtract one day to get the last day of the current month
    last_day = next_month - datetime.timedelta(days=1)
    return dt_class(last_day.year, last_day.month, last_day.day)


def monthly_ranges(start, end):
    monthly_range = xr.date_range(
        start=str(start),
        end=str(end),
        freq="ME",
        use_cftime=True,
        calendar=daily_ds.time.values[0].calendar,
    )

    print(monthly_range)

    # add the start and end in
    monthly_range = [start, *monthly_range]

    if end != monthly_range[-1]:
        monthly_range = [*monthly_range, end]

    return monthly_range


if __name__ == "__main__":

    client = Client(threads_per_worker=1)

    client

    # #If no directory option provided , then use latest
    # if [ -z $out_dir ]; then
    #     #latest output dir only
    #     out_dir=$(ls -drv archive/output*[0-9] | head -1)
    # fi

    out_dir = "/g/data/tm70/as2285/payu/peturb_tests/perturb_base/work"

    daily_f = glob.glob(f"{out_dir}/{CICE_DAILY_FN}")

    daily_ds = xr.open_mfdataset(
        daily_f,
        compat="override",
        data_vars="minimal",
        coords="minimal",
        chunks=-1,
        parallel=True,
        use_cftime=True,
    )

    times = daily_ds.time.values

    start = start_of_month(np.min(times))
    end = end_of_month(np.max(times))

    monthly_range = monthly_ranges(start, end)
    monthly_pairs = list(zip(monthly_range[:-1], monthly_range[1:]))
    monthly_ncs = list()
    for iRange in monthly_pairs:
        monthly_ncs.append(
            daily_ds.sel(time=slice(*iRange)).to_netcdf(
                f"{out_dir}/access-om3.cice.1day.mean.{str(iRange[1])[0:10]}.nc",
                compute=False,
            )
        )

    dask.compute(monthly_ncs)

    client.close()
