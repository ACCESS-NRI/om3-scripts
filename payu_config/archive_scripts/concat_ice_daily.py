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
import netCDF4
import shutil
import os


CICE_DAILY_FN = "access-om3.cice.1day.mean.1981-??-??.nc"
MONTHLY_STUB_FN = "access-om3.cice.1day.mean."


def start_of_day(dt):
    """
    For a cftime.datetime object, return the start of the month
    """
    # Get the class of the input datetime (e.g., DatetimeNoLeap, Datetime360Day, etc.)
    dt_class = type(dt)
    first_day = dt_class(dt.year, dt.month, 1)

    return dt_class(first_day.year, first_day.month, first_day.day)


def monthly_ranges(start, end):
    monthly_range = xr.date_range(
        start=str(start_of_day(start)),
        end=str(end),
        freq="MS",
        use_cftime=True,
        calendar=daily_ds.time.values[0].calendar,
    )

    # add the end in
    monthly_range = [*monthly_range, end]

    return monthly_range


# If no out_dir is set, pick the latest matching directory
# if 'out_dir' not in locals() or out_dir is None:
#     candidates = glob.glob("archive/output*[0-9]")
#     if candidates:
#         # Sort in reverse chronological order
#         candidates.sort(reverse=True)
#         out_dir = candidates[0]
#     else:
#         raise FileNotFoundError("No matching output directories found.")g
# print("Using:", out_dir)

if __name__ == "__main__":

    # we seem to need about ~6GB per worker to support chunking monthly
    mem_worker = 6 * 1024 * 1024 * 1024
    # find number of workers on first node to achieve this amount of memory
    n_worker = int(
        int(os.environ["PBS_VMEM"]) / int(os.environ["PBS_NNODES"]) / mem_worker
    )
    jobfs = os.environ["PBS_JOBFS"]

    print("number workers")
    print(n_worker)

    client = Client(
        threads_per_worker=1,
        n_workers=n_worker,
        memory_limit=mem_worker,
        local_directory=jobfs,
    )

    in_dir = "/g/data/tm70/as2285/payu/peturb_tests/perturb_base_no_thermo_check/archive/output005/"

    daily_f = glob.glob(f"{in_dir}/{CICE_DAILY_FN}")

    daily_ds = xr.open_mfdataset(
        daily_f,
        compat="override",
        data_vars="minimal",
        coords="minimal",
        combine_attrs="override",
        parallel=True,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )

    daily_ds

    times = daily_ds.time.values

    monthly_range = monthly_ranges(np.min(times), np.max(times))
    monthly_pairs = list(zip(monthly_range[:-1], monthly_range[1:]))
    monthly_ncs = list()

    for iRange in monthly_pairs:
        month_ds = daily_ds.sel(time=slice(*iRange))

        month_ds = month_ds.chunk({"time": len(month_ds.time)})

        monthly_ncs.append(
            month_ds.to_netcdf(
                f"/g/data/tm70/as2285/concat_test/{MONTHLY_STUB_FN}{str(iRange[0])[0:7]}.nc",
                compute=False,
            )
        )

    month_ds

    dask.compute(monthly_ncs)

    client.close()

    # how to test for success?
    # for iF in daily_f:
    # shutil.rm(iF)
