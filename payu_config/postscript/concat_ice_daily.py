#!python3
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
#
# For daily cice output files, concatenate them together into one file per month
# either use a provided directory through the --directory argument, or use the latest
# from the `archive/ouput???` folders
#
# dependencies: module use /g/data/xp65/public/modules ; module load conda/analysis3
#
# example:
# qsub -v PROJECT,SCRIPTS_DIR=/g/data/tm70/as2285/om3-scripts/ -lstorage=${PBS_NCI_STORAGE}+gdata/xp65 /g/data/tm70/as2285/om3-scripts/payu_config/postscript.sh

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
import warnings
from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from scripts_common import get_provenance_metadata, md5sum

CICE_DAILY_FN = "access-om3.cice.1day.mean.????-??-??.nc"
MONTHLY_STUB_FN = "access-om3.cice.1day.mean."


def start_of_day(dt):
    """
    For a cftime.datetime object, return the start of the month
    """
    # Get the class of the input datetime (e.g., DatetimeNoLeap, Datetime360Day, etc.)
    dt_class = type(dt)
    first_day = dt_class(dt.year, dt.month, 1)

    return dt_class(first_day.year, first_day.month, first_day.day)


def monthly_ranges(start, end, cal):
    monthly_range = xr.date_range(
        start=str(start_of_day(start)),
        end=str(end),
        freq="MS",
        use_cftime=True,
        calendar=cal,
    )

    # add the end in
    monthly_range = [*monthly_range, end]

    return monthly_range


def start_client(assume_gadi=True):
    """
    start dask using 6GB of memory per worker, and a max of one node (just incase
    this is a multi-node job)
    """
    if assume_gadi:
        mem_worker = 6 * 1024 * 1024 * 1024
        n_worker = int(
            int(os.environ["PBS_VMEM"]) / int(os.environ["PBS_NNODES"]) / mem_worker
        )
        jobfs = os.environ["PBS_JOBFS"]

        return Client(
            threads_per_worker=1,
            n_workers=n_worker,
            memory_limit=mem_worker,
            local_directory=jobfs,
        )
    else:
        return Client(threads_per_worker=1, n_workers=4)


def concat_ice_daily(directory=None, assume_gadi=True):

    if directory is None:
        output_f = glob.glob("archive/output*")
        if not output_f:
            warnings.warn(f"No output found in archive/output???")
            exit()
        output_f.sort()
        directory = output_f[-1]

    print(f"concat_ice_daily: joining daily ice files in {directory}")

    daily_f = glob.glob(f"{directory}/{CICE_DAILY_FN}")

    if not daily_f:
        warnings.warn(f"No daily output files found in {directory}")
        exit()

    client = start_client(assume_gadi)

    daily_ds = xr.open_mfdataset(
        daily_f,
        compat="override",
        data_vars="minimal",
        coords="minimal",
        combine_attrs="override",
        parallel=True,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )

    # del incorrect metadata
    del daily_ds.attrs["comment2"]
    del daily_ds.attrs["comment3"]

    # Add some info about how the file was generated
    this_file = os.path.normpath(__file__)
    runcmd = f"python3 {os.path.basename(this_file)} --directory={os.path.abspath(directory)}"
    daily_ds.attrs["postprocessing"] = get_provenance_metadata(this_file, runcmd)

    # find months in dataset
    times = daily_ds.time.values
    monthly_range = monthly_ranges(
        np.min(times), np.max(times), daily_ds.time.values[0].calendar
    )
    monthly_pairs = list(zip(monthly_range[:-1], monthly_range[1:]))
    monthly_ncs = list()

    # slice ds for each month, and make a dask delayed object to save to file
    for iRange in monthly_pairs:

        month_f = Path(f"{directory}/{MONTHLY_STUB_FN}{str(iRange[0])[0:7]}.nc")

        if not month_f.exists():
            month_ds = daily_ds.sel(time=slice(*iRange))

            month_ds = month_ds.chunk({"time": len(month_ds.time)})

            monthly_ncs.append(
                month_ds.to_netcdf(
                    month_f,
                    compute=False,
                )
            )

    # run all dask tasks concurrently
    dask.compute(monthly_ncs)

    client.close()

    # check output exists
    month_f = glob.glob(f"{directory}/{MONTHLY_STUB_FN}????-??.nc")

    # Extract the YYYY-MM from those filenames
    monthly_keys = {
        Path(f).stem[-7:] for f in month_f  # extracts the last 7 characters: YYYY-MM
    }

    # Delete daily file if corresponding monthly file exists
    for file in daily_f:
        yyyymm = Path(file).stem[-10:-3]  # extracts YYYY-MM from YYYY-MM-DD
        if yyyymm in monthly_keys:
            os.remove(file)

    print(f"concat_ice_daily: finished processing {directory}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Concatenate daily cice output into one file per month"
    )

    parser.add_argument(
        "--directory",
        type=str,
        required=False,
        help="The directory to be processed",
    )

    args = parser.parse_args()

    concat_ice_daily(args.directory)
