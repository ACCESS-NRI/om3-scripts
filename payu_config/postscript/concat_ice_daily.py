#!python3
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
#
# For daily cice output files, concatenate them together into one file per month
# either use a provided directory through the --directory argument, or use the latest
# from the `archive/ouput???` folders
#
# if some daily data is already in one file per month, don't change those files
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
MEM_WORKER = 8 * 1024 * 1024 * 1024


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


def lazy_open(files):
    """
    fast open a lot of files, in parallel, as dask objects
    """
    return xr.open_mfdataset(
        files,
        compat="override",
        data_vars="minimal",
        coords="minimal",
        combine_attrs="override",
        parallel=True,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )


def start_client(assume_gadi=True):
    """
    start dask using 8GB of memory per worker, and a max of one node (just incase
    this is a multi-node job)
    """
    if assume_gadi:
        n_worker = int(
            int(os.environ["PBS_VMEM"]) / int(os.environ["PBS_NNODES"]) / MEM_WORKER
        )
        jobfs = os.environ["PBS_JOBFS"]

        client = Client(
            threads_per_worker=1,
            n_workers=n_worker,
            memory_limit=MEM_WORKER,
            local_directory=jobfs,
        )
    else:
        client = Client(threads_per_worker=1, n_workers=1)

    def set_env():
        os.environ["PYTHONNOUSERSITE"] = "True"

    client.run(set_env)

    return client


@dask.delayed
def compare_dataarrays(input_da, expected_da):
    # use xr.testing because in a normal comparison NAN==NAN evaluates as false
    xr.testing.assert_equal(input_da, expected_da)


class Concat_Ice_Daily:

    def __init__(self, directory=None, assume_gadi=True):
        if directory is None:
            output_f = glob.glob("archive/output*")
            if not output_f:
                raise Exception(f"No output folder found (looked for archive/output*)")
            output_f.sort()
            directory = output_f[-1]

        print(f"concat_ice_daily: joining daily ice files in {directory}")
        self.daily_f = glob.glob(f"{directory}/{CICE_DAILY_FN}")
        if not self.daily_f:
            raise Exception(f"No daily output files found in {directory}")

        self.client = start_client(assume_gadi)
        daily_ds = lazy_open(self.daily_f).chunk({"time": 31}).persist()

        # del incorrect metadata
        del daily_ds.attrs["comment2"]
        del daily_ds.attrs["comment3"]

        # Add some info about how the file was generated
        this_file = os.path.normpath(__file__)
        runcmd = f"python3 {os.path.basename(this_file)} --directory={os.path.abspath(directory)}"
        daily_ds.attrs["postprocessing"] = get_provenance_metadata(this_file, runcmd)

        self.directory = directory
        self.daily_ds = daily_ds

    def cleanup_exit(self, error_msg, delete_monthf=False):
        for file in self.month_f:
            if file.exists() and delete_monthf:
                os.remove(file)
        self.client.close()

        raise Exception(error_msg)

    def process(self):

        # find months in dataset
        times = self.daily_ds.time.values
        monthly_range = monthly_ranges(np.min(times), np.max(times), times[0].calendar)
        monthly_pairs = list(zip(monthly_range[:-1], monthly_range[1:]))

        # slice ds for each month, and make a dask delayed object to save to file
        # ignore incomplete months
        monthly_ncs = list()
        self.month_ds = list()
        self.month_f = list()
        for pair in monthly_pairs:

            filename = Path(f"{self.directory}/{MONTHLY_STUB_FN}{str(pair[0])[0:7]}.nc")
            ds = self.daily_ds.sel(time=slice(*pair))
            ds = ds.chunk({"time": len(ds.time)})

            # check for whole month
            if ds.time.values[-1] != (
                ds.time.values[0]
                + datetime.timedelta(days=ds.time.values[0].daysinmonth - 1)
            ):
                print(
                    f"concat_ice_daily:ignoring incomplete month: {str(pair[0])[0:7]}"
                )
                if len(self.daily_ds.time) > len(ds.time):
                    self.daily_ds = self.daily_ds.drop_sel(time=ds.time.values)
            else:
                self.month_f.append(filename)
                self.month_ds.append(ds)

                # if monthly file already exists, don't process again
                if not filename.exists():
                    monthly_ncs.append(ds.to_netcdf(filename, compute=False))

        if len(self.month_f) == 0:
            self.cleanup_exit(
                f"concat_ice_daily: No whole months to concatenate found in {self.directory}"
            )

        # load and save all months concurrently
        try:
            dask.compute(monthly_ncs)
        except:
            self.cleanup_exit(
                "concat_ice_daily: dask compute of saving monthly output failed",
                delete_monthf=True,
            )

        print("cice concat finished")

        # test output is same as input, split into month & by var to parallelise
        # todo: split by chunk instead to parallelise
        tasks = list()
        for i in range(0, len(self.month_ds)):
            output_month_ds = lazy_open(self.month_f[i])
            tasks.append(
                [
                    compare_dataarrays(self.month_ds[i][var], output_month_ds[var])
                    for var in self.daily_ds.data_vars
                ]
            )

        try:
            dask.compute(*tasks)
        except:
            self.client.close()
            raise Exception(
                "concat_ice_daily: one or more output files do not match input"
            )

    def delete_daily_files(self):
        # Extract the YYYY-MM from the month_f filenames
        monthly_keys = {
            Path(f).stem[-7:]
            for f in self.month_f  # extracts the last 7 characters: YYYY-MM
        }

        # Delete daily file if corresponding monthly file exists
        for file in self.daily_f:
            yyyymm = Path(file).stem[-10:-3]  # extracts YYYY-MM from YYYY-MM-DD
            if yyyymm in monthly_keys:
                # print(f"removing {file}")
                os.remove(file)

        print(f"concat_ice_daily: finished processing {self.directory}")


def concat_ice_daily(directory=None, assume_gadi=True):
    concat = Concat_Ice_Daily(directory, assume_gadi)
    concat.process()
    concat.client.close()
    concat.delete_daily_files()


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
