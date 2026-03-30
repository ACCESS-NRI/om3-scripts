# Copyright 2025 COSIMA, ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate inter-annual (IAF) forcing file from  JRA55do output.
# This uses the JRA55do v1.6 from the NCI Replicated CMIP6Plus datasets
# This only combines data into multiple variables per file, no other adjustments are made
# Usage:
# First start up an interactive job or an ARE session on Gadi to get enough memory:
# qsub -I -q normal -l mem=128GB -l storage=gdata/xp65+gdata/qv56+gdata/tm70 -l wd
#
# Then run the following
# module use /g/data/xp65/public/modules ; module load conda/analysis3
# python3 make_iaf.py
#
# Dependencies
#     conda/analysis3-25.04
# =========================================================================================


import xarray
import netCDF4 as nc
import os
import datetime
from glob import glob
from calendar import isleap
import numpy as np
import dask
from dask.distributed import Client

from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum

FILLVALUE = 1e20

# Assume JRA55do v1.6, 1958-2024
jradir = "/g/data/qv56/replicas/input4MIPs/CMIP6Plus/OMIP/MRI/MRI-JRA55-do-1-6-0/"
if __name__ == "__main__":
    base = Path(jradir)
    client = Client(threads_per_worker=1)
    print(client.dashboard_link)
    ncs = list()

    for year in range(1958, 2024):

        files = {
            "atm_ave": sorted(base.glob(f"atmos/3hr/*/gr/*/*{year}*.nc")),
            "atm_pt": sorted(base.glob(f"atmos/3hrPt/*/gr/*/*{year}*.nc")),
            "rof_ave": sorted(base.glob(f"land*/day/*/gr/*/*{year}*.nc")),
        }

        ds = {}

        for group, filelist in files.items():
            iaf_files = str()
            print(group, year)

            ds = xarray.open_mfdataset(
                filelist,
                data_vars="all",
                compat="override",
                coords="minimal",
                decode_coords=False,
                parallel=True,
            )

            for dim in ds.dims:
                # Have to give all dimensions a useless FillValue attribute, otherwise xarray
                # makes it NaN and MOM does not like this
                ds[dim].encoding["_FillValue"] = FILLVALUE

            # Add some info about how the file was generated
            this_file = os.path.normpath(__file__)
            runcmd = f"python3 {os.path.basename(this_file)}"
            ds.attrs |= {"IAF_creation": get_provenance_metadata(this_file, runcmd)}
            for file in filelist:
                iaf_files += f"{file} (md5 hash: {md5sum(file)}, )"
            ds.attrs |= {"IAF_inputFiles": iaf_files}

            outfile = f"{group}_MRI-JRA55-do-1-6-0_{year}"
            print("Creating ", outfile)
            ncs.append(ds.to_netcdf(outfile, compute=False))

    print("Saving")
    dask.compute(ncs)

    client.close()
