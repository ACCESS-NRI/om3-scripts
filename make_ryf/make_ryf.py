# Copyright 2025 COSIMA, ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate a a repeat year forcing (RYF) forcing file from  JRA55do output.
# (May 1990 to May 1991)
# This uses the JRA55do v1.6 from the NCI Replicated CMIP6Plus datasets
# Paper:
# K.D. Stewart, W.M. Kim, S. Urakawa, A.McC. Hogg, S. Yeager, H. Tsujino, H. Nakano, A.E. Kiss, G. Danabasoglu,
# JRA55-do-based repeat year forcing datasets for driving ocean–sea-ice models,
# Ocean Modelling, Volume 147, 2020,
# https://doi.org/10.1016/j.ocemod.2019.101557.
#
# Usage:
# First start up an interactive job or an ARE session on Gadi to get enough memory:
# qsub -I -q express -l mem=32GB -l storage=gdata/xp65+gdata/qv56+gdata/tm70 -l wd
#
# Then run the following to create the May-May repeat year forcings
# module use /g/data/xp65/public/modules ; module load conda/analysis3
# python3 make_ryf.py
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

from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum

FILLVALUE = 1e20
# compression settings to use
COMPLEVEL = 1
COMPRESSION = "zlib"

source_data = "jra55v1p6"
# source_data = "jra55v1p4"

if source_data == "jra55v1p4":
    # see https://raw.githubusercontent.com/COSIMA/1deg_jra55_iaf/2d6fdf53ae89124e7e11d40176813c286a8279bb/atmosphere/forcing.json
    jradir = "/g/data/qv56/replicas/input4MIPs/CMIP6/OMIP/MRI/MRI-JRA55-do-1-4-0/"
elif source_data == "jra55v1p6":
    jradir = "/g/data/qv56/replicas/input4MIPs/CMIP6Plus/OMIP/MRI/MRI-JRA55-do-1-6-0/"
else:
    jradir = "/g/data/ua8/JRA55-do/v1-3/"

if source_data == "jra55v1p4" or source_data == "jra55v1p6":
    variables = [
        "rsds",
        "rlds",
        "prra",
        "prsn",
        "psl",
        "friver",
        "tas",
        "huss",
        "uas",
        "vas",
        "licalvf",
    ]
    years = (1990,)
else:
    variables = [
        "q_10",
        "rain",
        "rlds",
        "rsds",
        "slp",
        "snow",
        "t_10",
        "u_10",
        "v_10",
        "runoff_all",
    ]
    years = (1984, 1990, 2003)


# loop over years
for year1 in years:

    # By default the second half of year1 is stitched into first half of year2
    year2 = year1 + 1

    # If second year is a leap year make the "base year" year1 and use time slices from the beginning of the year
    if isleap(year2):
        baseyear = year1
        timeslice1 = slice(
            datetime.datetime(year1, 1, 1, 0, 0),
            datetime.datetime(year1, 4, 30, 23, 59),
        )
        # Take one less day in this slice, to account for the leap day
        timeslice2 = slice(
            datetime.datetime(year2, 1, 1, 0, 0),
            datetime.datetime(year2, 4, 29, 23, 59),
        )
    else:
        baseyear = year2
        timeslice1 = slice(datetime.datetime(year1, 5, 1, 0, 0), None)
        timeslice2 = slice(datetime.datetime(year2, 5, 1, 0, 0), None)

    ds = {}

    for var in variables:
        ryf_files = str()
        print(var)
        for y in (year1, year2):
            if source_data == "jra55v1p4" or source_data == "jra55v1p6":
                # see https://raw.githubusercontent.com/COSIMA/1deg_jra55_iaf/2d6fdf53ae89124e7e11d40176813c286a8279bb/atmosphere/forcing.json
                files = glob(
                    os.path.join(
                        jradir,
                        "atmos/3hr/{v}/gr/v*/{v}*{yr}*.nc".format(v=var, yr=y),
                    )
                )
                files += glob(
                    os.path.join(
                        jradir,
                        "atmos/3hrPt/{v}/gr/v*/{v}*{yr}*.nc".format(v=var, yr=y),
                    )
                )
                files += glob(
                    os.path.join(
                        jradir,
                        "land/day/{v}/gr/v*/{v}*{yr}*.nc".format(v=var, yr=y),
                    )
                )
                files += glob(
                    os.path.join(
                        jradir,
                        "landIce/day/{v}/gr/v*/{v}*{yr}*.nc".format(v=var, yr=y),
                    )
                )
            else:
                files = glob(os.path.join(jradir, "{}.{}.*.nc".format(var, y)))
            print("Loading {} for {}".format(files[0], y))
            ds[y] = xarray.open_dataset(files[0], decode_coords=False)
            # save info for metadata
            ryf_files += f"{files[0]} (md5 hash: {md5sum(files[0])}, )"
        # Make a copy of the second year without time_bnds
        ryf = ds[baseyear].drop_vars("time_bnds")
        ryf.encoding = ds[baseyear].encoding

        for varname in ryf.data_vars:
            # Have to give all variables a FillValue attribute, otherwise xarray
            # makes it NaN which causes floating point errors
            # copy FillValue if it exists, otherwise use default
            source_ds = ds[baseyear]
            ryf[varname].encoding["_FillValue"] = (
                source_ds[varname].encoding.get("_FillValue")
                or FILLVALUE
            )

            # Only process variables with 3 or more dimensions
            if len(ryf[varname].shape) < 3:
                continue

            print("Processing ", varname)
            if isleap(year2):
                # Set the Jan->Apr values to those from the first year
                ryf[varname].loc[dict(time=timeslice1)] = (
                    ds[year2][varname].sel(time=timeslice2).values
                )
            else:
                # Set the May->Dec values to those from the first year
                ryf[varname].loc[dict(time=timeslice2)] = (
                    ds[year1][varname].sel(time=timeslice1).values
                )

        for dim in ryf.dims:
            # Have to give all dimensions a useless FillValue attribute, otherwise xarray
            # makes it NaN and MOM does not like this
            ryf[dim].encoding["_FillValue"] = FILLVALUE

        # Make a new time dimension with no offset from origin (1900-01-01) so we don't get an offset after
        # changing calendar to noleap
        newtime = (
            ryf.indexes["time"].values - np.datetime64(f"{year2}-01-01", "D")
        ) + np.datetime64("1900-01-01", "D")
        ryf.indexes["time"].values[:] = newtime[:]

        ryf["time"].attrs = {
            "modulo": " ",
            "axis": "T",
            "cartesian_axis": "T",
            "standard_name": "time",
            # 'calendar':'noleap'
        }

        ryf[var].encoding.update(
            {
                "compression": COMPRESSION,
                "complevel": COMPLEVEL,
            }
        )

        # Add some info about how the file was generated
        this_file = os.path.normpath(__file__)
        runcmd = f"python3 {os.path.basename(this_file)}"
        ryf.attrs |= {"RYF_creation": get_provenance_metadata(this_file, runcmd)}
        ryf.attrs |= {"RYF_inputFiles": ryf_files}

        outfile = "RYF.{}.{}_{}.nc".format(var, year1, year2)
        print("Writing ", outfile)
        ryf.to_netcdf(outfile)

        # Open the file again directly with the netCDF4 library to change the calendar attribute. xarray
        # has a hissy fit as this violates it's idea of CF encoding if it is done before writing the file above
        ryf = nc.Dataset(outfile, mode="r+")
        ryf.variables["time"].calendar = "noleap"
        ryf.close()
