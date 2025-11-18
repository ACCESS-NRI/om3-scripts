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
import regionmask
from scipy import ndimage
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from distributed import Client

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum

xr.set_options(keep_attrs=True)


def fill_missing_data(field, wet_mask, maxiter=0):
    """
    Fill missing ocean values using a sparse Laplacian solve.
    Adapted from https://github.com/adcroft/interp_and_fill/blob/main/Interpolate%20and%20fill%20SeaWIFS.ipynb

    Parameters
    ----------
    field : numpy.ndarray
        Input data containing missing data
    wet_mask : numpy.ndarray
        Wet cell mask (0 land, 1 ocean)

    Returns
    -------
    numpy.ma.array
        Data array with missing ocean points filled.
    """

    def _process_neighbour(n, jn, in_):
        """Process neighbour at (jn, in_) for row n."""
        if wet_mask[jn, in_] <= 0:
            return

        ld[n] -= 1
        idx = ind[jn, in_]

        if idx >= 0:
            A[n, idx] = 1.0
        else:
            b[n] -= field[jn, in_]

    nj, ni = field.shape
    missing_mask = np.isnan(field)
    field = np.where(missing_mask, 0, field)

    # Index lookup for missing points
    missing_j, missing_i = np.where(missing_mask & (wet_mask > 0))
    n_missing = missing_j.size
    ind = np.full(field.shape, -1, dtype=int)
    ind[missing_j, missing_i] = np.arange(n_missing)

    # Sparse matrix in LIL format (fast incremental building)
    A = sp.lil_matrix((n_missing, n_missing))
    b = np.zeros(n_missing)
    ld = np.zeros(n_missing)

    # Build matrix row-by-row
    for n in range(n_missing):
        j = missing_j[n]
        i = missing_i[n]

        im1 = (i - 1) % ni
        ip1 = (i + 1) % ni
        jm1 = j - 1 if j > 0 else 0
        jp1 = j + 1 if j < nj - 1 else nj - 1

        if j > 0:
            _process_neighbour(n, jm1, i)
        _process_neighbour(n, j, im1)
        _process_neighbour(n, j, ip1)
        if j < nj - 1:
            _process_neighbour(n, jp1, i)

        # Tri-polar fold
        if j == nj - 1:
            fold_i = ni - 1 - i
            _process_neighbour(n, j, fold_i)

    # Set leading diagonal
    b[ld >= 0] = 0.0
    stabilizer = 1e-14
    diag_vals = ld - stabilizer
    A[np.arange(n_missing), np.arange(n_missing)] = diag_vals

    # Convert to CSR and solve
    A = A.tocsr()
    x = spla.spsolve(A, b)

    # Fill the missing values
    field[missing_j, missing_i] = x

    return field


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

    this_file = os.path.normpath(__file__)

    # Add some info about how the file was generated
    runcmd = (
        f"python3 {os.path.basename(this_file)} --input-directory={input_directory} "
        f"--output-filename={output_filename}"
    )

    history_attrs = {
        "history": get_provenance_metadata(this_file, runcmd),
    }

    # Load the input data and compute the monthly climatology
    input_files = sorted(glob.glob(f"{input_directory}/*.nc"))

    print("Calculating the monthly climatology...")

    with Client(threads_per_worker=1) as client:
        ds = xr.open_mfdataset(
            input_files, chunks={"lat": 1024, "lon": 1024}, parallel=True
        )
        chl = ds[["CHL"]].groupby("time.month").mean("time").compute()

    print("Filling missing data...")

    # Fill missing data for each month
    for month in range(1, 13):
        print(f"  Filling month {month}...")

        chl_month = chl["CHL"].sel(month=month)

        # Create land mask, eroded to ensure we have values at wet cells near coasts
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(chl_month)
        land_eroded = ndimage.binary_erosion(land == 0.0, structure=np.ones((100, 100)))

        # Remove chl values on land
        chl_month = chl_month.where(land != 0.0).values
        chl["CHL"].sel(month=month).values[:] = fill_missing_data(
            chl_month, 1.0 - land_eroded
        )

    # Add time array
    calendar = "gregorian"
    times = xr.date_range(
        start="0001",
        periods=13,
        freq="MS",
        calendar=calendar,
        use_cftime=True,
    )
    times = times[:-1] + times.diff()[1:] / 2  # Middle of the month
    chl = chl.rename({"month": "time"}).assign_coords({"time": times})

    # Add attrs and save
    chl.attrs["title"] = (
        "Global ocean surface Chlorophyll-a concentration filled monthly climatology"
    )
    chl.time.attrs["long_name"] = "Time"
    chl.time.attrs["standard_name"] = "time"
    chl.attrs |= history_attrs
    comp = dict(zlib=True, complevel=4)
    encoding = {var: comp for var in chl.data_vars}
    # Time coords should be double type according for CF conventions
    encoding |= {
        "time": {
            "dtype": "float64",
            "units": "days since 0001-01-01 00:00:00.000000",
            "calendar": calendar,
        }
    }

    chl.to_netcdf(output_filename, unlimited_dims="time", encoding=encoding)


if __name__ == "__main__":
    import argparse

    main()
