# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# These are common functions/classes which assist with regridding
# =========================================================================================

import os
import xesmf as xe
import xarray as xr
import argparse
import numpy as np
import regionmask
from scipy import ndimage
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from scripts_common import md5sum


def _guess_longitude_name(ds):
    coords = ds.cf.coordinates
    if "longitude" in coords:
        if len(coords["longitude"]) == 1:
            return coords["longitude"][0]
        else:
            raise KeyError("Multiple longitude variables exist. Please pass --lon-name")
    elif "lon" in ds:
        return "lon"
    else:
        raise KeyError(
            "Cannot automatically determine the name of the longitude variable. Please pass --lon-name"
        )


def _guess_latitude_name(ds):
    coords = ds.cf.coordinates
    if "latitude" in coords:
        if len(coords["latitude"]) == 1:
            return coords["latitude"][0]
        else:
            raise KeyError("Multiple latitude variables exist. Please pass --lat-name")
    elif "lat" in ds:
        return "lat"
    else:
        raise KeyError(
            "Cannot automatically determine the name of the latitude variable. Please pass --lat-name"
        )


class Regrid_Common:

    def __init__(self):
        self.method = "bilinear"
        self.extrap = "nearest_s2d"
        self.periodic = True
        self.parser = argparse.ArgumentParser(
            description=(
                "Interpolate a provided forcing file to a grid specified by a provided MOM supergrid file"
            )
        )

    def parse_cli(self):

        parser = self.parser

        parser.add_argument(
            "--hgrid-filename",
            type=str,
            required=True,
            help="The path to the MOM supergrid file to use as a grid for interpolation.",
        )

        parser.add_argument(
            "--output-filename",
            type=str,
            required=True,
            help="The path to the file to be outputted.",
        )

        parser.add_argument(
            "--mask-filename",
            type=str,
            default=None,
            help=(
                "The path to a land-sea (0-1) mask for the forcing file. Cells that fall on land in "
                "the destination grid will take their values from the nearest source grid cell."
            ),
        )

        parser.add_argument(
            "--lon-name",
            type=str,
            default=None,
            help=(
                "The name of the longitude variable in the input grid. If not passed, an attempt will "
                "be made to guess the name."
            ),
        )

        parser.add_argument(
            "--lat-name",
            type=str,
            default=None,
            help=(
                "The name of the latitude variable in the input grid. If not passed, an attempt will "
                "be made to guess the name."
            ),
        )

        args = parser.parse_args()
        self.args = args
        self.hgrid_filename = os.path.abspath(args.hgrid_filename)
        self.output_filename = os.path.abspath(args.output_filename)
        self.mask_filename = args.mask_filename
        self.lon_name = args.lon_name
        self.lat_name = args.lat_name

        if self.mask_filename:
            self.mask_filename = os.path.abspath(self.mask_filename)

        # some info about how the file was generated

        runcmd_args = f"--hgrid-filename={self.hgrid_filename} --output-filename={self.output_filename}"

        if self.mask_filename:
            runcmd_args += f" --mask-filename={self.mask_filename}"
        if self.lon_name:
            runcmd_args += f" --lon-name={self.lon_name}"
        if self.lat_name:
            runcmd_args += f" --lat-name={self.lat_name}"

        self.runcmd_args = runcmd_args

    ## NOTE: it's implied that forcing_filename is set outside this class

    def open_datasets(self):

        forcing_filename = self.forcing_filename
        hgrid_filename = self.hgrid_filename
        mask_filename = self.mask_filename
        lon_name = self.lon_name
        lat_name = self.lat_name

        # Load the input data
        forcing_src = xr.open_dataset(forcing_filename).compute()
        hgrid = xr.open_dataset(hgrid_filename).compute()
        if mask_filename:
            forcing_mask = xr.open_dataset(mask_filename).compute()

        # Drop "mask" variable from forcing_src if it exists
        if "mask" in forcing_src:
            forcing_src = forcing_src.drop_vars("mask")

        # Standardise lon/lat coordinate names
        if not lon_name:
            lon_name = _guess_longitude_name(forcing_src)

        if not lat_name:
            lat_name = _guess_latitude_name(forcing_src)

        forcing_src = forcing_src.rename({lon_name: "lon", lat_name: "lat"})

        # Get source and destination grid
        grid_src = forcing_src[["lon", "lat"]]
        if mask_filename:
            # Add the mask to the source grid so that land values are extrapolated
            if "mask" in forcing_mask:
                grid_src = grid_src.assign(mask=forcing_mask["mask"])
            else:
                raise ValueError(
                    f"Input mask-filename must contain a variable named 'mask'"
                )

        # Destination grid is tracer cell centres
        # hgrid = xr.open_dataset(hgrid_path)
        hgrid_x = hgrid.x[1::2, 1::2]
        hgrid_y = hgrid.y[1::2, 1::2]
        hgrid_xc = hgrid.x[::2, ::2]
        hgrid_yc = hgrid.y[::2, ::2]
        grid_dest = xr.Dataset(
            coords={
                "lon": (("ny", "nx"), hgrid_x.values),
                "lat": (("ny", "nx"), hgrid_y.values),
                "lon_b": (("nyp", "nxp"), hgrid_xc.values),
                "lat_b": (("nyp", "nxp"), hgrid_yc.values),
            },
        )

        self.forcing_src = forcing_src
        if mask_filename:
            self.forcing_mask = forcing_mask
        self.grid_src = grid_src
        self.grid_dest = grid_dest

    def regrid_forcing(self):

        # Regrid using bilinear interpolation with nearest neighbour extrapolation
        # NOTE: This will not conserve global quantities
        regridder = xe.Regridder(
            self.grid_src,
            self.grid_dest,
            method=self.method,
            extrap_method=self.extrap,
            periodic=self.periodic,
        )
        forcing_regrid = regridder(self.forcing_src, keep_attrs=True)

        # Add coodinates (required by data_table)
        forcing_regrid = forcing_regrid.assign_coords(
            lon=self.grid_dest["lon"], lat=self.grid_dest["lat"]
        )

        forcing_regrid["lat"].attrs = dict(
            long_name="Latitude of T-cell center",
            standard_name="latitude",
            units="degree_north",
        )
        forcing_regrid["lon"].attrs = dict(
            long_name="Longitude of T-cell center",
            standard_name="longitude",
            units="degrees_east",
        )

        self.forcing_regrid = forcing_regrid

    def save_output(self):

        forcing_regrid = self.forcing_regrid

        # Info about input data used
        file_hashes = [
            f"{self.forcing_filename} (md5 hash: {md5sum(self.forcing_filename)})",
            f"{self.hgrid_filename} (md5 hash: {md5sum(self.hgrid_filename)})",
        ]
        if self.mask_filename:
            file_hashes.append(
                f"{self.mask_filename} (md5 hash: {md5sum(self.mask_filename)})"
            )

        global_attrs = {
            "inputFile": ", ".join(file_hashes),
        }
        forcing_regrid.attrs = forcing_regrid.attrs | global_attrs

        # Save output
        # _FillValue is required by older (MOM5-era) versions of FMS
        var_encoding = dict(zlib=True, complevel=4, _FillValue=-1.0e10)
        for var in forcing_regrid.data_vars:
            forcing_regrid[var].encoding |= var_encoding
        # Coordinates should not have _FillValue
        coord_encoding = dict(_FillValue=None)
        for coord in forcing_regrid.coords:
            forcing_regrid[coord].encoding |= coord_encoding

        unlimited_dims = "time" if "time" in forcing_regrid.dims else None
        forcing_regrid.to_netcdf(self.output_filename, unlimited_dims=unlimited_dims)


def _fill_missing_horiz(field, mask, top_bound="none"):
    """
    Fill missing values using a sparse Laplacian solve.
    Adapted from https://github.com/adcroft/interp_and_fill/blob/main/Interpolate%20and%20fill%20SeaWIFS.ipynb

    Parameters
    ----------
    field : numpy.ndarray
        Input data containing missing data
    mask : numpy.ndarray
        Fill mask (0 or 1). Missing values are not filled where mask==1
    top_bound : {"none", "tripole", "regular"}, optional
        Connectivity across the northern boundary. "none" uses only the south, west and east
        neighbours at the top row. "tripole" uses tripolar connectivity; "regular" uses a
        regular lat/lon pole. Default is "none".

    Returns
    -------
    numpy.ndarray
        Data array with missing points filled.
    """

    def _process_neighbour(n, jn, in_):
        """Process neighbour at (jn, in_) for row n."""
        if mask[jn, in_] <= 0:
            return

        ld[n] -= 1
        idx = ind[jn, in_]

        if idx >= 0:
            A[n, idx] = 1.0
        else:
            b[n] -= field[jn, in_]

    nj, ni = field.shape

    if top_bound not in ("none", "regular", "tripole"):
        raise ValueError(
            f"top_bound must be one of ('none', 'regular', 'tripole'), got {top_bound}"
        )
    if top_bound in ("regular", "tripole") and ni % 2 != 0:
        raise ValueError(
            f"top_bound='{top_bound}' requires an even number of longitude points, got {ni}"
        )

    missing_mask = np.isnan(field)
    field = np.where(missing_mask, 0, field)

    # Index lookup for missing points
    missing_j, missing_i = np.where(missing_mask & (mask > 0))
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

        # Top boundary
        if (top_bound != "none") and (j == nj - 1):
            if top_bound == "tripole":
                fold_i = ni - 1 - i
            elif top_bound == "regular":
                fold_i = (i + ni // 2) % ni
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

    return np.where(mask, field, np.nan)


def fill_ocean_horiz(da, top_bound="none", n_erode=0, erode_first=True):
    """
    Fill missing ocean values using a sparse Laplacian solve.
    The land mask is determined from Natural Earth v5.0.0 and can be eroded using n_erode.

    Parameters
    ----------
    da : xr.DataArray
        Input data containing missing ocean data (horziontal slice)
    top_bound : {"none", "tripole", "regular"}, optional
        Connectivity across the northern boundary. "none" uses only the south, west and east
        neighbours at the top row. "tripole" uses tripolar connectivity; "regular" uses a
        regular lat/lon pole. Default is "none".
    n_erode : int
        The size of the structure used to erode the land mask. This can be useful for
        ensuring there are values at near coastal points, where the land-sea mask may
        differ between the input data and the model
    erode_first : boolean
        If False, fill missing values in two steps. First, fill the missing wet cells, then
        the eroded land areas (if n_erode > 0). If this is done in one step, high values
        near the coast have a larger weighting leading to larger values in high latitude
        filled regions.

    Returns
    -------
    xr.DataArray
        Data array with missing ocean points filled
    """

    land = (
        regionmask.defined_regions.natural_earth_v5_0_0.land_10.mask(da).values == 0.0
    )

    # Remove any values on land so that they don't influence the fill of ocean values
    da = da.where(np.logical_not(land))

    if erode_first and (n_erode > 0):
        land = ndimage.binary_erosion(land, structure=np.ones((n_erode, n_erode)))

    da_filled = _fill_missing_horiz(da, 1.0 - land, top_bound=top_bound)

    if (not erode_first) and (n_erode > 0):
        land_eroded = ndimage.binary_erosion(
            land, structure=np.ones((n_erode, n_erode))
        )
        da_filled = _fill_missing_horiz(
            da_filled, 1.0 - land_eroded, top_bound=top_bound
        )

    return da.copy(data=da_filled)
