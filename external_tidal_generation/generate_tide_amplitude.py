#!/usr/bin/env python3
# Copyright 2025 ACCESS-NRI and contributors.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Compute the barotropic tidal amplitude from TPXO10-atlas v2
# using eight primary constituents (M2, S2, N2, K2, K1, O1, P1 and Q1)
#
# Reference:
# Adcroft, Alistair, et al.
# "The GFDL Global Ocean and Sea Ice Model OM4.0: Model Description and Simulation Features"
# Journal of Advances in Modeling Earth Systems 11.10 (2019): 3167-3211.
#
# Usage:
#    python3 generate_tide_amplitude.py \
#       --hgrid-file /path/to/ocean_hgrid.nc \
#       --mask-file /path/to/ocean_mask.nc \
#       --data-path /path/to/TPXO10/ \
#       --method  conservative_normed \
#       --output  tideamp.nc
#
# Contact:
#    - Minghang Li <Minghang.Li1@anu.edu.au>
#
# Dependencies:
#   - xarray
#   - numpy
#   - xesmf
# =========================================================================================
from pathlib import Path
import sys
import os
import argparse
import numpy as np
import xarray as xr
import xesmf as xe

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum


PRIMARY_CONSTITUENTS = ["m2", "s2", "n2", "k2", "k1", "o1", "p1", "q1"]


def load_dataset(path: Path) -> xr.Dataset:
    """
    Load an input dataset from a netcdf file.
    """
    return xr.open_dataset(path)


def interp_complex(arr, lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """
    Linear interpolation for complex DataArrays.
    """
    re = arr.real.interp(x=lon, y=lat, method="linear")
    im = arr.imag.interp(x=lon, y=lat, method="linear")
    return (re + 1j * im).astype("complex64")


def compute_constituent_speed(
    trans_file: Path,
    hu: xr.DataArray,
    hv: xr.DataArray,
    lon_z: xr.DataArray,
    lat_z: xr.DataArray,
    lon_u: xr.DataArray,
    lat_v: xr.DataArray,
) -> xr.DataArray:
    """
    Barotropic speed amplitude (m/s) on centre points
    for a single tidal constituent.
    """
    ds = load_dataset(trans_file)

    # complex transports from cm^2/s to m^2/s
    Uc = (ds.uRe + 1j * ds.uIm).astype("complex64") * 1e-4
    Vc = (ds.vRe + 1j * ds.vIm).astype("complex64") * 1e-4

    # unify dimensions to (y,x)
    dim_map = {"ny": "y", "nx": "x"}
    Uc = Uc.rename(dim_map)
    Vc = Vc.rename(dim_map)
    hu = hu.rename(dim_map)
    hv = hv.rename(dim_map)
    lon_z = lon_z.rename({"nx": "x"})
    lat_z = lat_z.rename({"ny": "y"})
    lon_u = lon_u.rename({"nx": "x"})
    lat_v = lat_v.rename({"ny": "y"})

    # compute velocities by transport / depth
    u_vel = (Uc / hu).assign_coords({"x": lon_u, "y": lat_z})
    v_vel = (Vc / hv).assign_coords({"x": lon_z, "y": lat_v})

    # build the 2-D center grids in (y,x)
    lat_c, lon_c = xr.broadcast(lat_z[:-1], lon_z[:-1])

    # interpolate real/imag and calculate speed
    u_c = interp_complex(u_vel, lon_c, lat_c)
    v_c = interp_complex(v_vel, lon_c, lat_c)
    speed_vals = np.sqrt(np.abs(u_c) ** 2 + np.abs(v_c) ** 2)

    # construct da
    speed = xr.DataArray(
        speed_vals,
        dims=("y", "x"),
        coords={
            "lat": (("y", "x"), lat_c.values),
            "lon": (("y", "x"), lon_c.values),
        },
        name="speed",
    )

    return speed


def compute_tideamp(data_path: Path) -> xr.DataArray:
    """
    Compute tidal amplitude (speed) from eight primary constituents
    """
    # TPXO10 grids
    grid = load_dataset(data_path / "grid_tpxo10atlas_v2.nc")
    # depth at u and v points
    hu = grid["hu"].where(grid["hu"] > 0)
    hv = grid["hv"].where(grid["hv"] > 0)

    speed_sq_sum = None

    for cons in PRIMARY_CONSTITUENTS:
        fname = data_path / f"u_{cons}_tpxo10_atlas_30_v2.nc"
        print(f"Processing {fname}...")
        speed = compute_constituent_speed(
            fname, hu, hv, grid.lon_z, grid.lat_z, grid.lon_u, grid.lat_v
        )
        speed_sq_sum = speed**2 if speed_sq_sum is None else speed_sq_sum + speed**2

    # rms over all 8 constituents
    da = np.sqrt(speed_sq_sum).rename("tideamp")
    ds = da.to_dataset()

    # boundary grids for xesmf interpolation
    lon_ub = grid.lon_u.rename({"nx": "x"})
    lat_vb = grid.lat_v.rename({"ny": "y"})
    lat_b2d, lon_b2d = xr.broadcast(lat_vb, lon_ub)

    ds = ds.assign_coords(
        {
            "lon_b": (("y_b", "x_b"), lon_b2d.values),
            "lat_b": (("y_b", "x_b"), lat_b2d.values),
        }
    )

    ds.tideamp.attrs.update(
        units="m/s",
        long_name="Barotropic tidal current speed (8 constituents)",
        source="TPXO10-atlas v2",
        note="RMS of M2, S2, N2, K2, K1, O1, P1, Q1",
    )

    return ds


def regrid(
    hgrid_path: Path,
    mask_path: Path,
    tideamp: xr.Dataset,
    method: str = "conservative_normed",
) -> xr.Dataset:
    """
    Regrid tideamp onto the model grid using xesmf
    """
    # Load model grid
    hgrid = load_dataset(hgrid_path)
    hgrid_x = hgrid.x[1::2, 1::2]
    hgrid_y = hgrid.y[1::2, 1::2]
    hgrid_xc = hgrid.x[::2, ::2]
    hgrid_yc = hgrid.y[::2, ::2]

    # load model ocean mask
    ocean_mask = load_dataset(mask_path)

    # construct target dataset for tideamp
    target_ds = xr.Dataset(
        data_vars={"mask": (("y", "x"), ocean_mask.mask.values)},
        coords={
            "lon": (("y", "x"), hgrid_x.values),
            "lat": (("y", "x"), hgrid_y.values),
            "lon_b": (("y_b", "x_b"), hgrid_xc.values),
            "lat_b": (("y_b", "x_b"), hgrid_yc.values),
        },
    )

    regridder = xe.Regridder(tideamp, target_ds, method=method, periodic=True)

    target_ds["tideamp"] = regridder(tideamp["tideamp"])

    target_ds["tideamp"] = target_ds["tideamp"].fillna(0.0)

    return target_ds


def update_tideamp(tideamp: xr.DataArray) -> xr.DataArray:
    """
    Drop boundary-coordinate variables (lon_b, lat_b) and mask,
    Apply CF-compliant metadata to tideamp.
    """
    tideamp = tideamp.drop_vars(["lon_b", "lat_b", "mask"])

    tideamp_attrs = {
        "lon": {
            "long_name": "Longitude",
            "units": "degrees_east",
        },
        "lat": {
            "long_name": "Latitude",
            "units": "degrees_north",
        },
        "tideamp": {
            "long_name": "Tidal velocity amplitude",
            "units": "m/s",
            "regrid_method": "conservative_normed",
        },
    }

    for var, attrs in tideamp_attrs.items():
        tideamp[var].attrs.update(attrs)

    return tideamp


def main():
    parser = argparse.ArgumentParser(
        description="Compute tidal amplitude from TPXO10-atlas v2 and regrid onto a model grid."
    )
    parser.add_argument(
        "--hgrid-file", type=str, required=True, help="Path to ocean_hgrid.nc"
    )
    parser.add_argument(
        "--mask-file",
        type=str,
        required=True,
        help="Path to the ocean mask file.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="conservative_normed",
        help="Regridding method (e.g., bilinear, conservative, conservative_normed)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Directory containing TPXO10 files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output tideamp file.",
    )

    args = parser.parse_args()

    # Compute tide amplitude on the TPXO10 grid
    tideamp_tmp = compute_tideamp(Path(args.data_path))

    # Regrid tide amplitude onto the model grid
    tideamp = regrid(
        Path(args.hgrid_file), Path(args.mask_file), tideamp_tmp, args.method
    )

    # Remove boundary variables and update metadata to tideamp
    tideamp = update_tideamp(tideamp)

    # rename dims name to xh and yh
    tideamp = tideamp.rename({"x": "xh", "y": "yh"})

    # Add provenance metadata and MD5 hashes for input files.
    this_file = os.path.normpath(__file__)
    runcmd = (
        f"python3 {os.path.basename(this_file)} "
        f"--hgrid-file={args.hgrid_file} "
        f"--mask-file={args.mask_file} "
        f"--method={args.method} "
        f"--data-path={args.data_path} "
        f"--output={args.output} "
    )

    history = get_provenance_metadata(this_file, runcmd)
    global_attrs = {"history": history}

    # add md5 hashes for input files
    file_hashes = [
        f"{args.hgrid_file} (md5 hash: {md5sum(args.hgrid_file)})",
        f"{args.mask_file} (md5 hash: {md5sum(args.mask_file)})",
    ]
    global_attrs["inputFile"] = ", ".join(file_hashes)
    tideamp.attrs.update(global_attrs)

    tideamp.to_netcdf(args.output)
    print(f"Complete {args.output}!")


if __name__ == "__main__":
    main()
