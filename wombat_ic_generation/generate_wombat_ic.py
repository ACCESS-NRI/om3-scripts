# Copyright 2023 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate initial conditions for WOMBAT tracers that aren't initialised directly from source
# files. All constant variables are on the WOA23 silicon 1deg grid. sil (when required) is
# taken from WOA23 silicon (1deg grid). n2o (when required) is calculated from WOA23 temperature
# and salinity (0.25deg grid).
#
# To run:
#   python generate_wombat_ic.py --wombat-version=<lite|mid>
#       --output-file=<path-to-output-file>
#
# For more information, run `python generate_wombat_ic.py -h`
#
# The run command and full github url of the current version of this script is added to the
# metadata of the generated IC file. This is to uniquely identify the script and inputs used
# to generate the IC file. To produce IC files for sharing, ensure you are using a version
# of this script which is committed and pushed to github. For IC files intended for released
# configurations, use the latest version checked in to the main branch of the github repository.
#
# Contact:
#   Dougie Squire <dougie.squire@anu.edu.au>
#
# Dependencies:
#   argparse, xarray, numpy, scipy
# =========================================================================================

import os
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum

WOA23_PATH = "/g/data/av17/access-nri/OM3/woa23"
WOA23_Si_FILE = f"{WOA23_PATH}/woa23_all_i01_01.nc"
WOA23_T_FILE = f"{WOA23_PATH}/woa23_decav_t01_04.nc"
WOA23_S_FILE = f"{WOA23_PATH}/woa23_decav_s01_04.nc"

# no3, o2, dic, alk and fe are all initialised directly from other source files
WOMBATLITE_VARS = [
    "phy",
    "zoo",
    "det",
    "caco3",
    "pchl",
    "phyfe",
    "zoofe",
    "detfe",
    "dicp",
    "dicr",
    "det_sediment",
    "caco3_sediment",
    "detfe_sediment",
    "detbury",
    "caco3bury",
]
WOMBATMID_VARS = WOMBATLITE_VARS + [
    "nh4",
    "sil",
    "dia",
    "dchl",
    "diafe",
    "diasi",
    "mes",
    "mesfe",
    "bdet",
    "bdetfe",
    "bdetsi",
    "doc",
    "don",
    "nosdoc",
    "bac1",
    "bac2",
    "aoa",
    "n2o",
    "afe",
    "bafe",
    "detsi_sediment",
]
WOMBAT_VARS = {
    "lite": WOMBATLITE_VARS,
    "mid": WOMBATMID_VARS,
}

# "var": (constant_value, "units")
CONSTANT_VARS = {
    "nh4": (0.0, "mol kg-1"),
    "dia": (0.1e-6, "mol kg-1"),
    "mes": (0.1e-6, "mol kg-1"),
    "phy": (0.1e-6, "mol kg-1"),
    "zoo": (0.1e-6, "mol kg-1"),
    "det": (0.1e-6, "mol kg-1"),
    "bdet": (0.1e-6, "mol kg-1"),
    "doc": (5.0e-6, "mol kg-1"),
    "dicp": (0.0, "mol kg-1"),
    "dicr": (0.0, "mol kg-1"),
    "nosdoc": (0.5, "1"),
    "bac1": (0.1e-6, "mol kg-1"),
    "bac2": (0.1e-6, "mol kg-1"),
    "aoa": (0.1e-6, "mol kg-1"),
    "afe": (0.0, "mol kg-1"),
    "bafe": (0.0, "mol kg-1"),
    "det_sediment": (0.0, "mol m-2"),
    "caco3_sediment": (0.0, "mol m-2"),
    "detfe_sediment": (0.0, "mol m-2"),
    "detsi_sediment": (0.0, "mol m-2"),
    "detbury": (0.0, "mol m-2"),
    "caco3bury": (0.0, "mol m-2"),
}
# "var": ("var_to_scale", scaling_factor)
SCALED_VARS = {
    "pchl": ("phy", 0.004),
    "phyfe": ("phy", 7e-6),
    "dchl": ("dia", 0.004),
    "diafe": ("dia", 7e-6),
    "diasi": ("dia", 16 / 122),
    "zoofe": ("zoo", 7e-6),
    "mesfe": ("mes", 7e-6),
    "detfe": ("det", 7e-6),
    "bdetfe": ("bdet", 7e-6),
    "bdetsi": ("bdet", 16 / 122),
    "don": ("doc", 16 / 122),
}


def _extrapolate_over_land(da):
    """
    Fill masked cells at each depth level with the nearest value.
    This uses a simple (and incorrect) distance_transform_edt per level
    which is fine for this application since it's just to get some reasonable
    values in near-coast land cells that may be ocean in the model domain.
    """

    def fill_level(arr):
        mask = np.isnan(arr)
        if not mask.any():
            return arr
        _, indices = distance_transform_edt(mask, return_indices=True)
        return arr[tuple(indices)]

    filled = np.stack(
        [fill_level(da.isel(depth=i).values) for i in range(da.sizes["depth"])]
    )
    return da.copy(data=filled)


def _add_bottom_level(da, depth=6000):
    """Append a copy of the deepest depth level to `da` at the given depth."""
    return xr.concat([da, da.isel(depth=-1).assign_coords(depth=depth)], dim="depth")


def _open_woa_dataset(file, var, coord_int):
    """
    Opens a WOA23 dataset and returns the specified variable as a DataArray with
    some attributes needed by MOM6, renamed lon/lat coords to allow having
    different coordinates in one dataset and a bottom level added.
    """
    da = (
        xr.open_dataset(
            file,
            decode_times=False,
        )[var]
        .squeeze(drop=True)
        .compute()
    )
    da = _add_bottom_level(da)
    da.attrs.pop("standard_name", None)
    da.lon.attrs["modulo"] = 360.0
    da = da.rename({"lon": f"lon{coord_int}", "lat": f"lat{coord_int}"})
    return da


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate initial conditions for WOMBAT tracers that aren't initialised directly from "
            "source files"
        )
    )

    parser.add_argument(
        "--wombat-version",
        required=True,
        choices=["lite", "mid"],
        help="The version of WOMBAT to generate initial conditions for.",
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help="The path to the initial condition file to be outputted.",
    )

    args = parser.parse_args()
    wombat_version = args.wombat_version
    output_file = args.output_file

    this_file = os.path.normpath(__file__)

    # Add some info about how the file was generated
    runcmd = (
        f"python3 {this_file} --wombat-version={wombat_version} "
        f"--output-file={os.path.abspath(output_file)}"
    )

    global_attrs = {
        "history": get_provenance_metadata(this_file, runcmd),
        "inputFile": f"{WOA23_Si_FILE} (md5 hash: {md5sum(WOA23_Si_FILE)})",
    }

    xr.set_options(keep_attrs=True)
    woa_Si = _open_woa_dataset(WOA23_Si_FILE, "i_an", 1)

    # Do constant variables first, so that they can be used in the calculation
    # of other variables
    ds = {}
    for var in WOMBAT_VARS[wombat_version]:
        if var in CONSTANT_VARS:
            const, units = CONSTANT_VARS[var]
            da = 0 * woa_Si.fillna(0) + const
            da.attrs["units"] = units
            da.attrs["long_name"] = var
            ds[var] = da

    # Do scaled variables next
    for var in WOMBAT_VARS[wombat_version]:
        if var in SCALED_VARS:
            var_to_scale, scaling_factor = SCALED_VARS[var]
            da = ds[var_to_scale] * scaling_factor
            da.attrs["units"] = ds[var_to_scale].attrs["units"]
            da.attrs["long_name"] = var
            ds[var] = da

    # Do the special cases last
    for var in WOMBAT_VARS[wombat_version]:
        if var == "caco3":
            if wombat_version == "lite":
                da = ds["det"] * 0.08
            else:
                da = (ds["det"] + ds["bdet"]) * 0.08
            da.attrs["units"] = ds["det"].attrs["units"]
            da.attrs["long_name"] = var
            ds[var] = da
        elif var == "sil":
            # Convert from umol/kg to mol/kg and extrapolate over land
            da = _extrapolate_over_land(woa_Si * 1e-6)
            da.attrs["units"] = "mol kg-1"
            da.attrs["long_name"] = var
            ds[var] = da
        elif var == "n2o":
            woa_T = _open_woa_dataset(WOA23_T_FILE, "t_an", 2)
            woa_S = _open_woa_dataset(WOA23_S_FILE, "s_an", 2)
            global_attrs[
                "inputFile"
            ] += f", {WOA23_S_FILE} (md5 hash: {md5sum(WOA23_S_FILE)}), {WOA23_T_FILE} (md5 hash: {md5sum(WOA23_T_FILE)})"

            # Calculate n2o solubility as done in WOMBATmid
            # https://github.com/ACCESS-NRI/GFDL-generic-tracers/blob/19d9b3f4426ee5af30d10391622bf71503d471b7/generic_tracers/generic_WOMBATmid.F90#L8335-L8338
            a_1 = -168.2459
            a_2 = 226.0894
            a_3 = 93.2817
            a_4 = -1.48693
            b_1 = -0.060361
            b_2 = 0.033765
            b_3 = -0.0051862
            atm_n2o = 293.0e-9
            tk = 273.15 + woa_T.clip(min=0, max=40)
            sal = woa_S.clip(min=0, max=40)
            tk100 = tk / 100
            n2o_sol = np.exp(
                a_1
                + a_2 * (100 / tk)
                + a_3 * np.log(tk100)
                + a_4 * tk100**2
                + sal * (b_1 + b_2 * tk100 + b_3 * tk100**2)
            )
            da = _extrapolate_over_land(atm_n2o * n2o_sol)
            da.attrs["units"] = "mol kg-1"
            da.attrs["long_name"] = var
            ds[var] = da

    ds = xr.Dataset(ds)
    ds.attrs = global_attrs

    # MOM6 has issues with _FillValue = NaN
    var_encoding = dict(zlib=True, complevel=4, _FillValue=-1.0e10)
    for var in ds.data_vars:
        ds[var].encoding |= var_encoding
    # Coordinates should not have _FillValue
    coord_encoding = dict(_FillValue=None)
    for coord in ds.coords:
        ds[coord].encoding |= coord_encoding
    ds.to_netcdf(output_file)


if __name__ == "__main__":
    import argparse

    main()
