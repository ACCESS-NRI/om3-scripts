# Copyright 2023 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate initial conditions for WOMBATlite/mid. Initial conditions are constructed as follows:
# - no3: From WOA23. Use Jan average with depths below 800m filled in from annual data
# - o2: From WOA23. Use Jan average with depths below 1500m filled in from annual data
# - sil: From WOA23. Use Jan average with depths below 800m filled in from annual data
# - n2o: Calculated from WOA23 T and S. Use Jan average with depths below 1500m filled in from
#     annual data.
# - alk: From GLODAPv2 mapped fields
# - dic: From GLODAPv2 mapped fields
# - fe: From Huang 2022 (zenodo.org/records/6994318)
# All other variables are initialised as constants on the WOA23 no3 1deg grid or derived from
# the above variables.
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
#   argparse, xarray, numpy, scipy, regionmask
# =========================================================================================

import os
import numpy as np
import xarray as xr

from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from regrid_common import fill_ocean_horiz
from scripts_common import get_provenance_metadata, md5sum

WOA23_MON_PATH = "/g/data/av17/access-nri/OM3/woa23"
WOA23_ANN_PATH = "/g/data/av17/access-nri/OM3/woa23/annual_files/corrected_times"
GLODAP_PATH = "/g/data/av17/access-nri/OM3/GLODAPv2.2016b_MappedClimatologies"
HUANG_PATH = "/g/data/av17/access-nri/OM3/huang-2022-dFe"


def _open_woa_data(file, var, global_attrs, hcoord_int=None, vcoord_int=None):
    """
    Opens a WOA23 dataset and returns the specified variable as a DataArray with
    some attributes needed by MOM6 and renamed lon/lat coords to allow having
    different coordinates in one dataset.
    """
    da = (
        xr.open_dataset(
            file,
            decode_times=False,
        )[var]
        .squeeze(drop=True)
        .compute()
    )

    # Drop some attrs
    del da.attrs["cell_methods"]
    for coord in da.coords:
        del da[coord].attrs["bounds"]
    da.attrs["source"] = (
        "WOA23 (https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/)"
    )

    global_attrs["inputFile"].add(f"{file} (md5 hash: {md5sum(file)})")

    return _standardise_coords(da, hcoord_int, vcoord_int)


def _open_glodap_data(file, var, global_attrs, hcoord_int=None, vcoord_int=None):
    """
    Opens a GLODAPv2 dataset and returns the specified variable as a DataArray with
    some attributes needed by MOM6 and renamed lon/lat coords to allow having
    different coordinates in one dataset.
    """
    ds = xr.open_dataset(file).compute()

    # Assign depth coordinates
    da = ds.assign_coords({"depth_surface": ds.Depth})[var].rename(
        {"depth_surface": "depth"}
    )
    da.attrs["source"] = (
        "GLODAPv2 mapped data (https://glodap.info/index.php/mapped-data-product/)"
    )

    global_attrs["inputFile"].add(f"{file} (md5 hash: {md5sum(file)})")

    return _standardise_coords(da, hcoord_int, vcoord_int)


def _open_huang_data(file, global_attrs, hcoord_int=None, vcoord_int=None):
    """
    Opens the Huang et al 2022 dFe dataset and returns the dFe variable as a DataArray
    with some attributes needed by MOM6 and renamed lon/lat coords to allow having
    different coordinates in one dataset.
    """
    da = xr.open_dataset(file)["dFe_RF"].sel(Month=1, drop=True).compute()
    da = da.rename({"Longitude": "lon", "Latitude": "lat", "Depth": "depth"})
    da.attrs["source"] = (
        "Huang et al 2022 dFe data (https://zenodo.org/records/6994318)"
    )

    global_attrs["inputFile"].add(f"{file} (md5 hash: {md5sum(file)})")

    # Extend Latitudes from -90-90
    new_lats = np.concatenate(
        [
            np.arange(-90, da.lat.min(), 1.0),
            da.lat.values,
            np.arange(da.lat.max() + 1.0, 90.1, 1.0),
        ]
    )
    da = da.reindex(lat=new_lats)

    return _standardise_coords(da, hcoord_int, vcoord_int)


def _standardise_coords(da, hcoord_int, vcoord_int):
    """Rename coordinates by appending the provided integers to the name"""
    # Ensure each coordinate has attrs required by MOM6
    da.attrs.pop("standard_name", None)
    da.attrs.pop("grid_mapping", None)
    da.lon.attrs["standard_name"] = "longitude"
    da.lon.attrs["units"] = "degrees_east"
    da.lon.attrs["axis"] = "X"
    da.lon.attrs["modulo"] = 360.0
    da.lat.attrs["standard_name"] = "latitude"
    da.lat.attrs["units"] = "degrees_north"
    da.lat.attrs["axis"] = "Y"
    da.depth.attrs["standard_name"] = "depth"
    da.depth.attrs["positive"] = "down"
    da.depth.attrs["axis"] = "Z"

    if hcoord_int is not None:
        da = da.rename({"lon": f"lon{hcoord_int}", "lat": f"lat{hcoord_int}"})
    if vcoord_int is not None:
        da = da.rename({"depth": f"depth{vcoord_int}"})
    return da


def _fill_woa_at_depth(data_mon, data_ann, depth):
    """Use annual data to fill monthly data below a certain depth level"""
    return xr.concat(
        [
            data_mon.where(data_mon.depth <= depth, drop=True),
            data_ann.where(data_ann.depth > depth, drop=True),
        ],
        dim="depth",
    )


def _fill_ocean_all_levels(da, n_erode):
    """
    Fill all depths levels using harmonic interpolation (via fill_ocean_horiz)
    """
    # Get name of depth coordinate by looking for coordinate with axis attribute "Z"
    depth_coord = next(
        (name for name, coord in da.coords.items() if coord.attrs.get("axis") == "Z"),
        None,
    )

    da_filled = []
    for depth in da[depth_coord]:
        da_filled.append(
            fill_ocean_horiz(
                da.sel({depth_coord: depth}), top_bound="regular", n_erode=n_erode
            )
        )
    return xr.concat(da_filled, dim=depth_coord)


def _make_no3(global_attrs):
    """Make the no3 initial conditions using WOA23 NO3"""
    da_jan = _open_woa_data(
        f"{WOA23_MON_PATH}/woa23_all_n01_01.nc", "n_an", global_attrs
    )
    da_ann = _open_woa_data(
        f"{WOA23_ANN_PATH}/woa23_all_n00_01.nc", "n_an", global_attrs
    )

    # Fill depth from annual data
    da = _fill_woa_at_depth(da_jan, da_ann, 800)

    # Convert to mol/kg
    da = 1e-6 * da
    da.attrs["units"] = "mol/kg"

    return _fill_ocean_all_levels(da, n_erode=6).rename("no3")


def _make_o2(global_attrs):
    """Make the o2 initial conditions using WOA23 O2"""
    da_jan = _open_woa_data(
        f"{WOA23_MON_PATH}/woa23_all_o01_01.nc", "o_an", global_attrs
    )
    da_ann = _open_woa_data(
        f"{WOA23_ANN_PATH}/woa23_all_o00_01.nc", "o_an", global_attrs
    )

    # Fill depth from annual data
    da = _fill_woa_at_depth(da_jan, da_ann, 1500)

    # Convert to mol/kg
    da = 1e-6 * da
    da.attrs["units"] = "mol/kg"

    return _fill_ocean_all_levels(da, n_erode=6).rename("o2")


def _make_alk(global_attrs):
    """Make the alk initial conditions using GLODAPv2 TAlk"""
    da = _open_glodap_data(
        f"{GLODAP_PATH}/GLODAPv2.2016b.TAlk.nc",
        "TAlk",
        global_attrs,
        hcoord_int=1,
        vcoord_int=1,
    )

    # Convert to mol/kg
    da = 1e-6 * da
    da.attrs["units"] = "mol/kg"

    return _fill_ocean_all_levels(da, n_erode=6).rename("alk")


def _make_dic(global_attrs):
    """Make the dic initial conditions using GLODAPv2 TCO2"""
    da = _open_glodap_data(
        f"{GLODAP_PATH}/GLODAPv2.2016b.TCO2.nc",
        "TCO2",
        global_attrs,
        hcoord_int=1,
        vcoord_int=1,
    )

    # Convert to mol/kg
    da = 1e-6 * da
    da.attrs["units"] = "mol/kg"

    return _fill_ocean_all_levels(da, n_erode=6).rename("dic")


def _make_fe(global_attrs):
    """Make the fe initial conditions using Huang 2022 dFe"""
    da = _open_huang_data(
        f"{HUANG_PATH}/Monthly_dFe_V2.nc", global_attrs, hcoord_int=2, vcoord_int=2
    )

    # Convert to mol/kg
    da = 1e-9 / 1.025 * da
    da.attrs["units"] = "mol/kg"

    return _fill_ocean_all_levels(da, n_erode=7).rename("fe")


def _make_sil(global_attrs):
    """Make the sil initial conditions using WOA23 silica"""
    da_jan = _open_woa_data(
        f"{WOA23_MON_PATH}/woa23_all_i01_01.nc", "i_an", global_attrs
    )
    da_ann = _open_woa_data(
        f"{WOA23_ANN_PATH}/woa23_all_i00_01.nc", "i_an", global_attrs
    )

    # Fill depth from annual data
    da = _fill_woa_at_depth(da_jan, da_ann, 800)

    # Convert to mol/kg
    da = 1e-6 * da
    da.attrs["units"] = "mol/kg"

    return _fill_ocean_all_levels(da, n_erode=6).rename("sil")


def _make_n2o(global_attrs):
    """Make the n2o initial conditions using WOA23 T and S"""
    t_jan = _open_woa_data(
        f"{WOA23_MON_PATH}/woa23_decav_t01_04.nc", "t_an", global_attrs, hcoord_int=3
    )
    t_ann = _open_woa_data(
        f"{WOA23_ANN_PATH}/woa23_decav_t00_04.nc", "t_an", global_attrs, hcoord_int=3
    )
    s_jan = _open_woa_data(
        f"{WOA23_MON_PATH}/woa23_decav_s01_04.nc", "s_an", global_attrs, hcoord_int=3
    )
    s_ann = _open_woa_data(
        f"{WOA23_ANN_PATH}/woa23_decav_s00_04.nc", "s_an", global_attrs, hcoord_int=3
    )

    # Fill depth from annual data
    t = _fill_woa_at_depth(t_jan, t_ann, 1500)
    s = _fill_woa_at_depth(s_jan, s_ann, 1500)

    # Calculate n2o solubility as done in WOMBATmid
    # https://github.com/ACCESS-NRI/GFDL-generic-tracers/blob/19d9b3f4426ee5af30d10391622bf71503d471b7/generic_tracers/generic_WOMBATmid.F90#L8335-L8338
    a = (-168.2459, 226.0894, 93.2817, -1.48693)
    b = (-0.060361, 0.033765, -0.0051862)
    atm_n2o = 293.0e-9
    tk = 273.15 + t.clip(min=0, max=40)
    sal = s.clip(min=0, max=40)
    tk100 = tk / 100
    n2o_sol = np.exp(
        a[0]
        + a[1] * (100 / tk)
        + a[2] * np.log(tk100)
        + a[3] * tk100**2
        + sal * (b[0] + b[1] * tk100 + b[2] * tk100**2)
    )
    n2o = atm_n2o * n2o_sol
    n2o.attrs["long_name"] = "Nitrous oxide, calculated from WOA23 T and S"
    n2o.attrs["units"] = "mol/kg"

    return _fill_ocean_all_levels(n2o, n_erode=20).rename("n2o")


WOMBATLITE_VARS = [
    "no3",
    "o2",
    "dic",
    "alk",
    "fe",
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
    "nh4": (0.0, "mol/kg"),
    "dia": (0.1e-6, "mol/kg"),
    "mes": (0.1e-6, "mol/kg"),
    "phy": (0.1e-6, "mol/kg"),
    "zoo": (0.1e-6, "mol/kg"),
    "det": (0.1e-6, "mol/kg"),
    "bdet": (0.1e-6, "mol/kg"),
    "doc": (5.0e-6, "mol/kg"),
    "dicp": (0.0, "mol/kg"),
    "dicr": (0.0, "mol/kg"),
    "nosdoc": (0.5, "1"),
    "bac1": (0.1e-6, "mol/kg"),
    "bac2": (0.1e-6, "mol/kg"),
    "aoa": (0.1e-6, "mol/kg"),
    "det_sediment": (0.0, "mol/m2"),
    "caco3_sediment": (0.0, "mol/m2"),
    "detfe_sediment": (0.0, "mol/m2"),
    "detsi_sediment": (0.0, "mol/m2"),
    "detbury": (0.0, "mol/m2"),
    "caco3bury": (0.0, "mol/m2"),
}
# "var": function_to_calculate_var
SPATIAL_VARS = {
    "no3": _make_no3,
    "o2": _make_o2,
    "alk": _make_alk,
    "dic": _make_dic,
    "fe": _make_fe,
    "sil": _make_sil,
    "n2o": _make_n2o,
}
# Linear combinations of other variables
DERIVED_VARS = {
    "pchl": {"default": {"phy": 0.004}},
    "phyfe": {"default": {"phy": 7e-6}},
    "dchl": {"default": {"dia": 0.004}},
    "diafe": {"default": {"dia": 7e-6}},
    "diasi": {"default": {"dia": 16 / 122}},
    "zoofe": {"default": {"zoo": 7e-6}},
    "mesfe": {"default": {"mes": 7e-6}},
    "detfe": {"default": {"det": 7e-6}},
    "bdetfe": {"default": {"bdet": 7e-6}},
    "bdetsi": {"default": {"bdet": 16 / 122}},
    "don": {"default": {"doc": 16 / 122}},
    "afe": {"default": {"fe": 0.1}},
    "bafe": {"default": {"fe": 0.1}},
    "caco3": {
        "lite": {"det": 0.08},
        "mid": {"det": 0.08, "bdet": 0.08},
    },
}


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

    # Obtain metadata
    this_file = sys.argv[0]
    runcmd = f"{sys.executable} {' '.join(sys.argv)}"

    global_attrs = {
        "history": get_provenance_metadata(this_file, runcmd),
        "inputFile": set(),
    }

    xr.set_options(keep_attrs=True)
    template = _open_woa_data(
        f"{WOA23_MON_PATH}/woa23_all_n01_01.nc", "n_an", global_attrs
    )

    # Do constant and spatial variables first, so that they can be used in the calculation
    # of derived variables
    ds = {}
    for var in WOMBAT_VARS[wombat_version]:
        if var in CONSTANT_VARS:
            const, units = CONSTANT_VARS[var]
            da = xr.full_like(template, fill_value=const)
            da.attrs["units"] = units
            da.attrs["long_name"] = var
            ds[var] = da
        elif var in SPATIAL_VARS:
            ds[var] = SPATIAL_VARS[var](global_attrs)

    # Do derived variables
    for var in WOMBAT_VARS[wombat_version]:
        if var in DERIVED_VARS:
            terms = (
                DERIVED_VARS[var].get(wombat_version) or DERIVED_VARS[var]["default"]
            )
            da = sum(ds[src] * scale for src, scale in terms.items())
            da.attrs |= {
                "units": ds[next(iter(terms))].attrs["units"],
                "long_name": var,
            }
            ds[var] = da

    ds = xr.Dataset(ds)
    global_attrs["inputFile"] = ", ".join(sorted(global_attrs["inputFile"]))
    global_attrs["title"] = (
        f"Initial conditions for WOMBAT{wombat_version} in ACCESS-OM3"
    )
    global_attrs["Conventions"] = "CF-1.11"
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
