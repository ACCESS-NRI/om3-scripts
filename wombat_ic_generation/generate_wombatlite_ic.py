# Copyright 2023 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Generate constant initial conditions for WOMBATlite tracers phy, zoo, det, caco3,
# det_sediment and caco3_sediment using a provided MOM restart file as a template.
#
# To run:
#   python generate_wombatlite_ic.py --template-file=<path-to-restart-template-file>
#       --template-varname=<name-of-variable-in-restart-template-file>
#       --output-file=<path-to-output-file>
#
# For more information, run `python generate_wombatlite_ic.py -h`
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
#   argparse, xarray
# =========================================================================================

import os
import xarray as xr

from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate constant initial conditions for WOMBATlite tracers phy, zoo, det, caco3, "
            "det_sediment and caco3_sediment using a provided MOM restart file as a template."
        )
    )

    parser.add_argument(
        "--template-file",
        required=True,
        help="The path to an existing MOM restart file to use as a template for the WOMBATlite IC file.",
    )

    parser.add_argument(
        "--template-varname",
        required=True,
        help="The name of the variable to template from in the template file.",
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help="The path to the initial condition file to be outputted.",
    )

    args = parser.parse_args()
    template_file = os.path.abspath(args.template_file)
    template_varname = args.template_varname
    output_file = args.output_file

    this_file = os.path.normpath(__file__)

    # Add some info about how the file was generated
    runcmd = (
        f"python3 {this_file} --template-file={os.path.abspath(template_file)} "
        f"--template-varname={template_varname} --output-file={os.path.abspath(output_file)}"
    )

    global_attrs = {
        "history": get_provenance_metadata(this_file, runcmd),
        "inputFile": f"{template_file} (md5 hash: {md5sum(template_file)})",
    }

    # Generate the initial conditions
    init_vars = {
        "phy": (0.01e-6, "mol kg-1"),
        "zoo": (0.01e-6, "mol kg-1"),
        "det": (0.01e-6, "mol kg-1"),
        "caco3": (0.01e-6, "mol kg-1"),
        "pchl": (0.02e-8, "mol kg-1"),
        "phyfe": (0.01e-11, "mol kg-1"),
        "zoofe": (0.01e-11, "mol kg-1"),
        "detfe": (0.01e-11, "mol kg-1"),
        "dicp": (0.0, "mol kg-1"),
        "dicr": (0.0, "mol kg-1"),
        "det_sediment": (0.0, "mol m-2"),
        "caco3_sediment": (0.0, "mol m-2"),
        "detfe_sediment": (0.0, "mol m-2"),
        "detbury": (0.0, "mol m-2 s-1"),
        "caco3bury": (0.0, "mol m-2 s-1"),
    }

    xr.set_options(keep_attrs=True)
    template = xr.open_dataset(
        template_file,
        decode_cf=False,
        decode_times=False,
    )[template_varname].compute()
    ds = {}
    for name, (const, units) in init_vars.items():
        da = 0 * template + const
        da.attrs["units"] = units
        da.attrs["long_name"] = name
        ds[name] = da
    ds = xr.Dataset(ds)
    ds.attrs = global_attrs

    ds.to_netcdf(output_file)


if __name__ == "__main__":
    import argparse

    main()
