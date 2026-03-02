# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Interpolate a provided forcing file to a grid specified by a provided MOM supergrid file for
# use in the MOM data_table.
#
# To run:
#   python regrid_forcing.py --forcing-filename=<path-to-forcing-file>
#     --hgrid-filename=<path-to-supergrid-file> --output-filename=<path-to-output-file>
# these extra arguments are available if required:
#     --mask-filename=<path-to-mask-file>--lon-name=<lon-name> --lat-name=<lat-name>
#
# For more information, run `python regrid_forcing.py -h`
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
#   argparse, xarray, cf_xarray, xesmf
# =========================================================================================

import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from regrid_common import Regrid_Common
from scripts_common import get_provenance_metadata, md5sum


def main():

    # Init
    regrid = Regrid_Common()

    regrid.parser.add_argument(
        "--forcing-filename",
        type=str,
        required=True,
        help="The path to the forcing file to interpolate.",
    )

    regrid.parser.description = (
        regrid.parser.description + " For use in the MOM data_table."
    )

    # Parse
    regrid.parse_cli()
    regrid.forcing_filename = os.path.abspath(regrid.args.forcing_filename)

    regrid.open_datasets()

    forcing_regrid = regrid.regrid_forcing()

    # Add metadata required by data_table
    forcing_regrid = forcing_regrid.rename({"nyp": "ny", "nxp": "nx"})
    forcing_regrid = forcing_regrid.assign_coords(
        {
            "ny": ("ny", range(forcing_regrid.sizes["ny"])),
            "nx": ("nx", range(forcing_regrid.sizes["nx"])),
        }
    )
    forcing_regrid["time"].attrs = dict(axis="T", standard_name="time", modulo="y")
    # Both axis and cartesian_axis attributes are required to work with recent (MOM6-era) and older
    # (MOM5-era) versions of FMS
    forcing_regrid["ny"].attrs = dict(axis="Y", cartesian_axis="Y")
    forcing_regrid["nx"].attrs = dict(axis="X", cartesian_axis="X")

    # Older (MOM5-era) versions of FMS can't handle integer type dimensions
    forcing_regrid["nx"].encoding |= {"dtype": "float32"}
    forcing_regrid["ny"].encoding |= {"dtype": "float32"}

    # Add some info about how the file was generated
    this_file = os.path.normpath(__file__)

    runcmd = (
        f"python3 {os.path.basename(this_file)} --forcing-filename={regrid.forcing_filename} "
        f"{regrid.runcmd_args}"
    )

    global_attrs = {
        "history": get_provenance_metadata(this_file, runcmd),
    }

    forcing_regrid.attrs = forcing_regrid.attrs | global_attrs

    regrid.save_output()


if __name__ == "__main__":

    main()
