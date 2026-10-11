# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Calculate monthly scale factors for Antarctic frozen runoff (Forr_rofi), from the seasonal
# cycle of the Mankoff 2025 Antarctic iceberg melt climatology. The factors are for the
# rofi_scale_sh option in the CDEPS drof_nml namelist, which scales the Forr_rofi provided
# by DROF in the Southern Hemisphere to add a seasonal cycle. JRA55-do frozen runoff is
# constant in time around Antarctica, but already has a seasonal cycle around Greenland, so
# rofi_scale_nh is not set.
#
# The melt climatology is summed over all calving regions and integrated over area to give
# a total for each month.
#
# To run:
#   python calculate_rofi_scale.py
#
# Contact:
#   Anton Steketee <anton.steketee@anu.edu.au>
#
# Dependencies:
#   xarray, numpy, xesmf
# =========================================================================================

import os
import sys
from pathlib import Path

import xarray as xr
from xesmf.util import cell_area

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_git_url

# source data, see https://doi.org/10.5194/gmd-18-8333-2025
AQ_MELT_PATTERN = "/g/data/av17/access-nri/OM3/Mankoff_2025_V11/AQ_iceberg_melt.nc"


def monthly_scale_factors(melt_file):
    """
    For an iceberg melt climatology file, return the 12 monthly totals over all regions
    and space, divided by their mean
    """
    ds = xr.open_dataset(melt_file)

    monthly_total = (ds["melt"].fillna(0).sum("region") * cell_area(ds)).sum(
        ("latitude", "longitude")
    )

    return (monthly_total / monthly_total.mean()).values


def namelist_snippet(scale_sh, script_url):
    """
    Format the scale factors as a drof_nml namelist snippet, with the url of this script
    as a comment
    """
    return (
        f"! scale factors calculated using {script_url}\n"
        f"  rofi_scale_sh = {', '.join(f'{x:.3g}' for x in scale_sh)}\n"
        f"  rofi_scale_normalise = .true.\n"
    )


def main():
    scale_sh = monthly_scale_factors(AQ_MELT_PATTERN)

    script_url = get_git_url(os.path.abspath(__file__))

    print(namelist_snippet(scale_sh, script_url))


if __name__ == "__main__":

    main()
