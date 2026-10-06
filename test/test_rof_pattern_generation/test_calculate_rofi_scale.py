# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(Path(__file__).parents[2] / "rof_pattern_generation"))

from calculate_rofi_scale import monthly_scale_factors, namelist_snippet

# monthly cycle to recover, mean of 1
CYCLE = np.array([2.0, 1.8, 1.4, 1.0, 0.6, 0.4, 0.3, 0.3, 0.4, 0.8, 1.3, 1.7])
CYCLE = CYCLE / CYCLE.mean()


@pytest.fixture
def melt_file(tmp_path):
    """
    Write a synthetic iceberg melt climatology on a 2 degree grid, with two regions
    south of 50S. Region 2 has twice the melt of region 1, and the total for each month
    follows CYCLE.
    """
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(-179.0, 180.0, 2.0)
    in_range = (lat > -80) & (lat < -50)

    melt = np.full((2, 12, lat.size, lon.size), np.nan)
    for region, amount in enumerate([1.0, 2.0]):
        for month in range(12):
            melt[region, month][in_range, :] = amount * CYCLE[month]

    path = tmp_path / "AQ_iceberg_melt.nc"
    xr.Dataset(
        {"melt": (("region", "time", "latitude", "longitude"), melt)},
        coords={
            "region": [1, 2],
            "time": np.arange(1, 13),
            "latitude": lat,
            "longitude": lon,
        },
    ).to_netcdf(path)
    return path


def test_monthly_scale_factors(melt_file):
    np.testing.assert_allclose(monthly_scale_factors(melt_file), CYCLE, rtol=1e-10)


def test_namelist_snippet():
    snippet = namelist_snippet(CYCLE, "https://github.com/x")

    values = {}
    for line in snippet.splitlines():
        name, _, rhs = line.partition("=")
        if name.strip() == "rofi_scale_sh":
            values[name.strip()] = np.array([float(x) for x in rhs.split(",")])

    np.testing.assert_allclose(values["rofi_scale_sh"], CYCLE, rtol=5e-3)
    assert "rofi_scale_nh" not in snippet
    assert snippet.startswith("! scale factors calculated using https://github.com/x\n")
    assert "rofi_scale_normalise = .true." in snippet
