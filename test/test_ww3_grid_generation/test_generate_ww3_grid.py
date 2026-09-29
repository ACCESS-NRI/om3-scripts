# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.

from pathlib import Path
from subprocess import run

import numpy as np
import pytest
import regionmask
import xarray as xr

scripts_base = Path(__file__).parents[2]
run_cmd = [
    "python3",
    str(scripts_base / "ww3_grid_generation/generate_ww3_grid.py"),
]

# 4 degrees is the lowest resolution tested in ocean_model_grid_generator
_test_resolution = 4
GRIDNAME = "test_grid"
SUFFIXES = [".Lat", ".Lon", ".Dpt", ".Mask", ".Obstr"]


@pytest.fixture
def input_files(tmp_path):
    """
    Generate a tripole MOM supergrid (ocean_hgrid.nc) and a matching topography
    (topog.nc), with land taken from a rough regionmask landmask
    """
    grid_path = tmp_path / "ocean_hgrid.nc"
    topog_path = tmp_path / "topog.nc"

    run(
        [
            "ocean_grid_generator.py",
            "-r",
            str(1 / _test_resolution),
            "--no_south_cap",
            "--ensure_nj_even",
            "-f",
            grid_path,
        ],
        check=True,
    )

    ds = xr.open_dataset(grid_path)

    # h-points (cell centres) are every second point of the supergrid
    x_centres = ds.x[1::2, 1::2].values
    y_centres = ds.y[1::2, 1::2].values

    # Generate a rough landmask
    mask = regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50.mask(
        x_centres, y_centres
    ).notnull()

    # topog.nc stores land as a fill value
    depth = xr.DataArray(
        np.where(mask, 1000.0, np.nan), dims=("ny", "nx"), name="depth"
    )
    depth.to_netcdf(topog_path, encoding={"depth": {"_FillValue": -1e20}})

    return grid_path, topog_path


def test_generate_ww3_grid(input_files, tmp_path):
    """Check the script runs and writes the WW3 grid files and README"""
    grid_path, topog_path = input_files
    output_dir = tmp_path / "output"

    run(
        run_cmd
        + [
            f"--grid-filename={grid_path}",
            f"--topog-filename={topog_path}",
            f"--gridname={GRIDNAME}",
            f"--output-dir={output_dir}",
        ],
        check=True,
    )

    for suffix in SUFFIXES:
        assert (output_dir / (GRIDNAME + suffix)).is_file()
    assert (output_dir / (GRIDNAME + ".README.md")).is_file()
