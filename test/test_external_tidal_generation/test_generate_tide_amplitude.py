# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.

import sys
from pathlib import Path
from subprocess import run

import numpy as np
import pytest
import xarray as xr

script_dir = Path(__file__).parents[2] / "external_tidal_generation"
sys.path.append(str(script_dir))

from generate_tide_amplitude import (
    PRIMARY_CONSTITUENTS,
    TPXO10_DATASUFFIX,
    TPXO10_GRIDFILE,
)

run_cmd = ["python3", str(script_dir / "generate_tide_amplitude.py")]

# uniform tidal velocity (m/s) for each constituent in the synthetic TPXO data
U_VEL = 0.1
V_VEL = 0.05
DEPTH = 1000.0
# expected tideamp: RMS speed over all constituents
EXPECTED = np.sqrt(len(PRIMARY_CONSTITUENTS) * (U_VEL**2 + V_VEL**2))

# TPXO land box. Edges fall mid-way through model cells so that some model cells
# partially overlap TPXO land
TPXO_LAND = {"lon": (102.5, 142.5), "lat": (-27.5, 27.5)}
# model land box, smaller than the TPXO land box so that some model ocean cells lie
# entirely over TPXO land
MODEL_LAND = {"lon": (110.0, 130.0), "lat": (-20.0, 20.0)}


def in_box(lon, lat, box):
    return (
        (lon > box["lon"][0])
        & (lon < box["lon"][1])
        & (lat > box["lat"][0])
        & (lat < box["lat"][1])
    )


@pytest.fixture
def tpxo_path(tmp_path):
    """
    Write synthetic TPXO10-atlas grid and transport files on a 2 degree global grid,
    with uniform velocities in the ocean and a rectangular continent (TPXO_LAND).
    """
    path = tmp_path / "tpxo"
    path.mkdir()

    res = 2.0
    lon_u = np.arange(0.0, 360.0 + res / 2, res)
    lon_z = lon_u + res / 2
    lat_v = np.arange(-90.0, 90.0 + res / 2, res)
    lat_z = lat_v + res / 2

    lat_u2d, lon_u2d = np.meshgrid(lat_z, lon_u, indexing="ij")
    lat_v2d, lon_v2d = np.meshgrid(lat_v, lon_z, indexing="ij")
    hu = np.where(in_box(lon_u2d, lat_u2d, TPXO_LAND), 0.0, DEPTH)
    hv = np.where(in_box(lon_v2d, lat_v2d, TPXO_LAND), 0.0, DEPTH)

    dims = ("ny", "nx")
    xr.Dataset(
        {
            "hu": (dims, hu),
            "hv": (dims, hv),
            "lon_z": ("nx", lon_z),
            "lat_z": ("ny", lat_z),
            "lon_u": ("nx", lon_u),
            "lat_v": ("ny", lat_v),
        }
    ).to_netcdf(path / TPXO10_GRIDFILE)

    # transports in cm^2/s
    for cons in PRIMARY_CONSTITUENTS:
        xr.Dataset(
            {
                "uRe": (dims, hu * U_VEL * 1e4),
                "uIm": (dims, np.zeros_like(hu)),
                "vRe": (dims, hv * V_VEL * 1e4),
                "vIm": (dims, np.zeros_like(hv)),
            }
        ).to_netcdf(path / f"u_{cons}_{TPXO10_DATASUFFIX}")

    return path


@pytest.fixture
def model_grid(tmp_path):
    """
    Write a 5 degree global MOM6 supergrid and topography with a rectangular
    continent (MODEL_LAND).
    """
    res = 5.0
    x_super = np.arange(0.0, 360.0 + res / 4, res / 2)
    y_super = np.arange(-90.0, 90.0 + res / 4, res / 2)
    y2d, x2d = np.meshgrid(y_super, x_super, indexing="ij")

    hgrid_file = tmp_path / "ocean_hgrid.nc"
    xr.Dataset({"x": (("nyp", "nxp"), x2d), "y": (("nyp", "nxp"), y2d)}).to_netcdf(
        hgrid_file
    )

    lon = x2d[1::2, 1::2]
    lat = y2d[1::2, 1::2]
    depth = np.where(in_box(lon, lat, MODEL_LAND), 0.0, DEPTH)

    topog_file = tmp_path / "topog.nc"
    xr.Dataset({"depth": (("ny", "nx"), depth)}).to_netcdf(topog_file)

    # cell edges
    corner_x = x2d[::2, ::2]
    corner_y = y2d[::2, ::2]
    edges = {
        "west": corner_x[:-1, :-1],
        "east": corner_x[:-1, 1:],
        "south": corner_y[:-1, :-1],
        "north": corner_y[1:, :-1],
    }

    return hgrid_file, topog_file, edges, depth > 0


def overlaps_box(edges, box):
    """Model cells that overlap the box"""
    return (
        (edges["east"] > box["lon"][0])
        & (edges["west"] < box["lon"][1])
        & (edges["north"] > box["lat"][0])
        & (edges["south"] < box["lat"][1])
    )


def inside_box(edges, box):
    """Model cells that lie entirely inside the box"""
    return (
        (edges["west"] >= box["lon"][0])
        & (edges["east"] <= box["lon"][1])
        & (edges["south"] >= box["lat"][0])
        & (edges["north"] <= box["lat"][1])
    )


def test_tideamp_coastal_cells(tpxo_path, model_grid, tmp_path):
    """
    Model ocean cells that overlap TPXO land should not be zero. Those partially
    overlapping TPXO ocean should be renormalised over the valid TPXO points, and
    those with no TPXO ocean overlap should be filled by nearest neighbour from
    valid TPXO points.
    """
    hgrid_file, topog_file, edges, ocean = model_grid
    output = tmp_path / "tideamp.nc"

    # check the setup includes model ocean cells that partially overlap TPXO land
    # and model ocean cells that lie entirely over TPXO land (and so need filling)
    inside = ocean & inside_box(edges, TPXO_LAND)
    partial = ocean & overlaps_box(edges, TPXO_LAND) & ~inside
    assert inside.any()
    assert partial.any()

    # run from tmp_path, as the provenance README is written to the current directory
    run(
        run_cmd
        + [
            f"--hgrid-file={hgrid_file}",
            f"--topog-file={topog_file}",
            f"--data-path={tpxo_path}",
            f"--output={output}",
        ],
        cwd=tmp_path,
        check=True,
    )

    assert output.is_file()
    assert (tmp_path / "tideamp.README.md").is_file()

    tideamp = xr.open_dataset(output)["tideamp"].isel(time=0).values

    # ocean cells take the uniform TPXO value everywhere, including near TPXO land
    assert not np.isnan(tideamp).any()
    np.testing.assert_allclose(tideamp[ocean], EXPECTED, rtol=1e-5)

    # land cells are zero
    np.testing.assert_array_equal(tideamp[~ocean], 0.0)
