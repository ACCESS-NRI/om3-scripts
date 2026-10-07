# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.

import pytest

import esmpy
import numpy as np
import xarray as xr

from mesh_generation.generate_um_mesh import create_um_mesh, parse_um_resolution


@pytest.mark.parametrize(
    "atm, expected",
    [
        ("n96", (144, 192)),
        ("n96e", (144, 192)),
        ("n48", (72, 96)),
    ],
)
def test_parse_um_resolution(atm, expected):
    assert parse_um_resolution(atm) == expected


@pytest.mark.parametrize("atm", ["n97", "bogus", "n96x", ""])
def test_parse_um_resolution_invalid(atm):
    with pytest.raises(ValueError):
        parse_um_resolution(atm)


@pytest.fixture(scope="module")
def um_mesh_file(tmp_path_factory):
    """A small UM mesh, and the (nlat, nlon) used to create it"""
    nlat, nlon = 6, 8
    mesh_filename = tmp_path_factory.mktemp("um_mesh") / "um_mesh.nc"
    create_um_mesh(nlat, nlon, str(mesh_filename))
    return mesh_filename, nlat, nlon


def test_create_um_mesh_dims(um_mesh_file):
    mesh_filename, nlat, nlon = um_mesh_file
    ds = xr.open_dataset(mesh_filename)

    assert ds.sizes["elementCount"] == nlat * nlon
    assert ds.sizes["nodeCount"] == nlon * (nlat - 1) + 2


def test_create_um_mesh_element_types(um_mesh_file):
    mesh_filename, nlat, nlon = um_mesh_file
    ds = xr.open_dataset(mesh_filename)

    # poles are fans of triangles (3 nodes per element)
    assert (ds.numElementConn.values[:nlon] == 3).all()
    assert (ds.numElementConn.values[-nlon:] == 3).all()
    # everywhere else is quads (4 nodes per element)
    assert (ds.numElementConn.values[nlon:-nlon] == 4).all()


def test_create_um_mesh_total_area(um_mesh_file):
    mesh_filename, nlat, nlon = um_mesh_file
    ds = xr.open_dataset(mesh_filename)

    # sum of element areas (steradians) should be the surface area of the
    # whole sphere (4*pi steradians)
    assert ds.elementArea.values.sum() == pytest.approx(4 * np.pi)


def test_create_um_mesh_coords_in_range(um_mesh_file):
    mesh_filename, nlat, nlon = um_mesh_file
    ds = xr.open_dataset(mesh_filename)

    assert ds.centerCoords.values[:, 0].min() >= 0
    assert ds.centerCoords.values[:, 0].max() < 360
    assert ds.centerCoords.values[:, 1].min() >= -90
    assert ds.centerCoords.values[:, 1].max() <= 90
    assert ds.nodeCoords.values[:, 1].min() == -90
    assert ds.nodeCoords.values[:, 1].max() == 90


def test_create_um_mesh_esmpy_loadable(um_mesh_file):
    mesh_filename, nlat, nlon = um_mesh_file
    mesh = esmpy.Mesh(filename=str(mesh_filename), filetype=esmpy.FileFormat.ESMFMESH)

    fld = esmpy.Field(mesh, meshloc=esmpy.MeshLoc.ELEMENT)
    assert fld.data.size == nlat * nlon
