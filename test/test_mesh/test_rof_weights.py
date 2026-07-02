# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.

import pytest
from unittest.mock import patch
from pathlib import Path
from subprocess import run

import xarray as xr
import numpy as np
import esmpy
from copy import copy

from mesh_generation.generate_mesh import mom6_mask_detection, MomSuperGrid
from mesh_generation.generate_rof_weights import drof_remapping_weights

# create test grids at 4 degrees and 0.1 degrees
# 4 degress is the lowest tested in ocean_model_grid_generator
# going higher resolution than 0.1 has too much computational cost
_test_resolutions = [4, 0.1]


# so that our fixtures are only created once in this pytest module, we need this special version of 'tmp_path'
@pytest.fixture(scope="module")
def grid_path(tmp_path_factory: pytest.TempdirFactory) -> Path:
    return tmp_path_factory.mktemp("temp")


# ----------------
# test data:
class MomGridFixture:
    """Generate a sample tripole grid to use as test data"""

    def __init__(self, res, tmp_path):
        self.path = str(tmp_path) + "/ocean_hgrid.nc"
        self.mask_path = str(tmp_path) + "/ocean_mask.nc"

        # generate a tripolar grid as test data
        run(
            [
                "ocean_grid_generator.py",
                "-r",
                str(1 / res),
                "--no_south_cap",
                "--ensure_nj_even",
                "-f",
                self.path,
            ]
        )

        # open ocean_hgrid.nc
        self.ds = xr.open_dataset(self.path)

        # an ocean mask with a arbitrary mask
        self.mask_ds = xr.Dataset()
        self.mask_ds["mask"] = (
            self.ds.area.coarsen(ny=2).sum().coarsen(nx=2).sum()
        ) > 5e9
        self.mask_ds.to_netcdf(self.mask_path)

        self.ny = int(len(self.ds.ny) / 2)
        self.nx = int(len(self.ds.nx) / 2)


# pytest doesn't support class fixtures, so we need these constructor
@pytest.fixture(scope="module", params=_test_resolutions)
def mom_grid(request, grid_path):
    return MomGridFixture(request.param, grid_path)


def mesh_creator(filename):
    """
    Create esmpy flield object from a mesh file
    """
    mesh = esmpy.Mesh(
        filename=filename,
        filetype=esmpy.FileFormat.ESMFMESH,
    )

    fld = esmpy.Field(mesh, meshloc=esmpy.MeshLoc.ELEMENT)

    fld.get_area()
    area = copy(fld.data)

    return {"fld": fld, "area": area}


@pytest.fixture()
def mesh_in(mom_grid, tmp_path):
    """
    For the input mesh, use an unmasked mesh
    """
    mesh_filename_in = str(tmp_path) + "/mesh_in.nc"

    test_mesh = MomSuperGrid(mom_grid.path, topog_filename=None)
    test_mesh.create_mesh()
    test_mesh.write(mesh_filename_in)

    result = mesh_creator(mesh_filename_in)
    result["mom_super_grid"] = test_mesh

    return result


@pytest.fixture()
def mesh_out(mom_grid, tmp_path):
    """
    Patch the mom6_mask_detection function, and use the mask from the mom_grid fixture,
    rather than need a topog file
    """
    mesh_filename_out = str(tmp_path) + "/mesh_out.nc"

    with patch(
        "mesh_generation.generate_mesh.mom6_mask_detection",
        return_value=mom_grid.mask_ds.mask.values,
    ):
        test_mesh = MomSuperGrid(mom_grid.path, topog_filename=mom_grid.mask_path)

    test_mesh.create_mesh()
    test_mesh.write(mesh_filename_out)

    result = mesh_creator(mesh_filename_out)
    result["mom_super_grid"] = test_mesh

    return result


@pytest.fixture
def weights_file(mesh_out, mom_grid, tmp_path):

    drof_remapping_weights(
        str(tmp_path) + "/mesh_out.nc",
        str(tmp_path) + "/drof_remap_weights.nc",
        mom_grid.nx,
        mom_grid.ny,
    )

    if not Path(str(tmp_path) + "/drof_remap_weights.nc").exists():
        raise RuntimeError("drof remap weights not created")

    return str(tmp_path) + "/drof_remap_weights.nc"


# ----------------
# the actual tests:


def test_generate_mask(mesh_out, mom_grid):
    """
    This test just convinces us the patch mom6_mask_detection in mesh_out works
    """

    assert np.all(
        mesh_out["mom_super_grid"].mask == mom_grid.mask_ds.mask.values.flatten()
    )

    assert len(mesh_out["mom_super_grid"].mesh.elementCount.values) == (
        mom_grid.ny * mom_grid.nx
    )


@pytest.mark.parametrize("data", ["All", "None", "Ocean", "Land"])
def test_regrid_conservation(data, mesh_in, mesh_out, weights_file, tmp_path):
    """
    For some provided meshes, and weights file, confirm that the weights are conservative
    """

    fld_in = mesh_in["fld"]
    area_in = mesh_in["area"]
    fld_out = mesh_out["fld"]
    area_out = mesh_out["area"]

    match data:
        case "All":
            fld_in.data[:] = 1e10
        case "None":
            fld_in.data[:] = 0
        case "Ocean":
            fld_in.data[:] = mesh_out["mom_super_grid"].mesh.elementMask * 1e20
        case "Land":
            fld_in.data[:] = 1e-20 * (
                mesh_out["mom_super_grid"].mesh.elementMask == 0
            ).astype(int)

    # for unclear reasons, we need to zero the output field before populating it
    # it looks like cells which are not destinations in remapping can introduce rounding error
    fld_out.data[:] = 0

    esmpy.RegridFromFile(fld_in, fld_out, filename=weights_file)

    print(f"Total before Regrid : {np.sum(fld_in.data*area_in)}")
    print(f"Total after Regrid : {np.sum(fld_out.data*area_out)}")

    assert np.sum(fld_in.data * area_in) == pytest.approx(
        np.sum(fld_out.data * area_out), rel=1e-7
    )
