# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.

import pytest
from unittest.mock import patch
from mesh_generation.generate_mesh import mom6_mask_detection, MomSuperGrid
from pathlib import Path
from subprocess import run
import xarray as xr
import numpy as np
import esmpy

# create test grids at 4 degrees and 0.1 degrees
# 4 degress is the lowest tested in ocean_model_grid_generator
# going higher resolution than 0.1 has too much computational cost
_test_resolutions = [4]  # , 0.1]


# so that our fixtures are only created once in this pytest module, we need this special version of 'tmp_path'
@pytest.fixture(scope="module")
def tmp_path(tmp_path_factory: pytest.TempdirFactory) -> Path:
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


# pytest doesn't support class fixtures, so we need this constructor
@pytest.fixture(scope="module", params=_test_resolutions)
def mom_grid(request, tmp_path):
    return MomGridFixture(request.param, tmp_path)


@pytest.fixture()
def esmf_mask_mesh(mom_grid):
    """
    Patch the mom6_mask_detection function, and just return the mask from the mom_grid fixture
    """
    with patch(
        "mesh_generation.generate_mesh.mom6_mask_detection",
        return_value=mom_grid.mask_ds.mask.values,
    ):
        test_mesh = MomSuperGrid(mom_grid.path, topog_filename=mom_grid.mask_path)

    test_mesh.create_mesh()

    test_mesh.write(str(tmp_path) + "/esmf_mask_mesh.nc")

    return test_mesh


@pytest.fixture
def weights_file(esmf_mask_mesh, mom_grid):
    drof_remapping_weights(
        str(tmp_path) + "/esmf_mask_mesh.nc",
        str(tmp_path) + "/drof_remap_weights.nc",
        len(mom_grid.ds.n1) / 2,
        len(mom_grid.ds.ny) / 2,
    )

    # check str(tmp_path) + "/drof_remap_weights.nc" exists ?

    return str(tmp_path) + "/drof_remap_weights.nc"


def test_generate_mask(esmf_mask_mesh, mom_grid):
    """
    This test just convinces us we can patch the mom6_mask_detection needing a
    topography file, and instead provide an arbitrary mask when making a mesh
    """

    assert np.all(esmf_mask_mesh.mask == mom_grid.mask_ds.mask.values.flatten())

    assert (
        len(esmf_mask_mesh.mesh.elementCount.values)
        == len(mom_grid.ds.ny) / 2 * len(mom_grid.ds.nx) / 2
    )


# def test_generate_rof_weights(esmf_mask_mesh):


@pytest.mark.parametrize("data", [1, 0])
def test_regrid_conservation(data, esmf_mask_mesh, weights_file):

    mesh_filename_in = str(tmp_path) + "/esmf_mask_mesh.nc"
    mesh_filename_out = str(tmp_path) + "/esmf_mask_mesh.nc"

    model_mesh_in = esmpy.Mesh(
        filename=mesh_filename_in,
        filetype=esmpy.FileFormat.ESMFMESH,
    )

    fld_in = esmpy.Field(model_mesh_in, meshloc=esmpy.MeshLoc.ELEMENT)

    fld_in.get_area()
    area_in = copy(fld_in.data)

    model_mesh_out = esmpy.Mesh(
        filename=mesh_filename_out,
        filetype=esmpy.FileFormat.ESMFMESH,
    )

    fld_out = esmpy.Field(model_mesh_out, meshloc=esmpy.MeshLoc.ELEMENT)

    fld_out.get_area()
    area_out = copy(fld_out.data)

    fld_in.data[:] = data

    # for unclear reasons, we need to zero the output field before populating it
    # it looks like cells which are not destinations in remapping can introduce rounding error

    fld_out.data[:] = 0

    esmpy.RegridFromFile(fld_in, fld_out, filename=weights_file)

    print(f"Total before Regrid : {np.sum(fld_in.data*area_in)}")
    print(f"Total after Regrid : {np.sum(fld_out.data*area_out)}")

    assert np.sum(fld_in.data * area_in) == pytest.approx(
        np.sum(fld_out.data * area_out), rel=1e-7
    )
