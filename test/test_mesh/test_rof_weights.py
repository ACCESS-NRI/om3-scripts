# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.

import pytest
from pathlib import Path
from subprocess import run

from matplotlib import pyplot as plt

import xarray as xr
import numpy as np
import esmpy
import regionmask
from scipy.ndimage import binary_dilation
from copy import copy

from mesh_generation.generate_mesh import MomSuperGrid
from mesh_generation.generate_rof_weights import gen_rof_weights, Rof_Remapping_Weights

# create test grids at 4 degrees and 1 degrees
# 4 degress is the lowest tested in ocean_model_grid_generator
# going higher resolution than 1 has too much computational cost (aka crashed the github runner)
_test_resolutions = [4]


# so that our fixtures are only created once in this pytest module, we need this special version of 'tmp_path'
@pytest.fixture(scope="module")
def mod_tmp_path(tmp_path_factory: pytest.TempdirFactory) -> Path:
    return tmp_path_factory.mktemp("temp")


# ----------------
# test data:
class MomGridFixture:
    """Generate a sample tripole grid to use as test data"""

    def __init__(self, res, tmp_path: Path):
        self.path = tmp_path / "ocean_hgrid.nc"

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
        self.ds = xr.open_dataset(str(self.path))

        self.ny = int(len(self.ds.ny) / 2)
        self.nx = int(len(self.ds.nx) / 2)

    def destroy(self):

        self.path.unlink()


# pytest doesn't support class fixtures, so we need these constructor
@pytest.fixture(scope="module", params=_test_resolutions)
def mom_grid(request, mod_tmp_path):

    grid = MomGridFixture(request.param, mod_tmp_path)

    yield grid

    grid.destroy()


def mesh_creator(filename: str):
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

    return {"fld": fld, "area": area, "path": filename}


@pytest.fixture(scope="module")
def mesh_in(mom_grid, mod_tmp_path: Path):
    """
    For the input mesh, use an unmasked mesh
    """
    mesh_in_path = mod_tmp_path / "mesh_in.nc"

    test_grid = MomSuperGrid(mom_grid.path, topog_filename=None)
    test_grid.create_mesh()
    test_grid.write(str(mesh_in_path))

    result = mesh_creator(str(mesh_in_path))
    result["mom_super_grid"] = test_grid

    yield result

    mesh_in_path.unlink()


@pytest.fixture(scope="module")
def mesh_out(mom_grid, mod_tmp_path: Path):
    """
    For the output mesh, make the Grid object without a mask, then splice in a mask
    from regionmask package
    """

    mesh_out_path = mod_tmp_path / "mesh_out.nc"

    test_grid = MomSuperGrid(mom_grid.path)

    # Generate a rough landmask
    mask = regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50.mask(
        np.reshape(test_grid.x_centres, (mom_grid.ny, mom_grid.nx)),
        np.reshape(test_grid.y_centres, (mom_grid.ny, mom_grid.nx)),
    ).notnull()
    test_grid.mask = mask.astype(np.int8).values.flatten()

    # Save a plot of the landmask for debugging
    plt.figure()
    plt.pcolor(mask)
    plt.colorbar()
    plt.title("Test mask")
    plt.savefig("test_mask.png")

    test_grid.create_mesh()
    test_grid.write(str(mesh_out_path))

    result = mesh_creator(str(mesh_out_path))
    result["mom_super_grid"] = test_grid

    yield result

    mesh_out_path.unlink()


def make_masked_data(data, mesh_out, mom_grid):
    match data:
        # This is where runoff exist on the input mesh
        case "All":
            mask = 1e10
        case "None":
            mask = 0
        case "Ocean":
            mask = mesh_out["mom_super_grid"].mesh.elementMask * 1e20
        case "Land":
            mask = 1e-20 * (mesh_out["mom_super_grid"].mesh.elementMask == 0).astype(
                int
            )
        case "Ocean_Touching_Land":
            mask_2d = np.reshape(
                mesh_out["mom_super_grid"].mesh.elementMask.values,
                (mom_grid.ny, mom_grid.nx),
            )
            # make new mask of land plus one adjacent cell of ocean
            land_neighbours = binary_dilation(mask_2d == 0)
            # target for runoff is ocean cells which are adjacent land
            mask = ((land_neighbours & mask_2d) == 1).flatten()
        case "Ocean_Not_Touching_Land":
            mask_2d = np.reshape(
                mesh_out["mom_super_grid"].mesh.elementMask.values,
                (mom_grid.ny, mom_grid.nx),
            )
            # make new mask of land plus one adjacent cell of ocean
            land_neighbours = binary_dilation(mask_2d == 0)
            # target for runoff is ocean cells which are adjacent land
            mask = (((land_neighbours == 0) & mask_2d) == 1).flatten()
        case _:
            raise ValueError(f"{data} is not an implement mask type")

    return mask


@pytest.fixture(scope="module")
def weights_path(mod_tmp_path: Path) -> Path:
    return mod_tmp_path / "drof_remap_weights.nc"


@pytest.fixture(scope="module")
def weights_file(mom_grid, mesh_out, weights_path: Path, request):

    gen_rof_weights(
        mesh_out["path"],
        str(weights_path),
        mom_grid.nx,
        mom_grid.ny,
        spread=request.param,
    )

    if not weights_path.exists():
        raise RuntimeError("drof remap weights not created")

    yield {"path": str(weights_path), "spread": request.param}

    weights_path.unlink()


# ----------------
# the actual test:


@pytest.mark.parametrize(
    "data",
    ["All", "None", "Ocean", "Land", "Ocean_Touching_Land", "Ocean_Not_Touching_Land"],
)
@pytest.mark.parametrize("weights_file", [False, True], indirect=True)
def test_regrid_conservation(data, mesh_in, mesh_out, weights_file, mom_grid):
    """
    For some provided meshes, and weights file, confirm that the weights are conservative
    """

    fld_in = mesh_in["fld"]
    area_in = mesh_in["area"]
    fld_out = mesh_out["fld"]
    area_out = mesh_out["area"]

    fld_in.data[:] = make_masked_data(data, mesh_out, mom_grid)

    # for unclear reasons, we need to zero the output field before populating it
    # it looks like cells which are not destinations in remapping can introduce rounding error
    fld_out.data[:] = 0

    esmpy.RegridFromFile(fld_in, fld_out, filename=weights_file["path"])

    # Save a plot of the new runoff, for debugging
    plt.figure()
    plt.pcolor(np.reshape(fld_out.data, (mom_grid.ny, mom_grid.nx)))
    plt.colorbar()
    plt.title(f"Runoff cells {data}")
    plt.savefig(f"Runoff cells {data}.png")

    print(f"Total before Regrid : {np.sum(fld_in.data*area_in)}")
    print(f"Total after Regrid : {np.sum(fld_out.data*area_out)}")

    # Check conservation
    assert np.sum(fld_in.data * area_in) == pytest.approx(
        np.sum(fld_out.data * area_out), rel=1e-7
    )

    if not weights_file["spread"]:

        # In no spread case, all runoff enters cells touching land
        test_target = make_masked_data("Ocean_Touching_Land", mesh_out, mom_grid) > 0
        np.testing.assert_array_equal(
            test_target & (fld_out.data > 0), (fld_out.data > 0)
        )

    if weights_file["spread"]:

        # All runoff in ocean, therefore land & zero runoff is still land
        land_mask = make_masked_data("Land", mesh_out, mom_grid) > 0
        np.testing.assert_array_equal(land_mask & (fld_out.data <= 0), land_mask)
        # Something has been spread
        # if source data is "Ocean_Not_Touching_Land", result depends on size of SRC_DIST
        if data != "None" and data != "Ocean_Not_Touching_Land":
            assert np.any(
                (fld_out.data > 0)
                & (make_masked_data("Ocean_Not_Touching_Land", mesh_out, mom_grid) > 0)
            )


@pytest.mark.parametrize("spread", ["True", "False"])
def test___init__(mesh_out, mom_grid, weights_path, spread):
    """
    Confirm the init function assigns the variables needed for later steps
    """

    test_weights = Rof_Remapping_Weights(
        mesh_out["path"], str(weights_path), mom_grid.nx, mom_grid.ny, spread
    )

    assert test_weights.nx == mom_grid.nx
    assert test_weights.ny == mom_grid.ny
    assert test_weights.mesh_ds.equals(xr.open_dataset(mesh_out["path"]))
    assert test_weights.mesh_filename == mesh_out["path"]
    assert Path(test_weights.weights_f) == weights_path
    assert test_weights.spread == spread


@pytest.mark.parametrize("spread", ["False", "True"])
def test_target_masks(mesh_out, mom_grid, weights_path, spread):
    """
    Confirm that the target masks is the whole ocean for points being spread,
    and only ocean cells touching land for others
    """

    test_weights = Rof_Remapping_Weights(
        mesh_out["path"], str(weights_path), mom_grid.nx, mom_grid.ny, spread
    )

    test_weights.target_masks()

    np.testing.assert_array_equal(
        test_weights.target_mask_spread, mesh_out["mom_super_grid"].mask.flatten()
    )

    np.testing.assert_array_equal(
        test_weights.target_mask_nospread,
        make_masked_data("Ocean_Touching_Land", mesh_out, mom_grid),
    )


def test_source_masks_no_spread(mesh_out, mom_grid, weights_path):
    """
    Confirm that no cells are allocated into the spreading mask when spread=False
    """

    test_weights = Rof_Remapping_Weights(
        mesh_out["path"], str(weights_path), mom_grid.nx, mom_grid.ny, spread=False
    )

    test_weights.source_mask()

    np.testing.assert_array_equal(
        test_weights.spread_source_mask.astype(np.int8),
        np.zeros(mom_grid.nx * mom_grid.ny),
    )


def test_source_masks_spread(mesh_out, mom_grid, weights_path):
    """
    Confirm that some cells are allocated into the spreading mask when spread=True
    """

    test_weights = Rof_Remapping_Weights(
        mesh_out["path"], str(weights_path), mom_grid.nx, mom_grid.ny, spread=True
    )

    test_weights.source_mask()

    # to-do, whats a better tests here? Compare against SPREAD_POINTS again?
    assert np.sum(test_weights.spread_source_mask.astype(np.int8)) > 0
