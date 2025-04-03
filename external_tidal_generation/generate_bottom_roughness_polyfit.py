#!/usr/bin/env python3
# Copyright 2025 ACCESS-NRI and contributors.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Compute bottom roughness (h2) on an ocean model grid by fitting a
# polynomial to high-resolution bathymetry (1/240 degree) using MPI.
# Jayne, Steven R., and Louis C. St. Laurent. "Parameterizing tidal dissipation over rough topography." Geophysical Research Letters 28.5 (2001): 811-814.
#
# Usage:
#    mpirun -n <ranks> python3 generate_bottom_roughness_polyfit.py \
#         --topo-file /path/to/topog.nc \
#         --grid-file /path/to/ocean_static.nc \
#         --mask-file /path/to/ocean_mask.nc \
#         --output h2.nc
#
# Contact:
#    - Minghang Li <Minghang.Li1@anu.edu.au>
#
# Dependencies:
#   - xarray
#   - numpy
#   - mpi4py
# =========================================================================================
from pathlib import Path
import os
import sys
import argparse
import numpy as np
import xarray as xr
from mpi4py import MPI
import subprocess

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum


def polyfit_roughness(H: np.ndarray, xv: np.ndarray, yv: np.ndarray) -> float:
    """
    Fit a polynomial of the form:
        H(x,y) = a + b*x + c*y + d*(x*y)
    to the 2D topography array H, and compute the RMS
    of the residuals (h2).
    """
    H1 = H.ravel()
    valid = ~np.isnan(H1)
    # At least 4 valid points
    if np.count_nonzero(valid) < 4:
        return np.nan

    X1 = xv.ravel()
    Y1 = yv.ravel()
    X = np.column_stack(
        [np.ones(np.sum(valid)), X1[valid], Y1[valid], (X1 * Y1)[valid]]
    )
    Y_valid = H1[valid]
    try:
        coeffs, *_ = np.linalg.lstsq(X, Y_valid, rcond=None)

        H_fit = (
            coeffs[0]
            + coeffs[1] * X1[valid]
            + coeffs[2] * Y1[valid]
            + coeffs[3] * (X1[valid] * Y1[valid])
        )
        res = Y_valid - H_fit
        h2 = np.sqrt(np.mean(res**2))
    except Exception as e:
        h2 = np.nan
    return h2


def compute_h2_poly_cell(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    topo_da: xr.DataArray,
) -> float:
    """
    Given bounding coordinates (lon_min, lon_max, lat_min, lat_max),
    select the corresponding sub-region from the high-res bathymetry
    and fit a polynomial (H = a + b*x + c*y + d*x*y) to that
    sub-region.
    """
    sub_da = topo_da.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max),
    )

    H = sub_da.values
    x = sub_da.lon.values
    y = sub_da.lat.values
    xv, yv = np.meshgrid(x, y)
    h2 = polyfit_roughness(H, xv, yv)
    return h2


def load_topo(
    path: str, chunk_lat: int = 800, chunk_lon: int = 1600
) -> (xr.DataArray, float):
    """
    Load a high-resolution bathymetry file
    """
    ds = xr.open_dataset(path, chunks={"lat": chunk_lat, "lon": chunk_lon})
    depth = ds["z"].where(ds["z"] < 0, np.nan)
    new_lon = xr.where(depth.lon >= 80, depth.lon - 360, depth.lon)
    depth = depth.assign_coords(lon=new_lon).sortby("lon")
    return depth


def load_model_grids(path: str):
    """
    Load the ocean model grid information from a static file.
    """
    ds = xr.open_dataset(path)
    xh = ds.xh
    yh = ds.yh
    ds.close()
    return xh, yh


def compute_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """
    Given an array of cell center coordinates, compute the
    corresponding cell edges.
    """
    N = len(centers)
    edges = np.empty(N + 1, dtype=centers.dtype)
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    for i in range(1, N):
        edges[i] = 0.5 * (centers[i - 1] + centers[i])
    edges[N] = centers[N - 1] + 0.5 * (centers[N - 1] - centers[N - 2])
    return edges


def evaluate_roughness(
    topo_da: xr.DataArray,
    ocean_mask: xr.DataArray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    NxCells: int,
    NyCells: int,
    comm: MPI.Comm,
) -> np.ndarray:
    """
    Distribute roughness computations across all MPI ranks, and
    gather the final 2D array of h2 values on rank 0.
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    total_rows = NyCells
    block_size = total_rows // size
    rem = total_rows % size

    y_start = rank * block_size + min(rank, rem)
    y_count = block_size + (1 if rank < rem else 0)
    y_end = y_start + y_count

    if rank == 0:
        print(
            f"[Rank {rank}] Domain partition: total rows {total_rows}, "
            f"Rank {rank} covers rows {y_start} to {y_end - 1}."
        )

    # Initilise local h2 array
    # local_h2 = np.zeros((y_count, NxCells), dtype=np.float32)
    local_h2 = np.full((y_count, NxCells), np.nan, dtype=np.float32)
    # Compute h2 for each rank
    for j in range(y_start, y_end):
        for i in range(NxCells):
            # Skip land cells
            if ocean_mask[j, i] == 0:
                continue
            h2_val = compute_h2_poly_cell(
                xedges[i], xedges[i + 1], yedges[j], yedges[j + 1], topo_da
            )
            local_h2[j - y_start, i] = h2_val

        if (j - y_start) % 3 == 0:
            print(
                f"[Rank {rank}] Processed global row {j} (local index {j - y_start})",
            )

    # Collect all local data to rank 0
    local_1d = local_h2.ravel()
    local_size = local_1d.size

    # Gather sizes from all ranks
    all_sizes = comm.gather(local_size, root=0)

    if rank == 0:
        total_size = sum(all_sizes)
        global_1d = np.empty(total_size, dtype=np.float32)
        displs = np.insert(np.cumsum(all_sizes), 0, 0)[:-1]
    else:
        global_1d = None
        displs = None

    comm.Gatherv(sendbuf=local_1d, recvbuf=(global_1d, (all_sizes, displs)), root=0)

    # On rank 0, reshape back into a 2D array
    if rank == 0:
        final_h2 = np.full((NyCells, NxCells), np.nan, dtype=np.float32)
        offset = 0
        for r in range(size):
            block = all_sizes[r]
            sub_data = global_1d[offset : offset + block]
            offset += block

            block_rows = block // NxCells
            r_start = r * block_size + min(r, rem)
            final_h2[r_start : r_start + block_rows, :] = sub_data.reshape(
                block_rows, NxCells
            )

        return final_h2
    else:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute bottom roughness (h2) via polynomial fit with a high-resolution topography."
    )
    parser.add_argument(
        "--topo-file",
        type=str,
        required=True,
        help="Path to a high-resolution topography file.",
    )
    parser.add_argument(
        "--grid-file", type=str, required=True, help="Path to the ocean static file."
    )
    parser.add_argument(
        "--mask-file", type=str, required=True, help="Path to the ocean mask file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="h2.nc",
        help="Output roughness filename (default: h2.nc).",
    )
    parser.add_argument(
        "--chunk-lat",
        type=int,
        default=800,
        help="Dask chunk size along lat dimension (default:800).",
    )
    parser.add_argument(
        "--chunk-lon",
        type=int,
        default=1600,
        help="Dask chunk size along lon dimension (default:1600).",
    )
    args = parser.parse_args()

    # create communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Rank 0 loads data and initializes shared info
    if rank == 0:
        topo_da = load_topo(
            args.topo_file, chunk_lat=args.chunk_lat, chunk_lon=args.chunk_lon
        )
        ocean_mask = xr.open_dataset(args.mask_file).mask
        xh, yh = load_model_grids(args.grid_file)
        xedges = compute_edges_from_centers(xh)
        yedges = compute_edges_from_centers(yh)
        NxCells = len(xedges) - 1
        NyCells = len(yedges) - 1
    else:
        # Placeholders on other ranks
        topo_da = None
        ocean_mask = None
        xedges = None
        yedges = None
        NxCells = None
        NyCells = None
        xh = None
        yh = None

    topo_da = comm.bcast(topo_da, root=0)
    ocean_mask = comm.bcast(ocean_mask, root=0)
    NxCells = comm.bcast(NxCells, root=0)
    NyCells = comm.bcast(NyCells, root=0)
    xedges = comm.bcast(xedges, root=0)
    yedges = comm.bcast(yedges, root=0)
    xh = comm.bcast(xh, root=0)
    yh = comm.bcast(yh, root=0)

    final_h2 = evaluate_roughness(
        topo_da=topo_da,
        ocean_mask=ocean_mask,
        xedges=xedges,
        yedges=yedges,
        NxCells=NxCells,
        NyCells=NyCells,
        comm=comm,
    )

    # Rank 0 writes results to file
    if rank == 0:
        h2_out = xr.Dataset(
            {"h2": (("yh", "xh"), final_h2)},
            coords={
                "yh": yh,
                "xh": xh,
            },
            attrs={
                "long_name": "Polynomial-fit roughness (h2) per model cell",
                "units": "m",
            },
        )
        # Add metadata
        this_file = os.path.normpath(__file__)
        runcmd = (
            f"mpirun -n $PBS_NCPUS"
            f"python3 {os.path.basename(this_file)} "
            f"--topo-file={args.topo_file}"
            f"--chunk_lat={args.chunk_lat}"
            f"--chunk_lon={args.chunk_lon}"
            f"--grid-file={args.grid_file}"
            f"--mask-file={args.mask_file}"
            f"--output={args.output}"
        )

        try:
            history = get_provenance_metadata(this_file, runcmd)
        except subprocess.CalledProcessError:
            history = "Provenance metadata unavailable (not a git repo?)"
        global_attrs = {"history": history}
        # add md5 hashes for input files
        file_hashes = [
            f"{args.topo_file} (md5 hash: {md5sum(args.topo_file)})",
            f"{args.grid_file} (md5 hash: {md5sum(args.grid_file)})",
            f"{args.mask_file} (md5 hash: {md5sum(args.mask_file)})",
        ]
        global_attrs["inputFile"] = ", ".join(file_hashes)

        h2_out.attrs.update(global_attrs)

        h2_out.to_netcdf(args.output)


if __name__ == "__main__":
    main()
