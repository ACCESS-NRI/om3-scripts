#!/usr/bin/env python3
# Copyright 2025 ACCESS-NRI and contributors.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Compute bottom roughness on an ocean model grid by fitting a
# polynomial to high-resolution bathymetry (1/240 degree) using mpi.
# NB: The output nc file contains the squared bottom roughness values, as these are used directly in the source.
# Reference:
# Jayne, Steven R., and Louis C. St. Laurent.
# "Parameterizing tidal dissipation over rough topography."
# Geophysical Research Letters 28.5 (2001): 811-814.
#
# Usage:
#    mpirun -n <ranks> python3 generate_bottom_roughness_polyfit.py \
#         --high-res-topo-file /path/to/topo_high_res.nc \
#         --hgrid-file /path/to/ocean_hgrid.nc \
#         --topog-file /path/to/topog.nc \
#         --chunk-lat chunk_lat
#         --chunk-lon chunk_lon
#         --output output.nc
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
from mpi4py import MPI
import numpy as np
import xarray as xr

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum
from mesh_generation.generate_mesh import mom6_mask_detection


def load_high_res_topo(
    path: str, chunk_lat: int = 800, chunk_lon: int = 1600
) -> xr.DataArray:
    """
    Load a high-resolution bathymetry file
    """
    ds = xr.open_dataset(path, chunks={"lat": chunk_lat, "lon": chunk_lon})
    depth = ds["z"].where(ds["z"] < 0, np.nan)
    return depth


def project_lon(lon: float) -> float:
    """
    Project longitude to the range [-280, 80].
    """

    return ((lon + 280) % 360) - 280


def align_lon_coords(da: xr.DataArray) -> xr.DataArray:
    """
    Align high resolution topography lon coord to the range [-280, 80].
    """

    da = da.assign_coords(lon=project_lon(da.lon))
    da = da.sortby("lon")

    return da


def polyfit_roughness(H: np.ndarray, xv: np.ndarray, yv: np.ndarray) -> float:
    """
    Fit a polynomial of the form:
        H(x,y) = a + b*x + c*y + d*(x*y)
    to the 2D topography array H, and compute the RMS
    of the residuals (hrms).
    """
    H1 = H.ravel()
    valid = ~np.isnan(H1)
    # At least 4 valid points
    if np.count_nonzero(valid) < 4:
        return np.nan

    # prepare X matrix for least-squares fitting
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
        hrms = np.sqrt(np.mean(res**2))
    except Exception as e:
        hrms = np.nan

    return hrms


def compute_hrms_poly_cell(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    topo_high_res: xr.DataArray,
) -> float:
    """
    Given bounding coordinates (lon_min, lon_max, lat_min, lat_max),
    select the corresponding sub-region from the high-res bathymetry
    and fit a polynomial (H = a + b*x + c*y + d*x*y) to that
    sub-region.
    """
    sub_da = topo_high_res.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max),
    )

    H = sub_da.values
    x = sub_da.lon.values
    y = sub_da.lat.values
    xv, yv = np.meshgrid(x, y)

    R = 6.371229e6  # earth radius in meters
    deg2rad = np.pi / 180
    lat0 = np.mean(y)  # a reference lat
    lon0 = np.mean(x)  # a reference lon

    xlon = (xv - lon0) * np.cos(yv * deg2rad) * deg2rad * R
    ylat = (yv - lat0) * deg2rad * R
    hrms = polyfit_roughness(H, xlon, ylat)
    return hrms


def evaluate_roughness(
    topo_high_res: xr.DataArray,
    ocean_mask: xr.DataArray,
    lon_b: np.ndarray,
    lat_b: np.ndarray,
    nx: int,
    ny: int,
    comm: MPI.Comm,
) -> np.ndarray:
    """
    Distribute roughness computations across all MPI ranks, and
    gather the final 2D array of hrms values on rank 0.
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    total_rows = ny
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

    local_hrms = np.full((y_count, nx), np.nan, dtype=np.float32)

    # Compute hrms for each rank
    for j in range(y_start, y_end):
        for i in range(nx):
            # Skip land cells
            if ocean_mask[j, i] == 0:
                continue

            this_lon_corners = [
                lon_b[j, i],
                lon_b[j, i + 1],
                lon_b[j + 1, i],
                lon_b[j + 1, i + 1],
            ]
            this_lat_corners = [
                lat_b[j, i],
                lat_b[j, i + 1],
                lat_b[j + 1, i],
                lat_b[j + 1, i + 1],
            ]

            lon_min = np.min(this_lon_corners)
            lon_max = np.max(this_lon_corners)
            lat_min = np.min(this_lat_corners)
            lat_max = np.max(this_lat_corners)

            hrms_val = compute_hrms_poly_cell(
                lon_min, lon_max, lat_min, lat_max, topo_high_res
            )
            local_hrms[j - y_start, i] = hrms_val

        if (j - y_start) % 3 == 0:
            print(
                f"[Rank {rank}] Processed row {j} (local index {j - y_start})",
            )

    # Collect all local data to rank 0
    local_1d = local_hrms.ravel()
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
        final_hrms = np.full((ny, nx), np.nan, dtype=np.float32)
        offset = 0
        for r in range(size):
            block = all_sizes[r]
            sub_data = global_1d[offset : offset + block]
            offset += block

            block_rows = block // nx
            r_start = r * block_size + min(r, rem)
            final_hrms[r_start : r_start + block_rows, :] = sub_data.reshape(
                block_rows, nx
            )
        return final_hrms
    else:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute bottom roughness via polynomial fit with a high-resolution topography."
    )
    parser.add_argument(
        "--high-res-topo-file",
        type=str,
        required=True,
        help="Path to a high-resolution topography file.",
    )
    parser.add_argument(
        "--hgrid-file", type=str, required=True, help="Path to ocean_hgrid.nc"
    )
    parser.add_argument(
        "--topog-file",
        type=str,
        required=True,
        help="Path to the model topography file, which is used to generate the mask.",
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
    parser.add_argument(
        "--output",
        type=str,
        default="bottom_roughness.nc",
        help="Output roughness filename (default: bottom_roughness.nc).",
    )
    args = parser.parse_args()

    # create communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Load high-resolution bathymetry ranging from [-180, 180]
        topo_high_res = load_high_res_topo(
            args.high_res_topo_file, chunk_lat=args.chunk_lat, chunk_lon=args.chunk_lon
        )

        # Load model topog file then generate ocean mask ranging from [-280, 80]
        topo = xr.open_dataset(Path(args.topog_file))
        mask = mom6_mask_detection(topo)

        # Convert lon coords to a [-280, 80] to match the range of the model grid
        topo_high_res = align_lon_coords(topo_high_res)

        # Load hgrid
        hgrid = xr.open_dataset(args.hgrid_file)

        # Get the boundary values of the model grid
        lon_b = hgrid.x[::2, ::2].values
        lat_b = hgrid.y[::2, ::2].values
        lon = hgrid.x[1::2, 1::2].values
        lat = hgrid.y[1::2, 1::2].values

        # Get dims of the model grid
        ny = lon.shape[0]
        nx = lon.shape[1]

        print(f"[Rank 0] grid dimensions: {ny} cells in y, {nx} cells in x.")

    else:
        topo_high_res = None
        mask = None
        lon_b = None
        lat_b = None
        lon = None
        lat = None
        ny = None
        nx = None

    # Broadcast fields and grid sizes to all ranks.
    topo_high_res = comm.bcast(topo_high_res, root=0)
    lon_b = comm.bcast(lon_b, root=0)
    lat_b = comm.bcast(lat_b, root=0)
    mask = comm.bcast(mask, root=0)
    ny = comm.bcast(ny, root=0)
    nx = comm.bcast(nx, root=0)

    # Evaluate bottom roughness on the grid.
    final_hrms = evaluate_roughness(
        topo_high_res=topo_high_res,
        ocean_mask=mask,
        lon_b=lon_b,
        lat_b=lat_b,
        nx=nx,
        ny=ny,
        comm=comm,
    )

    # Rank 0 writes results to file
    if rank == 0:
        h2_out = xr.Dataset(
            {
                "h2": xr.DataArray(
                    np.nan_to_num(final_hrms**2, nan=0.0),
                    dims=("yh", "xh"),
                    attrs={
                        "long_name": (
                            "Polynomial-fit bottom roughness squared (h^2) per model grid cell "
                            "following Jayne & Laurent (2001)"
                        ),
                        "units": "m^2",
                    },
                ),
                "lon": xr.DataArray(
                    lon,
                    dims=("yh", "xh"),
                    attrs={
                        "long_name": "Longitude",
                        "units": "degrees_east",
                    },
                ),
                "lat": xr.DataArray(
                    lat,
                    dims=("yh", "xh"),
                    attrs={
                        "long_name": "Latitude",
                        "units": "degrees_north",
                    },
                ),
            },
        )

        # Add provenance metadata and MD5 hashes for input files.
        this_file = os.path.normpath(__file__)
        runcmd = (
            f"mpirun -n $PBS_NCPUS python3 {os.path.basename(this_file)} "
            f"--high-res-topo-file={args.high_res_topo_file} "
            f"--hgrid-file={args.hgrid_file} "
            f"--topog-file={args.topog_file} "
            f"--chunk-lat={args.chunk_lat} "
            f"--chunk-lon={args.chunk_lon} "
            f"--output={args.output}"
        )

        history = get_provenance_metadata(this_file, runcmd)
        global_attrs = {"history": history}
        file_hashes = [
            f"{args.high_res_topo_file} (md5 hash: {md5sum(args.high_res_topo_file)})",
            f"{args.hgrid_file} (md5 hash: {md5sum(args.hgrid_file)})",
            f"{args.topog_file} (md5 hash: {md5sum(args.topog_file)})",
        ]
        global_attrs["inputFile"] = ", ".join(file_hashes)
        h2_out.attrs.update(global_attrs)

        h2_out.to_netcdf(args.output)
        print(f"[Rank 0] Square of bottom roughness output written to: {args.output}")


if __name__ == "__main__":
    main()
