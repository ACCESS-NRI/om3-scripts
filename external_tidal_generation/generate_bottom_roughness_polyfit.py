#!/usr/bin/env python3
# Copyright 2025 ACCESS-NRI and contributors.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Compute bottom roughness on an ocean model grid by fitting a
# polynomial to high-resolution bathymetry (1/240 degree) using mpi.
#
# Reference:
# Jayne, Steven R., and Louis C. St. Laurent.
# "Parameterizing tidal dissipation over rough topography."
# Geophysical Research Letters 28.5 (2001): 811-814.
#
# Usage:
#    mpirun -n <ranks> python3 generate_bottom_roughness_polyfit.py \
#         --topo-file /path/to/topog.nc \
#         --hgrid-file /path/to/ocean_hgrid.nc \
#         --regrid-mask-file /path/to/ocean_mask.nc \
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


def load_topo(path: str, chunk_lat: int = 800, chunk_lon: int = 1600) -> xr.DataArray:
    """
    Load a high-resolution bathymetry file
    """
    ds = xr.open_dataset(path, chunks={"lat": chunk_lat, "lon": chunk_lon})
    depth = ds["z"].where(ds["z"] < 0, np.nan)
    return depth


def project_lon(
    lon: float,
) -> float:
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


def load_dataset(path: str) -> xr.Dataset:
    """
    Load an input dataset.
    """
    ds = xr.open_dataset(path)
    return ds


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

    R = 6.371229e6  # earth radius in meters
    deg2rad = np.pi / 180
    lat0 = np.mean(y)  # a reference lat
    lon0 = np.mean(x)  # a reference lon

    xlon = (xv - lon0) * np.cos(yv * deg2rad) * deg2rad * R
    ylat = (yv - lat0) * deg2rad * R
    hrms = polyfit_roughness(H, xlon, ylat)
    return hrms


def evaluate_roughness(
    topo_da: xr.DataArray,
    ocean_mask: xr.DataArray,
    hgrid_xc: np.ndarray,
    hgrid_yc: np.ndarray,
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
                hgrid_xc[j, i],
                hgrid_xc[j, i + 1],
                hgrid_xc[j + 1, i],
                hgrid_xc[j + 1, i + 1],
            ]
            this_lat_corners = [
                hgrid_yc[j, i],
                hgrid_yc[j, i + 1],
                hgrid_yc[j + 1, i],
                hgrid_yc[j + 1, i + 1],
            ]

            lon_min = np.min(this_lon_corners)
            lon_max = np.max(this_lon_corners)
            lat_min = np.min(this_lat_corners)
            lat_max = np.max(this_lat_corners)

            hrms_val = compute_hrms_poly_cell(
                lon_min, lon_max, lat_min, lat_max, topo_da
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
        "--topo-file",
        type=str,
        required=True,
        help="Path to a high-resolution topography file.",
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
        "--mask-file",
        type=str,
        required=True,
        help="Path to the ocean mask file.",
    )
    parser.add_argument(
        "--hgrid", type=str, required=True, help="Path to ocean_hgrid.nc"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="conservative_normed",
        help="Regridding method (e.g., bilinear, conservative, conservative_normed)",
    )
    parser.add_argument(
        "--agg-factor",
        type=int,
        default=1,
        help="Coarse factor. Eg, 1 for original grid (eg 0.25deg); 2 for 0.5deg or 4 for 1deg resolution (default: 1 means original model grid).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="interpolated_hrms.nc",
        help="Output roughness filename (default: interpolated_hrms.nc).",
    )
    args = parser.parse_args()

    # create communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Load high-resolution bathymetry
        topo_da = load_topo(
            args.topo_file, chunk_lat=args.chunk_lat, chunk_lon=args.chunk_lon
        )
        ocean_mask = load_dataset(args.mask_file)
        if "mask" not in ocean_mask:
            raise KeyError("Missing variable 'mask' in ocean_mask file!")

        # Convert lon coords to a [-280, 80] to match the range of the model grid
        topo_da = align_lon_coords(topo_da)

        # Load hgrid
        hgrid = load_dataset(args.hgrid)
        hgrid_x = hgrid.x[1::2, 1::2]
        hgrid_y = hgrid.y[1::2, 1::2]
        hgrid_xc = hgrid.x[::2, ::2]
        hgrid_yc = hgrid.y[::2, ::2]

        # Load model ocean mask and tweak coords
        ocean_mask = ocean_mask.drop_vars(["geolon_t", "geolat_t"])
        ocean_mask = ocean_mask.rename({"ny": "lat", "nx": "lon"})

        ocean_mask = ocean_mask.assign_coords(
            {
                "lon": (("lat", "lon"), hgrid_x.values),
                "lat": (("lat", "lon"), hgrid_y.values),
                "lon_b": (("lat_b", "lon_b"), hgrid_xc.values),
                "lat_b": (("lat_b", "lon_b"), hgrid_yc.values),
            }
        )

        # check lon range for high resolution topo
        print(topo_da.lon.values)

        # Get the full boundary values of the ocean mask
        lon_b_full = ocean_mask.lon_b.values
        lat_b_full = ocean_mask.lat_b.values
        ocean_mask_full = ocean_mask["mask"].values

        # Get the dims of the full ocean mask
        ny_full = lon_b_full.shape[0] - 1
        nx_full = lon_b_full.shape[1] - 1

        fac = args.agg_factor
        if fac > 1:
            # Compute number of coarse cells
            nC_y = ny_full // fac
            nC_x = nx_full // fac

            # Crop boundaries for cell edges
            ny_edge = nC_y * fac + 1
            nx_edge = nC_x * fac + 1
            lon_b_crop = lon_b_full[:ny_edge, :nx_edge]
            lat_b_crop = lat_b_full[:ny_edge, :nx_edge]

            # Coarse using striding
            lon_b_coarse = lon_b_crop[::fac, ::fac]
            lat_b_coarse = lat_b_crop[::fac, ::fac]

            # Compute coarse cell centers by averaging the four surrounding boundaries.
            lon_coarse = 0.25 * (
                lon_b_coarse[:-1, :-1]
                + lon_b_coarse[:-1, 1:]
                + lon_b_coarse[1:, :-1]
                + lon_b_coarse[1:, 1:]
            )
            lat_coarse = 0.25 * (
                lat_b_coarse[:-1, :-1]
                + lat_b_coarse[:-1, 1:]
                + lat_b_coarse[1:, :-1]
                + lat_b_coarse[1:, 1:]
            )

            # Crop ocean_mask at cell center
            ny_crop = (ny_full // fac) * fac
            nx_crop = (nx_full // fac) * fac
            ocean_mask_cropped = ocean_mask_full[:ny_crop, :nx_crop]
            # ocean_mask_coarse = ocean_mask_cropped.reshape(ny_crop // fac, fac,nx_crop // fac, fac).max(axis=(1, 3))
            ocean_mask_coarse = ocean_mask_cropped[::fac, ::fac]

            # Update dimensions
            ny = ocean_mask_coarse.shape[0]
            nx = ocean_mask_coarse.shape[1]

        else:
            # No coarsen needed
            lon_b_coarse = lon_b_full
            lat_b_coarse = lat_b_full
            ocean_mask_coarse = ocean_mask_full
            ny = ny_full
            nx = nx_full

            lon_coarse = ocean_mask.lon.values
            lat_coarse = ocean_mask.lat.values

        print(f"[Rank 0] coarsen grid dimensions: {ny} cells in y, {nx} cells in x.")

    else:
        topo_da = None
        lon_b_coarse = None
        lat_b_coarse = None
        ocean_mask_coarse = None
        lon_coarse = None
        lat_coarse = None
        ny = None
        nx = None

    # Broadcast fields and grid sizes to all ranks.
    topo_da = comm.bcast(topo_da, root=0)
    lon_b_coarse = comm.bcast(lon_b_coarse, root=0)
    lat_b_coarse = comm.bcast(lat_b_coarse, root=0)
    ocean_mask_coarse = comm.bcast(ocean_mask_coarse, root=0)
    ny = comm.bcast(ny, root=0)
    nx = comm.bcast(nx, root=0)

    # Evaluate HRMS on the grid.
    final_hrms = evaluate_roughness(
        topo_da=topo_da,
        ocean_mask=ocean_mask_coarse,
        hgrid_xc=lon_b_coarse,
        hgrid_yc=lat_b_coarse,
        nx=nx,
        ny=ny,
        comm=comm,
    )

    # Rank 0 writes results to file
    if rank == 0:
        if args.agg_factor == 1:
            # No regridding and no coarsen,
            # hence output results on the original model grid
            h2_out = xr.Dataset(
                {
                    "h2": (("y", "x"), final_hrms**2),
                },
                coords={
                    "lon": (("y", "x"), lon_coarse),
                    "lat": (("y", "x"), lat_coarse),
                    "lon_b": (("y_b", "x_b"), lon_b_coarse),
                    "lat_b": (("y_b", "x_b"), lat_b_coarse),
                },
                attrs={
                    "long_name": (
                        "Polynomial-fit roughness square h^2 per model grid cell "
                        "following Jayne & Laurent (2001)"
                    ),
                    "units": "m",
                },
            )
            h2_out["h2_0"] = h2_out["h2"].fillna(0.0)

        # Add provenance metadata and MD5 hashes for input files.
        this_file = os.path.normpath(__file__)
        runcmd = (
            f"mpirun -n $PBS_NCPUS python3 {os.path.basename(this_file)} "
            f"--topo-file={args.topo_file} "
            f"--hgrid={args.hgrid} "
            f"--chunk-lat={args.chunk_lat} "
            f"--chunk-lon={args.chunk_lon} "
            f"--mask-file={args.mask_file} "
            f"--agg-factor={args.agg_factor} "
            f"--output={args.output}"
        )

        history = get_provenance_metadata(this_file, runcmd)
        global_attrs = {"history": history}
        file_hashes = [
            f"{args.topo_file} (md5 hash: {md5sum(args.topo_file)})",
            f"{args.hgrid} (md5 hash: {md5sum(args.hgrid)})",
            f"{args.mask_file} (md5 hash: {md5sum(args.mask_file)})",
        ]
        global_attrs["inputFile"] = ", ".join(file_hashes)
        h2_out.attrs.update(global_attrs)

        h2_out.to_netcdf(args.output)
        print(f"[Rank 0] HRMS output written to: {args.output}")


if __name__ == "__main__":
    main()
