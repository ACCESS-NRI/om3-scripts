#!/usr/bin/env python3
# Copyright 2026 ACCESS-NRI and contributors.
# SPDX-License-Identifier: Apache-2.0

# =========================================================================================
# Bottom roughness for internal-tide generation (h^2) from WOA23 + synbath
#
# This script builds a bottom-roughness field intended for parameterising internal-tide
# generation in ocean model configurations. The roughness is defined relative to bathymetry
# variability on the length scale "seen" by a mode-1 internal tide, rather than the model
# grid spacing. That choice makes the resulting h^2 largely independent of model resolution,
# which is important when the model does not explicitly generate internal tides.
#
# What the script does
# --------------------
# 1) Compute buoyancy frequency squared (N^2)
#    - Reads WOA23 temperature and salinity climatologies.
#    - Uses TEOS-10 (gsw) to compute N^2 on vertical mid-levels.
#
# 2) Compute the mode-1 internal tide wavelength scale (lambda1)
#    - For a specified tidal frequency (default: M2), computes lambda1 as a function of
#      latitude/longitude from N^2 and the Coriolis parameter.
#    - lambda1 is used as the smoothing / sampling radius for the next steps.
#
# 3) Compute Gaussian-weighted mean depth from high-resolution bathymetry (synbath)
#    - For each WOA grid point with valid lambda1, samples the high-res bathymetry on a
#      polar stencil whose radius scales with lambda1.
#    - Applies a Gaussian-like weighting to produce a smoothed mean depth
#      of the mode-1 internal tide length scale.
#
# 4) Compute depth variance (h^2) relative to the smoothed mean depth
#    - Re-samples synbath on the same polar stencil.
#    - Interpolates the smoothed mean depth to the stencil points.
#    - Computes the Gaussian-weighted variance of depth residuals:
#         h^2 = <(depth - mean_depth)^2>
#
# 5) Regrid to the MOM6 grid (done separately)
#    - Regridding is performed by generate_bottom_roughness_regrid.py.
#    - This is intentionally split out due to a known xesmf + mpi4py issue in the analysis
#      environment: https://github.com/ACCESS-NRI/ACCESS-Analysis-Conda/issues/207
#    - The regrid step conservatively maps h^2 from the regular WOA grid onto a MOM6 mesh.
#
# Usage:
#   mpirun -n <ranks> python3 generate_bottom_roughness_intermediate_woa.py \
#       --woa23_temp_file /path/to/woa23_temp.nc \
#       --woa23_salt_file /path/to/woa23_salt.nc \
#       --synbath_file    /path/to/SYNBATH.nc \
#       --chunk_lat 800 \
#       --chunk_lon 1600 \
#       --nmodes 100 \
#       --ntheta 180 \
#       --omega 1.405189e-4 # M2 \
#       --woa_intermediate_file woa_intermediates.nc
#
# Notes:
# - The implementation follows the matlab reference workflow provided by Callum Shakespeare
#   and was adapted for parallel execution and regridding to MOM6.
# - Background and discussion:
#   https://github.com/ACCESS-NRI/access-om3-configs/issues/457
#
# Contact:
#    - Minghang Li <Minghang.Li1@anu.edu.au>
#
# Dependencies:
#   - gsw
#   - xarray
#   - numpy
#   - mpi4py
#
# Modules:
#   module use /g/data/xp65/public/modules
#   module load conda/analysis3-25.05
#   module load openmpi/4.1.7
#   module load git
# =========================================================================================
import sys
import os
import argparse
from pathlib import Path
import numpy as np
from mpi4py import MPI
import xarray as xr
import gsw
from dataclasses import dataclass

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from scripts_common import get_provenance_metadata, md5sum


def coriolis_f(lat: xr.DataArray) -> xr.DataArray:
    """
    Compute Coriolis parameter f at given latitudes.
    Units: s^-1
    """
    Omega = 2 * np.pi / (24 * 3600)
    return 2 * Omega * np.sin(np.deg2rad(lat))


def compute_lambda(
    N2: xr.DataArray,
    lat: xr.DataArray,
    depth: xr.DataArray,
    omega: float,
) -> xr.DataArray:
    """
    Compute the vertical mode wavelength lambda1 for internal tides.
    """
    dz = xr.DataArray(
        np.diff(depth.values),
        dims=("depth_mid",),
        name="dz",
    )
    f = coriolis_f(lat)
    H = xr.where(N2 > omega**2, 1, 0)
    integrand = np.sqrt((N2 - omega**2) / np.abs(omega**2 - f**2) * H)
    npi_on_K = (integrand * dz).sum(dim="depth_mid", skipna=True)
    lambda1 = 2 * npi_on_K
    lambda1.name = "lambda1"

    return lambda1


@dataclass
class PolarWeights:
    """
    The depth seen by a "mode 1" internal tide at polar coords (r, theta)
    around a point with Gaussian weights.
    """

    r: np.ndarray
    cos_t: np.ndarray
    sin_t: np.ndarray
    weight: np.ndarray
    weight_sum: float

    @classmethod
    def build(cls, nmodes: int, ntheta: int):
        nr = 2 * nmodes + 1
        r = np.linspace(-1, 1, nr)
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)

        cos_t = np.cos(theta)[None, :]
        sin_t = np.sin(theta)[None, :]
        r_col = r[:, None]

        weight = np.exp(-((2 * r_col) ** 2)) * (2 * np.pi) * np.abs(r_col)
        weight = np.broadcast_to(weight, (nr, ntheta))

        return cls(
            r=r,
            cos_t=cos_t,
            sin_t=sin_t,
            weight=weight,
            weight_sum=np.sum(weight),
        )


def bilinear_interp(
    field: np.ndarray,
    lon0: float,
    lat0: float,
    dres: float,
    lon_x: np.ndarray,
    lat_y: np.ndarray,
) -> np.ndarray:
    """
    Bilinear on regular grid - if any corner is NaN -> output NaN.
    """
    ny, nx = field.shape

    ix = (lon_x - lon0) / dres
    iy = (lat_y - lat0) / dres

    ix0 = np.floor(ix).astype(np.int64)
    iy0 = np.floor(iy).astype(np.int64)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    inside = (ix0 >= 0) & (ix1 < nx) & (iy0 >= 0) & (iy1 < ny)
    out = np.full(lon_x.shape, np.nan)
    if not np.any(inside):
        return out

    vf = inside.ravel()
    ix0f = ix0.ravel()[vf]
    ix1f = ix1.ravel()[vf]
    iy0f = iy0.ravel()[vf]
    iy1f = iy1.ravel()[vf]
    ixf = ix.ravel()[vf]
    iyf = iy.ravel()[vf]

    v_ll = field[iy0f, ix0f]
    v_lr = field[iy0f, ix1f]
    v_ul = field[iy1f, ix0f]
    v_ur = field[iy1f, ix1f]

    finite = (
        np.isfinite(v_ll) & np.isfinite(v_lr) & np.isfinite(v_ul) & np.isfinite(v_ur)
    )
    if not np.any(finite):
        return out

    v_ll = v_ll[finite]
    v_lr = v_lr[finite]
    v_ul = v_ul[finite]
    v_ur = v_ur[finite]

    wx = ixf[finite] - ix0f[finite]
    wy = iyf[finite] - iy0f[finite]

    # manual bilinear interpolation to minimise overhead
    interp_val = (
        v_ll * (1.0 - wx) * (1.0 - wy)
        + v_lr * wx * (1.0 - wy)
        + v_ul * (1.0 - wx) * wy
        + v_ur * wx * wy
    )

    tmp = out.ravel()
    idx = np.flatnonzero(vf)[finite]
    tmp[idx] = interp_val
    return tmp.reshape(out.shape)


def read_lonbuffer_patch(
    topo_da: xr.DataArray, j0: int, j1: int, i0_buf: int, i1_buf: int, nlon: int
) -> np.ndarray | None:
    """
    Read slice from [lon-360, lon, lon+360] buffer.
    i0_buf & i1_buf in [0, 3*nlon-1]
    """
    pieces = []

    # seg0
    a0 = max(i0_buf, 0)
    b0 = min(i1_buf, nlon - 1)
    if a0 <= b0:
        pieces.append(topo_da.isel(lat=slice(j0, j1 + 1), lon=slice(a0, b0 + 1)).values)

    # seg1
    a1 = max(i0_buf, nlon)
    b1 = min(i1_buf, 2 * nlon - 1)
    if a1 <= b1:
        pieces.append(
            topo_da.isel(
                lat=slice(j0, j1 + 1), lon=slice(a1 - nlon, b1 - nlon + 1)
            ).values
        )

    # seg2
    a2 = max(i0_buf, 2 * nlon)
    b2 = min(i1_buf, 3 * nlon - 1)
    if a2 <= b2:
        pieces.append(
            topo_da.isel(
                lat=slice(j0, j1 + 1), lon=slice(a2 - 2 * nlon, b2 - 2 * nlon + 1)
            ).values
        )

    if not pieces:
        return None

    h = np.concatenate(pieces, axis=1)
    h[h >= 0.0] = np.nan
    return h


def split_rows(nj: int, size: int, rank: int) -> tuple[int, int, int, int, int]:
    """
    Compute a simple block+remainder row decomposition.
    """
    block_size = nj // size
    rem = nj % size
    j_start = rank * block_size + min(rank, rem)
    j_count = block_size + (1 if rank < rem else 0)
    j_end = j_start + j_count
    return block_size, rem, j_start, j_end, j_count


def gatherv_indexed(
    local_idx: np.ndarray, local_val: np.ndarray, total_size: int, comm: MPI.Comm
) -> np.ndarray | None:
    """
    Gather variable-length (index, value) arrays to rank 0 and rebuild it to a 1D global array.
    Total size is nj*ni
    """
    rank = comm.Get_rank()

    n_local = local_idx.size
    counts = comm.gather(n_local, root=0)

    if rank == 0:
        displs = np.zeros_like(counts)
        displs[1:] = np.cumsum(counts)[:-1]

        all_idx = np.empty(np.sum(counts), dtype=local_idx.dtype)
        all_val = np.empty(np.sum(counts), dtype=local_val.dtype)

    else:
        counts = None
        displs = None
        all_idx = None
        all_val = None

    comm.Gatherv(local_idx, (all_idx, (counts, displs)), root=0)
    comm.Gatherv(local_val, (all_val, (counts, displs)), root=0)

    if rank != 0:
        return None

    out1d = np.full(total_size, np.nan, dtype=local_val.dtype)
    out1d[all_idx] = all_val
    return out1d


def compute_mean_depth_and_var_points(
    lon_np: np.ndarray,
    lat_np: np.ndarray,
    lambda1_np: np.ndarray,
    nmodes: int,
    ntheta: int,
    RE: float,
    synbath_file: str,
    chunk_lat: int,
    chunk_lon: int,
    synbath_lon_np: np.ndarray,
    synbath_lat_np: np.ndarray,
    synbath_nlon: int,
    comm: MPI.Comm,
    print_every: int = 100,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Compute Gaussian-weighted mean depth and depth variance at WOA lon/lat points.

    This evalautes depth over a polar sampling pattern whose physical radius is scaled
    by the local mode-1 internal tide wavelength (lambda1). The sampling is performed
    on a high-resolution bathymetry dataset (synbath) using nan-preserving bilinear interpolation
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    ni = lon_np.size
    nj = lat_np.size
    total = nj * ni

    if rank == 0:
        mask = lambda1_np > 0.0
        tasks = np.flatnonzero(mask.ravel()).astype(np.int64)
        print(f"total points={total}, tasks(ocean)={tasks.size}, ranks={size}")
    else:
        tasks = None

    tasks = comm.bcast(tasks, root=0)

    local_tasks = tasks[rank::size]
    n_local = local_tasks.size
    if rank == 0:
        print(f"avg tasks per rank ~ {tasks.size/size:.1f}")
    print(f"[Rank {rank}] local tasks = {n_local}")

    # Precompute polar weights
    polar = PolarWeights.build(nmodes, ntheta)

    # Load synbath topo
    ds_topog = xr.open_dataset(
        synbath_file, chunks={"lat": chunk_lat, "lon": chunk_lon}
    )
    topog = ds_topog["z"].where(ds_topog["z"] < 0, np.nan).transpose("lat", "lon")

    synbath_lon0 = synbath_lon_np[0]
    synbath_lat0 = synbath_lat_np[0]
    synbath_res = synbath_lon_np[1] - synbath_lon_np[0]
    synbath_lon_buf0 = synbath_lon0 - 360

    deg_per_m_lat = 180 / (np.pi * RE)

    # Precompute per-lat deg_per_m_lon for speed
    coslat = np.cos(np.deg2rad(lat_np))
    deg_per_m_lon_by_j = 180 / (np.pi * RE * coslat)

    # Compute local mean depth
    local_idx_mean_depth = np.empty(n_local, dtype=np.int64)
    local_val_mean_depth = np.empty(n_local, dtype=np.float64)
    local_val_mean_depth.fill(np.nan)

    for n, idx1d in enumerate(local_tasks):
        j, i = divmod(int(idx1d), ni)
        lonm = lon_np[i]
        latm = lat_np[j]
        d = float(lambda1_np[j, i])

        deg_per_m_lon = deg_per_m_lon_by_j[j]

        # Polar sampling points
        rm = (d * polar.r)[:, None]
        x = rm * polar.cos_t
        y = rm * polar.sin_t

        # Convert meters -> degrees
        lon_x = lonm + x * deg_per_m_lon
        lat_y = latm + y * deg_per_m_lat

        # Patch bounds in degrees
        lon_extent = d * deg_per_m_lon
        lat_extent = d * deg_per_m_lat
        lon_min = lonm - lon_extent - 2 * abs(synbath_res)
        lon_max = lonm + lon_extent + 2 * abs(synbath_res)
        lat_min = latm - lat_extent - 2 * abs(synbath_res)
        lat_max = latm + lat_extent + 2 * abs(synbath_res)

        i0_buf = int(np.floor((lon_min - synbath_lon_buf0) / synbath_res))
        i1_buf = int(np.ceil((lon_max - synbath_lon_buf0) / synbath_res))
        j0 = int(np.floor((lat_min - synbath_lat0) / synbath_res))
        j1 = int(np.ceil((lat_max - synbath_lat0) / synbath_res))

        h_patch = read_lonbuffer_patch(topog, j0, j1, i0_buf, i1_buf, synbath_nlon)
        local_idx_mean_depth[n] = idx1d
        if h_patch is None:
            continue

        lon_patch0 = synbath_lon_buf0 + i0_buf * synbath_res
        lat_patch0 = synbath_lat0 + j0 * synbath_res

        depth = bilinear_interp(
            h_patch, lon_patch0, lat_patch0, synbath_res, lon_x, lat_y
        )

        num = np.nansum(depth * polar.weight)
        if np.isfinite(num):
            local_val_mean_depth[n] = -num / polar.weight_sum

        if print_every and ((n + 1) % print_every == 0):
            print(f"[Rank {rank}] {n+1}/{n_local} tasks done")

    mean_depth_1d = gatherv_indexed(
        local_idx_mean_depth, local_val_mean_depth, total, comm
    )

    if rank == 0:
        mean_depth = mean_depth_1d.reshape(nj, ni)
    else:
        mean_depth = None

    # Now broadcast mean_depth to all ranks
    mean_depth = comm.bcast(mean_depth, root=0)

    # Now for depth variance
    local_idx_var_depth = np.empty(n_local, dtype=np.int64)
    local_val_var_depth = np.empty(n_local, dtype=np.float64)
    local_val_var_depth.fill(np.nan)

    for n, idx1d in enumerate(local_tasks):
        j, i = divmod(int(idx1d), ni)
        lonm = float(lon_np[i])
        latm = float(lat_np[j])
        d = float(lambda1_np[j, i])

        local_idx_var_depth[n] = idx1d

        deg_per_m_lon = float(deg_per_m_lon_by_j[j])

        rm = (d * polar.r)[:, None]
        x = rm * polar.cos_t
        y = rm * polar.sin_t

        lon_x = lonm + x * deg_per_m_lon
        lat_y = latm + y * deg_per_m_lat

        # Interpolate synbath depth at polar points - same as the step for mean depth
        lon_extent = d * deg_per_m_lon
        lat_extent = d * deg_per_m_lat

        lon_min = lonm - lon_extent - 2.0 * abs(synbath_res)
        lon_max = lonm + lon_extent + 2.0 * abs(synbath_res)
        lat_min = latm - lat_extent - 2.0 * abs(synbath_res)
        lat_max = latm + lat_extent + 2.0 * abs(synbath_res)

        i0_buf = int(np.floor((lon_min - synbath_lon_buf0) / synbath_res))
        i1_buf = int(np.ceil((lon_max - synbath_lon_buf0) / synbath_res))
        j0 = int(np.floor((lat_min - synbath_lat0) / synbath_res))
        j1 = int(np.ceil((lat_max - synbath_lat0) / synbath_res))

        h_patch = read_lonbuffer_patch(topog, j0, j1, i0_buf, i1_buf, synbath_nlon)
        if h_patch is None:
            continue

        lon_patch0 = synbath_lon_buf0 + i0_buf * synbath_res
        lat_patch0 = synbath_lat0 + j0 * synbath_res
        depth = bilinear_interp(
            h_patch, lon_patch0, lat_patch0, synbath_res, lon_x, lat_y
        )

        # Interpolate mean_depth to polar points
        lon_x_wrapped = lon_x.copy()
        base_min = lon_np[0]
        lon_x_wrapped = (lon_x_wrapped - base_min) % 360 + base_min

        # Then bilinear on mean_depth
        depth_m = bilinear_interp(
            mean_depth,
            lon_np[0],
            lat_np[0],
            lon_np[1] - lon_np[0],
            lon_x_wrapped,
            lat_y,
        )

        depth_m_neg = -depth_m

        # Compute variance
        variance = (depth - depth_m_neg) ** 2

        num = np.nansum(variance * polar.weight)
        if np.isfinite(num):
            local_val_var_depth[n] = num / polar.weight_sum

        if print_every and ((n + 1) % print_every == 0):
            print(f"[Rank {rank}] {n+1}/{n_local} tasks done for variance")

    depth_var_1d = gatherv_indexed(
        local_idx_var_depth, local_val_var_depth, total, comm
    )

    if rank == 0:
        depth_var = depth_var_1d.reshape(nj, ni)
        return mean_depth, depth_var

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Compute depth variance based on lambda1 computed from WOA23."
    )
    parser.add_argument(
        "--woa23_temp_file",
        type=str,
        required=True,
        help="Path to WOA23 temperature file.",
    )
    parser.add_argument(
        "--woa23_salt_file",
        type=str,
        required=True,
        help="Path to WOA23 salinity file.",
    )
    parser.add_argument(
        "--synbath_file",
        type=str,
        required=True,
        help="Path to synthetic bathymetry file.",
    )
    parser.add_argument(
        "--nmodes", type=int, default=100, help="Number of modes for polar weights."
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=180,
        help="Number of theta divisions for polar weights.",
    )
    parser.add_argument(
        "--earth_radius", type=float, default=6371000.0, help="Earth radius in meters."
    )
    parser.add_argument(
        "--chunk_lat", type=int, default=800, help="Latitude chunk size for processing."
    )
    parser.add_argument(
        "--chunk_lon",
        type=int,
        default=1600,
        help="Longitude chunk size for processing.",
    )
    parser.add_argument(
        "--print_every", type=int, default=100, help="Print progress every N rows."
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.405189e-4,
        help="Tidal frequency in rad/s (default M2)",
    )
    parser.add_argument(
        "--woa_intermediate_file",
        type=str,
        default=None,
        help="Intermediate output file including lambda1, mean_depth, depth_var on WOA grid.",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        temp_ds = xr.open_dataset(args.woa23_temp_file, drop_variables="time")
        salt_ds = xr.open_dataset(args.woa23_salt_file, drop_variables="time")

        sea_water_temp = temp_ds["t_an"].squeeze().transpose("depth", "lat", "lon")
        sea_water_salt = salt_ds["s_an"].squeeze().transpose("depth", "lat", "lon")

        lon = temp_ds["lon"]
        lat = temp_ds["lat"]
        depth = temp_ds["depth"]  # positive down, 1D(102)

        # Build pressure p(depth, lat, lon); gsw uses z (positive up) so pass "-depth"
        p2d = xr.DataArray(
            gsw.p_from_z(-depth.values[:, None], lat.values[None, :]),
            coords={"depth": depth, "lat": lat},
            dims=("depth", "lat"),
            name="pressure",
        )
        p3d = p2d.broadcast_like(sea_water_temp)

        # Compute N^2
        print("Computing N^2...")
        lat3 = lat.values[None, :, None]
        N2, p_mid = gsw.Nsquared(
            sea_water_salt,
            sea_water_temp,
            p3d,
            lat3,
        )

        depth_mid_values = 0.5 * (depth.values[:-1] + depth.values[1:])
        depth_mid = xr.DataArray(
            depth_mid_values,
            dims=("depth_mid",),
            coords={"depth_mid": depth_mid_values},
            name="depth_mid",
        )

        N2_da = xr.DataArray(
            N2,
            dims=("depth_mid", "lat", "lon"),
            coords={
                "depth_mid": depth_mid_values,
                "lat": lat,
                "lon": lon,
            },
            name="N2",
            attrs={"units": "s^-2", "long_name": "Buoyancy frequency squared"},
        )

        print("computing lambda1")
        da_lambda1 = compute_lambda(N2_da, lat, depth, args.omega)

        lon_np = lon.values
        lat_np = lat.values
        lambda1_np = da_lambda1.values

        print("loading high-res topo")
        ds_meta = xr.open_dataset(args.synbath_file, decode_times=False)
        synbath_lon_np = ds_meta["lon"].values
        synbath_lat_np = ds_meta["lat"].values
        synbath_nlon = synbath_lon_np.size
    else:
        lon_np = None
        lat_np = None
        lambda1_np = None
        synbath_lon_np = None
        synbath_lat_np = None
        synbath_nlon = None

    lon_np = comm.bcast(lon_np, root=0)
    lat_np = comm.bcast(lat_np, root=0)
    lambda1_np = comm.bcast(lambda1_np, root=0)

    synbath_lon_np = comm.bcast(synbath_lon_np, root=0)
    synbath_lat_np = comm.bcast(synbath_lat_np, root=0)
    synbath_nlon = comm.bcast(synbath_nlon, root=0)

    mean_depth, depth_var = compute_mean_depth_and_var_points(
        lon_np=lon_np,
        lat_np=lat_np,
        lambda1_np=lambda1_np,
        nmodes=args.nmodes,
        ntheta=args.ntheta,
        RE=args.earth_radius,
        synbath_file=args.synbath_file,
        chunk_lat=args.chunk_lat,
        chunk_lon=args.chunk_lon,
        synbath_lon_np=synbath_lon_np,
        synbath_lat_np=synbath_lat_np,
        synbath_nlon=synbath_nlon,
        comm=comm,
        print_every=args.print_every,
    )

    if rank == 0:
        ds_woa_output = xr.Dataset(
            {
                "lambda1": xr.DataArray(
                    lambda1_np,
                    dims=("lat", "lon"),
                    coords={"lat": lat_np, "lon": lon_np},
                    attrs={
                        "long_name": "Mode-1 internal tide wavelength",
                        "units": "m",
                    },
                ),
                "mean_depth": xr.DataArray(
                    mean_depth,
                    dims=("lat", "lon"),
                    coords={"lat": lat_np, "lon": lon_np},
                    attrs={
                        "long_name": "Gaussian-weighted mean depth using internal tide scale",
                        "units": "m",
                    },
                ),
                "depth_var": xr.DataArray(
                    depth_var,
                    dims=("lat", "lon"),
                    coords={"lat": lat_np, "lon": lon_np},
                    attrs={
                        "long_name": "Gaussian-weighted variance of depth residuals",
                        "units": "m^2",
                    },
                ),
            },
            attrs={
                "nmodes": args.nmodes,
                "ntheta": args.ntheta,
                "earth_radius_m": args.earth_radius,
                "omega_rad_s": args.omega,
            },
        )
        ds_woa_output.to_netcdf(args.woa_intermediate_file)
        print(f"Output written to {args.woa_intermediate_file}")

        # Add provenance metadata and MD5 hashes for input files.
        this_file = os.path.normpath(__file__)
        runcmd = (
            f"mpirun -n $PBS_NCPUS python3 {os.path.basename(this_file)} "
            f"--woa23_temp_file={args.woa23_temp_file} "
            f"--woa23_salt_file={args.woa23_salt_file} "
            f"--synbath_file={args.synbath_file} "
            f"--chunk-lat={args.chunk_lat} "
            f"--chunk-lon={args.chunk_lon} "
            f"--nmodes={args.nmodes} "
            f"--ntheta={args.ntheta} "
            f"--earth-radius={args.earth_radius} "
            f"--omega={args.omega} "
            f"--print-every={args.print_every} "
            f"--woa-intermediate-file={args.woa_intermediate_file} "
        )

        history = get_provenance_metadata(this_file, runcmd)
        global_attrs = {"history": history}
        file_hashes = [
            f"{args.woa23_temp_file} (md5 hash: {md5sum(args.woa23_temp_file)})",
            f"{args.woa23_salt_file} (md5 hash: {md5sum(args.woa23_salt_file)})",
            f"{args.synbath_file} (md5 hash: {md5sum(args.synbath_file)})",
        ]
        global_attrs["inputFile"] = ", ".join(file_hashes)
        ds_woa_output.attrs.update(global_attrs)

        ds_woa_output.to_netcdf(args.woa_intermediate_file)
        print(f"Output written to {args.woa_intermediate_file}")


if __name__ == "__main__":
    main()
