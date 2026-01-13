#!/usr/bin/env python3

import argparse
import numpy as np
from mpi4py import MPI
import xarray as xr
import gsw
from dataclasses import dataclass


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


def bilinear_interp(field, lon0, lat0, dres, lon_x, lat_y):
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
) -> np.ndarray:
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


def split_rows(nj, size, rank):
    block_size = nj // size
    rem = nj % size
    j_start = rank * block_size + min(rank, rem)
    j_count = block_size + (1 if rank < rem else 0)
    j_end = j_start + j_count
    return block_size, rem, j_start, j_end, j_count


def gatherv_indexed(local_idx, local_val, total_size, comm):
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

        all_idx = np.empty(np.sum(counts))
        all_val = np.empty(np.sum(counts))

    else:
        counts = None
        displs = None
        all_idx = None
        all_val = None

    comm.Gatherv(local_idx, (all_idx, (counts, displs)), root=0)
    comm.Gatherv(local_val, (all_val, (counts, displs)), root=0)

    if rank != 0:
        return None

    out1d = np.full(total_size, np.nan)
    out1d[all_idx] = all_val
    return out1d


def compute_mean_depth_points(
    lon_np,
    lat_np,
    lambda1_np,
    nmodes,
    ntheta,
    RE,
    synbath_file,
    chunk_lat,
    chunk_lon,
    synbath_lon_np,
    synbath_lat_np,
    synbath_nlon,
    comm,
    print_every=1,
):
    rank = comm.Get_rank()
    size = comm.Get_size()

    ni = lon_np.size
    nj = lat_np.size
    total = nj * ni

    if rank == 0:
        mask = lambda1_np > 0.0
        tasks = np.flatnonzero(mask.ravel()).astype(np.int64)
        print(f"[Rank 0] total points={total}, tasks(ocean)={tasks.size}, ranks={size}")
    else:
        tasks = None

    tasks = comm.bcast(tasks, root=0)

    local_tasks = tasks[rank::size]
    n_local = local_tasks.size
    if rank == 0:
        print(f"[Rank 0] avg tasks per rank ~ {tasks.size/size:.1f}")
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
    local_idx = np.empty(n_local)
    local_val = np.empty(n_local)
    local_val.fill(np.nan)

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
        if h_patch is None:
            continue

        lon_patch0 = synbath_lon_buf0 + i0_buf * synbath_res
        lat_patch0 = synbath_lat0 + j0 * synbath_res

        depth = bilinear_interp(
            h_patch, lon_patch0, lat_patch0, synbath_res, lon_x, lat_y
        )

        num = np.nansum(depth * polar.weight)
        if np.isfinite(num):
            local_val[n] = -num / polar.weight_sum

        local_idx[n] = idx1d

        if print_every and ((n + 1) % print_every == 0):
            print(f"[Rank {rank}] {n+1}/{n_local} tasks done")

    out1d = gatherv_indexed(local_idx, local_val, total, comm)
    if rank != 0:
        return None

    return out1d.reshape(nj, ni)


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean depth based on lambda1 computed from WOA23."
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
        "--print_every", type=int, default=1, help="Print progress every N rows."
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.405189e-4,
        help="Tidal frequency in rad/s (default M2)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="mean_depth_lambda1_filtered.nc",
        help="Output for mean depth.",
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
        print(f"[Rank {rank}] Computing N^2...")
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

        print(f"[Rank {rank}] computing lambda1")
        da_lambda1 = compute_lambda(N2_da, lat, depth, args.omega)

        lon_np = lon.values
        lat_np = lat.values
        lambda1_np = da_lambda1.values

        print(f"[Rank {rank}] loading high-res topo")
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

    mean_depth = compute_mean_depth_points(
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
        ds_out_mean_depth = xr.Dataset(
            {
                "mean_depth": xr.DataArray(
                    mean_depth,
                    dims=("lat", "lon"),
                    coords={"lat": lat_np, "lon": lon_np},
                    attrs={
                        "long_name": "Gaussian-weighted mean depth using internal tide scale",
                        "units": "m",
                    },
                ),
            }
        )
        ds_out_mean_depth.to_netcdf(args.output_file)
        print(f"Output written to {args.output_file}")


if __name__ == "__main__":
    main()
