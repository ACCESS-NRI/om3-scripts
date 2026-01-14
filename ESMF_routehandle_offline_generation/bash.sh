#!/bin/bash
set -euo pipefail

module purge
module load intel-compiler/2021.10.0
module load openmpi/4.1.7
module use /g/data/vk83/modules && module load esmf/v8.7.0

module -t list

F90I=$(awk -F= '/^ESMF_F90COMPILEPATHS/{print $2}' "$ESMFMKFILE")
F90L=$(awk -F= '/^ESMF_F90LINKPATHS/{print $2}'    "$ESMFMKFILE")
F90LIBS=$(awk -F= '/^ESMF_F90ESMFLINKLIBS/{print $2}' "$ESMFMKFILE")

mpif90 -c -O2 $F90I ESMF_route_handle_offline_generation.f90
mpif90 -O2 $F90L ESMF_route_handle_offline_generation.o $F90LIBS -o ESMF_route_handle_offline_generation

ESMF_LIBSDIR=$(awk -F= '/^ESMF_LIBSDIR/ {gsub(/[ \t]/,"",$2);print $2}' "$ESMFMKFILE")
export LD_LIBRARY_PATH="${ESMF_LIBSDIR}:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$(nc-config --libdir):$(nf-config --prefix)/lib:$LD_LIBRARY_PATH"
export PIO_LIBDIR=/g/data/vk83/apps/spack/0.22/release/linux-rocky8-x86_64_v4/intel-2021.10.0/parallelio-2.6.2-no7vsc4leb7pja5dv24ittgldfraqiui/lib
export LD_LIBRARY_PATH="$PIO_LIBDIR:$LD_LIBRARY_PATH"

mpirun -n 24 \
  ./ESMF_route_handle_offline_generation \
  --mesh_atm /g/data/vk83/configurations/inputs/access-om3/share/meshes/global.100km/2024.01.25/access-om2-100km-nomask-ESMFmesh.nc \
  --mesh_ice /g/data/vk83/configurations/inputs/access-om3/share/meshes/global.100km/2024.01.25/access-om2-100km-ESMFmesh.nc \
  --mesh_ocn /g/data/vk83/configurations/inputs/access-om3/share/meshes/global.100km/2024.01.25/access-om2-100km-ESMFmesh.nc \
  --mesh_rof /g/data/vk83/configurations/inputs/access-om3/share/meshes/global.100km/2024.01.25/access-om2-100km-nomask-ESMFmesh.nc