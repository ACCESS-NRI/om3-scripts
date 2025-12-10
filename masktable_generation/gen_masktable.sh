#!/usr/bin/bash
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
#
# This script generates masktables for MOM5/MOM6.
# Help is available by running:
# `./gen_masktable.sh -h`.
#
# For more details, see https://github.com/COSIMA/mom6-panan/wiki/Preparing-inputs-for-a-new-configuration

set -euo pipefail

# Help
Help() {
    echo "Generate MOM6 mask tables."
    echo
    echo "Syntax: $(basename "$0") -g HGRID -t TOPOG [-l X Y] [-r MIN MAX] [-h]"
    echo
    echo "  -h   Show this help message"
    echo
    echo "Required arguments:"
    echo "  -g   Path to ocean_hgrid.nc"
    echo "  -t   Path to ocean_topog.nc"
    echo
    echo "Mode selection (exactly one of):"
    echo "  -l X Y        Use an exact layout of X by Y processors and generate mask table"
    echo
    echo "  -r MIN MAX    Search over PE range [MIN, MAX] and generate mask tables"
    echo
    echo "Optional:"
    echo "  -x PERIODX    Period in the x-direction (default: 360)"
    echo
    echo "  -y PERIODY    Period in the y-direction (optional; default is aperiodic)"
    echo
    echo "Examples:"
    echo "  # Exact layout mode (default)"
    echo "  ./gen_masktable.sh -g /path/to/hgrid.nc -t /path/to/topog.nc -l 16 32"
    echo
    echo "  # PE range mode"
    echo "  ./gen_masktable.sh -g /path/to/hgrid.nc -t /path/to/topog.nc -r 200 400"
    exit 0
}

# default settings
EXACT_LAYOUT=True
PERIODX=360
PERIODY=""   # no periody by default (aperiodic in y)
LAYOUT_X=""
LAYOUT_Y=""
MODE=""      # "layout" or "range"
MIN_PROCESSORS=""
MAX_PROCESSORS=""
OCEAN_HGRID=""
OCEAN_TOPOG=""

while getopts ":hg:t:l:r:x:y:" option; do
    case "${option}" in
        h)
            Help
            ;;
        g)
            OCEAN_HGRID="${OPTARG}"
            ;;
        t)
            OCEAN_TOPOG="${OPTARG}"
            ;;
        # EXACT layout mode: -l X Y
        l)
            MODE="layout"
            LAYOUT_X="${OPTARG}"
            LAYOUT_Y="${!OPTIND}"
            OPTIND=$((OPTIND + 1))
            ;;
        # PE range mode: -r MIN MAX
        r)
            MODE="range"
            MIN_PROCESSORS="${OPTARG}"
            MAX_PROCESSORS="${!OPTIND}"
            OPTIND=$((OPTIND + 1))
            ;;
        x)
            PERIODX="${OPTARG}"
            ;;
        y)
            PERIODY="${OPTARG}"
            ;;
        \?)
            echo "ERROR: Invalid option -${OPTARG}" >&2
            exit 1
            ;;
        :)
            echo "ERROR: Missing argument for -${OPTARG}" >&2
            exit 1
            ;;
    esac
done

# load modules
module use /g/data/vk83/modules
module load model-tools/fre-nctools/2024.05-1
module load nco

# Required inputs - hgrid.nc and topog.nc
: "${OCEAN_HGRID:?ERROR: -g HGRID is required}"
: "${OCEAN_TOPOG:?ERROR: -t TOPOG is required}"

if [[ -z "${MODE}" ]]; then
    echo "ERROR: You must specify exactly one of -l X Y (layout) or -r MIN MAX (PE range)." >&2
    exit 1
fi

if [[ "${MODE}" == "layout" ]]; then
    : "${LAYOUT_X:?ERROR: -l X Y requires two arguments (X and Y).}"
    : "${LAYOUT_Y:?ERROR: -l X Y requires two arguments (X and Y).}"
elif [[ "${MODE}" == "range" ]]; then
    : "${MIN_PROCESSORS:?ERROR: -r MIN MAX requires two arguments (MIN and MAX).}"
    : "${MAX_PROCESSORS:?ERROR: -r MIN MAX requires two arguments (MIN and MAX).}"
fi

HGRID_FILE=$(basename "${OCEAN_HGRID}")
TOPOG_FILE=$(basename "${OCEAN_TOPOG}")

# `make_quick_mosaic` requires local hgrid.nc
echo "-- Copying ${OCEAN_HGRID} -> ./${HGRID_FILE}"
if [[ -e "${HGRID_FILE}" ]]; then
    rm -f "${HGRID_FILE}"
fi
cp "${OCEAN_HGRID}" "${HGRID_FILE}"

# `make_quick_mosaic` requires additional ntiles dimension in topog.nc
# hence copy to local directory and add ntiles dimension
echo "-- Copying ${OCEAN_TOPOG} -> ./${TOPOG_FILE}"
if [[ -e "${TOPOG_FILE}" ]]; then
    rm -f "${TOPOG_FILE}"
fi
cp "${OCEAN_TOPOG}" "${TOPOG_FILE}" 

# Add ntiles dimension to topog file
ncap2 -s 'defdim("ntiles",1)' -A "${TOPOG_FILE}" "${TOPOG_FILE}"

# Generate ocean mosaic
msg=(
    --num_tiles 1
    --dir .
    --mosaic_name ocean_mosaic
    --tile_file "${HGRID_FILE}"
    --periodx "${PERIODX}"
)

# Only append --periody if user explicitly provided -y
if [[ -n "${PERIODY}" ]]; then
    msg+=(--periody "${PERIODY}")
fi

make_solo_mosaic "${msg[@]}"

# Generate exchange grids
make_quick_mosaic \
    --input_mosaic ocean_mosaic.nc \
    --mosaic_name grid_spec \
    --ocean_topog "${TOPOG_FILE}"

# Generate masktable(s)
if [[ "${MODE}" == "layout" ]]; then
    echo "-- Running check_mask with layout ${LAYOUT_X},${LAYOUT_Y}"
    check_mask \
        --grid_file ocean_mosaic.nc \
        --ocean_topog "${OCEAN_TOPOG}" \
        --layout "${LAYOUT_X},${LAYOUT_Y}"
else
    echo "-- Running check_mask with PE range ${MIN_PROCESSORS}-${MAX_PROCESSORS}"
    check_mask \
        --grid_file ocean_mosaic.nc \
        --ocean_topog "${OCEAN_TOPOG}" \
        --min_pe "${MIN_PROCESSORS}" \
        --max_pe "${MAX_PROCESSORS}"
fi
