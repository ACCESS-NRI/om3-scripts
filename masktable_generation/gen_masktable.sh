#!/usr/bin/bash
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
#
# This script generates masktables for MOM5/MOM6.
# Help is available by running the script with `./gen_masktable.sh -h`.
#
# For more details, see https://github.com/COSIMA/mom6-panan/wiki/Preparing-inputs-for-a-new-configuration

set -euo pipefail

# Help
Help() {
    echo "Generate MOM6 mask tables."
    echo
    echo "Syntax: $(basename "$0") -g HGRID -t TOPOG [-e -l X:Y] [-r MIN:MAX] [-h]"
    echo
    echo "Required arguments:"
    echo "  -g   Path to ocean_hgrid.nc"
    echo "  -t   Path to ocean_topog.nc"
    echo
    echo "Mode selection (one of the following):"
    echo "  -e            Enable EXACT_LAYOUT mode (default)"
    echo "  -l X:Y        Layout dimensions when EXACT_LAYOUT=True"
    echo
    echo "  -r MIN:MAX    Use min/max PE range (sets EXACT_LAYOUT=False)"
    echo
    echo "  -x PERIODX     Period in the x-direction (default: 360)"
    echo
    echo "  -y PERIODY     Period in the y-direction (default: 360)"
    echo
    echo "Other:"
    echo "  -h            Show this help message"
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
PERIODY=360
LAYOUT_X=""
LAYOUT_Y=""
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
            EXACT_LAYOUT=True
            LAYOUT_X="${OPTARG}"
            LAYOUT_Y="${!OPTIND}"
            OPTIND=$((OPTIND + 1))
            ;;
        # PE range mode: -r MIN MAX
        r)
            EXACT_LAYOUT=False
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

# Required inputs - hgrid.nc and topog.nc
: "${OCEAN_HGRID:?ERROR: -g HGRID is required}"
: "${OCEAN_TOPOG:?ERROR: -t TOPOG is required}"

# Exact layout or PE range mode
if [[ "${EXACT_LAYOUT}" == "True" ]]; then
    : "${LAYOUT_X:?ERROR: -l X Y requires two arguments}"
    : "${LAYOUT_Y:?ERROR: -l X Y requires two arguments}"
else
    : "${MIN_PROCESSORS:?ERROR: -r MIN MAX requires two arguments}"
    : "${MAX_PROCESSORS:?ERROR: -r MIN MAX requires two arguments}"
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
make_solo_mosaic \
    --num_tiles 1 \
    --dir . \
    --mosaic_name ocean_mosaic \
    --tile_file "${HGRID_FILE}" \
    --periodx "${PERIODX}" \
    --periody "${PERIODY}"

# Generate exchange grids
make_quick_mosaic \
    --input_mosaic ocean_mosaic.nc \
    --mosaic_name grid_spec \
    --ocean_topog "${TOPOG_FILE}"

# Generate masktable(s)
if [[ "${EXACT_LAYOUT}" == "True" ]]; then
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
