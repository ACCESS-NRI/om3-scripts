#!/usr/bin/env bash
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
#
# This script generates masktables for MOM5/MOM6.
# Help is available by running:
# `./gen_masktable.sh -h`.
#
# For more details, see
#   1. https://github.com/COSIMA/mom6-panan/wiki/Preparing-inputs-for-a-new-configuration
#   2. https://github.com/ACCESS-NRI/MOM6/issues/42#issuecomment-5337278337

set -euo pipefail

# Captured before getopts runs (which does not modify $@) so it can be
# recorded as provenance in the generated mask tables.
ORIGINAL_INVOCATION="$0 $*"

# Help
Help() {
    echo "Generate MOM5/MOM6 mask tables."
    echo
    echo "Syntax: $(basename "$0") -g HGRID -t TOPOG [-l X Y] [-r MIN MAX] [-m MODEL] [-a] [-h]"
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
    echo "  -m MODEL      Target model: mom5 or mom6 (default: mom6)"
    echo "                Both models read the same mask-table format, but only"
    echo "                MOM6 constrains which tables are usable (see -a). With"
    echo "                -m mom5 the MOM6 check is skipped and every table"
    echo "                produced by check_mask is kept as-is."
    echo
    echo "  -a            Automatically adjust the number of masked land-only"
    echo "                domains when required for MOM6 compatibility."
    echo "                Supported with both -l and -r. MOM6 only."
    echo
    echo "  -x PERIODX    Period in the x-direction (default: 360)"
    echo
    echo "  -y PERIODY    Period in the y-direction (optional; default is aperiodic)"
    echo
    echo "Examples:"
    echo "  # Exact layout mode (default)"
    echo "  ./gen_masktable.sh -g /path/to/hgrid.nc -t /path/to/topog.nc -l 16 32"
    echo
    echo "  # Exact layout mode with automatic land-only domain adjustment"
    echo "  ./gen_masktable.sh -g /path/to/hgrid.nc -t /path/to/topog.nc -l 16 32 -a"
    echo
    echo "  # PE range mode"
    echo "  ./gen_masktable.sh -g /path/to/hgrid.nc -t /path/to/topog.nc -r 200 400"
    echo "  # PE range mode with automatic land-only domain adjustment"
    echo "  ./gen_masktable.sh -g /path/to/hgrid.nc -t /path/to/topog.nc -r 200 400 -a"
    echo
    echo "  # MOM5: generate tables without the MOM6 compatibility check"
    echo "  ./gen_masktable.sh -g /path/to/hgrid.nc -t /path/to/topog.nc -r 200 400 -m mom5"
    echo
    exit 0
}

# Read a dimension from a NetCDF file
get_dim() {
    local file=$1
    local dim=$2

    ncdump -h "${file}" | awk -v dim="${dim}" '
        $1 == dim && $2 == "=" {
            gsub(/;/, "", $3)
            print $3
        }
    '
}

# Reproduce MOM_define_layout() from MOM6
# https://github.com/ACCESS-NRI/MOM6/blob/6432010e3ab29df43994adabb413b69fe718d94c/src/framework/MOM_domains.F90#L466-L486
mom_define_layout() {
    local nx=$1
    local ny=$2
    local npes=$3

    local idiv
    local jdiv

    idiv=$(
        awk \
            -v npes="${npes}" \
            -v nx="${nx}" \
            -v ny="${ny}" \
            'BEGIN {
                value = sqrt(npes * nx / ny)

                # Equivalent to NINT() for positive values.
                rounded = int(value + 0.5)

                if (rounded < 1) {
                    rounded = 1
                }

                print rounded
            }'
    )

    while (( npes % idiv != 0 )); do
        idiv=$((idiv - 1))
    done

    jdiv=$((npes / idiv))

    echo "${idiv} ${jdiv}"
}

# Check whether the automatically selected MOM6 unmasked layout is compatible
# with the factor-2 coarsened domain created by create_MOM_domain().
#
# Returns 0 if compatible, 1 if not compatible.
mom6_layout_is_compatible() {
    local nx=$1
    local ny=$2
    local npes=$3

    local unmasked_x
    local unmasked_y

    read -r unmasked_x unmasked_y < <(
        mom_define_layout "${nx}" "${ny}" "${npes}"
    )

    # MOM6 creates mpp_domain_d2 with coarsen=2.
    local coarse_nx=$((nx / 2))
    local coarse_ny=$((ny / 2))

    if (( unmasked_x > coarse_nx || unmasked_y > coarse_ny )); then
        return 1
    fi

    return 0
}

# Read metadata from a mask table
read_masktable_metadata() {
    local file=$1

    local n_mask
    local layout_x
    local layout_y

    n_mask=$(sed -n '1p' "${file}")

    read -r layout_x layout_y < <(
        sed -n '2p' "${file}" |
            tr ',' ' '
    )

    echo "${n_mask} ${layout_x} ${layout_y}"
}

# Calculate the approximate processor overhead (in percent)
approx_processor_overhead() {
    local original_active=$1
    local adjusted_active=$2

    awk \
        -v original="${original_active}" \
        -v adjusted="${adjusted_active}" \
        'BEGIN {
            if (adjusted <= 0) {
                printf "0.000"
                exit
            }

            overhead = (1.0 - original / adjusted) * 100.0
            printf "%.3f", overhead
        }'
}

# Find the largest compatible number of masks <= max_masks.
find_compatible_mask_count() {
    local nx=$1
    local ny=$2
    local layout_x=$3
    local layout_y=$4
    local max_masks=$5

    local n_mask
    local active_pes

    for ((n_mask=max_masks; n_mask>=0; n_mask--)); do
        active_pes=$((layout_x * layout_y - n_mask))

        if mom6_layout_is_compatible \
            "${nx}" \
            "${ny}" \
            "${active_pes}"
        then
            echo "${n_mask}"
            return 0
        fi
    done

    return 1
}

# Create a new mask table containing fewer masked land-only domains
write_adjusted_masktable() {
    local input=$1
    local new_n_mask=$2

    local old_n_mask
    local layout_x
    local layout_y

    read -r old_n_mask layout_x layout_y < <(
        read_masktable_metadata "${input}"
    )

    local output="mask_table.${new_n_mask}.${layout_x}x${layout_y}"

    awk \
        -v keep="${new_n_mask}" \
        -v new_count="${new_n_mask}" \
        '
        NR == 1 {
            print new_count
            next
        }

        NR == 2 {
            print
            next
        }

        NR >= 3 && NR < 3 + keep {
            print
        }
        ' \
        "${input}" > "${output}"

    echo "${output}"
}

# Print details of the MOM6 compatibility calculation.
#
# Returns:
#   0 = compatible
#   1 = incompatible
validate_masktable() {
    local file=$1
    local nx=$2
    local ny=$3

    local n_mask
    local layout_x
    local layout_y

    read -r n_mask layout_x layout_y < <(
        read_masktable_metadata "${file}"
    )

    local total_domains=$((layout_x * layout_y))
    local active_pes=$((total_domains - n_mask))

    local unmasked_x
    local unmasked_y

    read -r unmasked_x unmasked_y < <(
        mom_define_layout "${nx}" "${ny}" "${active_pes}"
    )

    local coarse_nx=$((nx / 2))
    local coarse_ny=$((ny / 2))

    echo
    echo "-- MOM6 compatibility check: ${file}"
    echo "   Logical layout:           ${layout_x} x ${layout_y}"
    echo "   Total logical domains:    ${total_domains}"
    echo "   Masked land domains:      ${n_mask}"
    echo "   Active PEs:               ${active_pes}"
    echo "   MOM6 unmasked layout:     ${unmasked_x} x ${unmasked_y}"
    echo "   Coarsened global domain:  ${coarse_nx} x ${coarse_ny}"

    if (( unmasked_x > coarse_nx || unmasked_y > coarse_ny )); then
        echo "   MOM6 compatibility:       FAILED"
        echo

        echo "ERROR: ${file} is incompatible with MOM6's factor-2" >&2
        echo "coarsened unmasked domain." >&2
        echo >&2
        echo "MOM6 would attempt to decompose:" >&2
        echo "    ${coarse_nx} x ${coarse_ny}" >&2
        echo "using the unmasked layout:" >&2
        echo "    ${unmasked_x} x ${unmasked_y}" >&2
        echo >&2

        if (( unmasked_x > coarse_nx )); then
            echo "The x layout (${unmasked_x}) exceeds the number of coarse" >&2
            echo "x grid points (${coarse_nx})." >&2
            echo >&2
        fi

        if (( unmasked_y > coarse_ny )); then
            echo "The y layout (${unmasked_y}) exceeds the number of coarse" >&2
            echo "y grid points (${coarse_ny})." >&2
            echo >&2
        fi

        echo "This would produce zero-sized FMS domains and can trigger:" >&2
        echo >&2
        echo "  MPP_DEFINE_DOMAINS(mpp_compute_extent):" >&2
        echo "  domain extents must be positive definite." >&2

        return 1
    fi

    echo "   MOM6 compatibility:       OK"

    return 0
}

# Record how a mask table was generated as '#'-prefixed comment lines,
# inserted immediately after the mandatory nmask/layout header (lines 1-2).
#
# FMS's parse_mask_table (shared by MOM5 and MOM6) reads lines 1 and 2 at
# fixed positions, then scans from line 3 onward for mask-list entries,
# skipping any line whose first character is '#'. Comments are therefore
# only safe from line 3 onward, never before it.
add_provenance() {
    local file=$1
    shift

    local tmp
    tmp=$(mktemp)

    {
        sed -n '1,2p' "${file}"
        for line in "${PROVENANCE_COMMON[@]}" "$@"; do
            echo "# ${line}"
        done
        sed -n '3,$p' "${file}"
    } > "${tmp}"

    mv "${tmp}" "${file}"
}


# default settings
PERIODX=360
PERIODY=""   # no periody by default (aperiodic in y)
LAYOUT_X=""
LAYOUT_Y=""
MODE=""      # "layout" or "range"
MIN_PROCESSORS=""
MAX_PROCESSORS=""
OCEAN_HGRID=""
OCEAN_TOPOG=""
AUTO_ADJUST=false
TARGET_MODEL="mom6"

while getopts ":ham:g:t:l:r:x:y:" option; do
    case "${option}" in
        h)
            Help
            ;;
        a)
            AUTO_ADJUST=true
            ;;
        m)
            case "${OPTARG,,}" in
                mom5)
                    TARGET_MODEL="mom5"
                    ;;
                mom6)
                    TARGET_MODEL="mom6"
                    ;;
                *)
                    echo "ERROR: -m must be one of: mom5, mom6 (got '${OPTARG}')" >&2
                    exit 1
                    ;;
            esac
            ;;
        g)
            OCEAN_HGRID="${OPTARG}"
            ;;
        t)
            OCEAN_TOPOG="${OPTARG}"
            ;;
        # EXACT layout mode: -l X Y
        l)
            if [[ -n "${MODE}" ]]; then
                echo "ERROR: Specify only one of -l or -r." >&2
                exit 1
            fi

            MODE="layout"
            LAYOUT_X="${OPTARG}"

            if (( OPTIND > $# )); then
                echo "ERROR: -l requires two arguments: -l X Y" >&2
                exit 1
            fi

            LAYOUT_Y="${!OPTIND}"
            OPTIND=$((OPTIND + 1))
            ;;
        # PE range mode: -r MIN MAX
        r)
            if [[ -n "${MODE}" ]]; then
                echo "ERROR: Specify only one of -l or -r." >&2
                exit 1
            fi

            MODE="range"
            MIN_PROCESSORS="${OPTARG}"

            if (( OPTIND > $# )); then
                echo "ERROR: -r requires two arguments: -r MIN MAX" >&2
                exit 1
            fi

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

# Required inputs - hgrid.nc and topog.nc
: "${OCEAN_HGRID:?ERROR: -g HGRID is required}"
: "${OCEAN_TOPOG:?ERROR: -t TOPOG is required}"

if [[ ! -f "${OCEAN_HGRID}" ]]; then
    echo "ERROR: HGRID does not exist: ${OCEAN_HGRID}" >&2
    exit 1
fi

if [[ ! -f "${OCEAN_TOPOG}" ]]; then
    echo "ERROR: TOPOG does not exist: ${OCEAN_TOPOG}" >&2
    exit 1
fi

if [[ -z "${MODE}" ]]; then
    echo "ERROR: You must specify exactly one of:" >&2
    echo "  -l X Y" >&2
    echo "  -r MIN MAX" >&2
    exit 1
fi

if [[ "${MODE}" == "layout" ]]; then
    : "${LAYOUT_X:?ERROR: -l X Y requires two arguments}"
    : "${LAYOUT_Y:?ERROR: -l X Y requires two arguments}"

    if [[ ! "${LAYOUT_X}" =~ ^[0-9]+$ ||
          ! "${LAYOUT_Y}" =~ ^[0-9]+$ ||
          "${LAYOUT_X}" -le 0 ||
          "${LAYOUT_Y}" -le 0 ]]
    then
        echo "ERROR: -l X Y requires positive integers." >&2
        exit 1
    fi

elif [[ "${MODE}" == "range" ]]; then
    : "${MIN_PROCESSORS:?ERROR: -r MIN MAX requires two arguments}"
    : "${MAX_PROCESSORS:?ERROR: -r MIN MAX requires two arguments}"

    if [[ ! "${MIN_PROCESSORS}" =~ ^[0-9]+$ ||
          ! "${MAX_PROCESSORS}" =~ ^[0-9]+$ ||
          "${MIN_PROCESSORS}" -le 0 ||
          "${MAX_PROCESSORS}" -le 0 ]]
    then
        echo "ERROR: -r MIN MAX requires positive integers." >&2
        exit 1
    fi

    if (( MIN_PROCESSORS > MAX_PROCESSORS )); then
        echo "ERROR: MIN must not exceed MAX." >&2
        exit 1
    fi
fi

# The MOM6 land-domain adjustment has no meaning for MOM5, which does not
# impose the coarsened-domain constraint that -a works around.
if [[ "${TARGET_MODEL}" == "mom5" && "${AUTO_ADJUST}" == true ]]; then
    echo "ERROR: -a is a MOM6-only adjustment and cannot be combined with -m mom5." >&2
    echo "       MOM5 imposes no constraint on the mask table, so no" >&2
    echo "       adjustment is required." >&2
    exit 1
fi

# Periods are passed straight through to make_solo_mosaic, which expects a
# non-negative number of degrees (0 means aperiodic in that direction).
if [[ ! "${PERIODX}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: -x PERIODX must be a non-negative number (got '${PERIODX}')." >&2
    exit 1
fi

if [[ -n "${PERIODY}" && ! "${PERIODY}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: -y PERIODY must be a non-negative number (got '${PERIODY}')." >&2
    exit 1
fi


# load modules
module use /g/data/vk83/modules
module load model-tools/fre-nctools/2024.05-1
module load nco
module load netcdf

# -----------------------------------------------------------------------------
# Determine MOM grid dimensions
# -----------------------------------------------------------------------------

NX=$(get_dim "${OCEAN_TOPOG}" "nx")
NY=$(get_dim "${OCEAN_TOPOG}" "ny")

if [[ -z "${NX}" || -z "${NY}" ]]; then
    echo "ERROR: Could not determine nx/ny from ${OCEAN_TOPOG}" >&2
    exit 1
fi

echo "-- MOM grid size: ${NX} x ${NY}"

# Provenance recorded in every generated mask table (see add_provenance()).
PROVENANCE_COMMON=(
    "Generated by masktable_generation/gen_masktable.sh (ACCESS-NRI/om3-scripts)"
    "Date: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    "Command: ${ORIGINAL_INVOCATION}"
    "hgrid: ${OCEAN_HGRID}"
    "topog: ${OCEAN_TOPOG}"
    "Grid size: ${NX} x ${NY}"
    "Target model: ${TARGET_MODEL}"
)

# -----------------------------------------------------------------------------
# Prepare local input files
# -----------------------------------------------------------------------------
HGRID_FILE=$(basename "${OCEAN_HGRID}")
TOPOG_FILE=$(basename "${OCEAN_TOPOG}")

# If the source file already lives in the working directory under the same
# name, use a distinctly-named local copy instead of clobbering the source.
if [[ -e "${HGRID_FILE}" ]] && [[ "${OCEAN_HGRID}" -ef "${HGRID_FILE}" ]]; then
    HGRID_FILE="local_${HGRID_FILE}"
fi
if [[ -e "${TOPOG_FILE}" ]] && [[ "${OCEAN_TOPOG}" -ef "${TOPOG_FILE}" ]]; then
    TOPOG_FILE="local_${TOPOG_FILE}"
fi

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
solo_mosaic_args=(
    --num_tiles 1
    --dir .
    --mosaic_name ocean_mosaic
    --tile_file "${HGRID_FILE}"
    --periodx "${PERIODX}"
)

# Only append --periody if user explicitly provided -y
if [[ -n "${PERIODY}" ]]; then
    solo_mosaic_args+=(--periody "${PERIODY}")
fi

make_solo_mosaic "${solo_mosaic_args[@]}"

# Generate exchange grids
make_quick_mosaic \
    --input_mosaic ocean_mosaic.nc \
    --mosaic_name grid_spec \
    --ocean_topog "${TOPOG_FILE}"

# Generate masktable(s)
#
# check_mask is given the original topog file rather than the local copy: it
# reads the `depth` field and takes its tile count from the mosaic file, so it
# does not need the `ntiles` dimension added above for make_quick_mosaic.

# Keep check_mask's normal terminal output, while also recording it so that
# the generated mask-table filenames can be obtained directly from its:
#
#   used=..., masked=..., layout=...
#
# output lines.
CHECK_MASK_LOG=$(mktemp)
cleanup() {
    rm -rf "${CHECK_MASK_LOG}"
}

trap cleanup EXIT

if [[ "${MODE}" == "layout" ]]; then
    echo "-- Running check_mask with layout ${LAYOUT_X},${LAYOUT_Y}"
    check_mask \
        --grid_file ocean_mosaic.nc \
        --ocean_topog "${OCEAN_TOPOG}" \
        --layout "${LAYOUT_X},${LAYOUT_Y}" \
        2>&1 | tee "${CHECK_MASK_LOG}"
else
    echo "-- Running check_mask with PE range ${MIN_PROCESSORS}-${MAX_PROCESSORS}"
    check_mask \
        --grid_file ocean_mosaic.nc \
        --ocean_topog "${OCEAN_TOPOG}" \
        --min_pe "${MIN_PROCESSORS}" \
        --max_pe "${MAX_PROCESSORS}" \
        2>&1 | tee "${CHECK_MASK_LOG}"
fi

# -----------------------------------------------------------------------------
# Locate mask tables generated by this invocation
# -----------------------------------------------------------------------------

mapfile -t GENERATED_MASKTABLES < <(
    sed -nE \
        's/.*masked=([0-9]+),[[:space:]]*layout=([0-9]+),[[:space:]]*([0-9]+).*/mask_table.\1.\2x\3/p' \
        "${CHECK_MASK_LOG}" |
        sort -u
)

if (( ${#GENERATED_MASKTABLES[@]} == 0 )); then
    echo "ERROR: check_mask did not generate any mask tables." >&2
    exit 1
fi

# Make sure the files reported by check_mask actually exist.
for masktable in "${GENERATED_MASKTABLES[@]}"; do
    if [[ ! -f "${masktable}" ]]; then
        echo "ERROR: check_mask reported ${masktable}, but the file does not exist." >&2
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# MOM5: nothing further to check
#
# MOM5 reads the same mask-table format as MOM6 (both use the FMS
# parse_mask_table routine), but it creates a single ocean domain and has no
# equivalent of MOM6's separate unmasked domain or its factor-2 coarsened clone.
# The constraint checked below therefore does not apply, and every table
# check_mask produced is usable as-is.
# -----------------------------------------------------------------------------

if [[ "${TARGET_MODEL}" == "mom5" ]]; then
    for masktable in "${GENERATED_MASKTABLES[@]}"; do
        add_provenance "${masktable}" \
            "MOM6 compatibility check skipped (target model: mom5)"
    done

    echo
    echo "-- Mask-table generation complete (target model: MOM5)"
    echo "   Generated by check_mask:   ${#GENERATED_MASKTABLES[@]}"

    for masktable in "${GENERATED_MASKTABLES[@]}"; do
        echo "     ${masktable}"
    done

    echo
    echo "   NOTE: The MOM6 coarsened-domain constraint does not apply to MOM5,"
    echo "   so no compatibility check or adjustment was performed."

    exit 0
fi

# -----------------------------------------------------------------------------
# MOM6 compatibility validation / adjustment
#
# When a mask table is present, MOM6 additionally builds an *unmasked* domain to
# write the complete ocean geometry, whose layout is derived from the active PE
# count via MOM_define_layout(). create_MOM_domain() then unconditionally clones
# a factor-2 coarsened copy of it. A skewed unmasked layout that is valid at full
# resolution can therefore become invalid on the nx/2 x ny/2 coarsened domain,
# producing zero-sized FMS domains.
# -----------------------------------------------------------------------------

N_COMPATIBLE=0
N_INCOMPATIBLE=0
N_ADJUSTED=0
N_UNADJUSTABLE=0

for masktable in "${GENERATED_MASKTABLES[@]}"; do

    # -------------------------------------------------------------------------
    # Already compatible
    # -------------------------------------------------------------------------

    if validate_masktable "${masktable}" "${NX}" "${NY}"; then
        add_provenance "${masktable}" "MOM6 compatibility: OK"
        N_COMPATIBLE=$((N_COMPATIBLE + 1))
        continue
    fi

    N_INCOMPATIBLE=$((N_INCOMPATIBLE + 1))

    read -r n_mask layout_x layout_y < <(
        read_masktable_metadata "${masktable}"
    )

    total_domains=$((layout_x * layout_y))
    original_active_pes=$((total_domains - n_mask))

    compatible_n_mask=""

    if ! compatible_n_mask=$(
        find_compatible_mask_count \
            "${NX}" \
            "${NY}" \
            "${layout_x}" \
            "${layout_y}" \
            "${n_mask}"
    ); then
        echo
        echo "WARNING: Could not find a MOM6-compatible adjustment for:"
        echo "         ${masktable}"
        echo

        add_provenance "${masktable}" \
            "MOM6 compatibility: FAILED; no compatible mask-domain count found"

        N_UNADJUSTABLE=$((N_UNADJUSTABLE + 1))
        continue
    fi

    compatible_active_pes=$((layout_x * layout_y - compatible_n_mask))
    extra_land_pes=$((compatible_active_pes - original_active_pes))

    processor_overhead=$(
        approx_processor_overhead \
            "${original_active_pes}" \
            "${compatible_active_pes}"
    )

    read -r compatible_unmasked_x compatible_unmasked_y < <(
        mom_define_layout \
            "${NX}" \
            "${NY}" \
            "${compatible_active_pes}"
    )

    echo
    echo "-- Nearest MOM6-compatible mask configuration"
    echo "   Logical layout:                ${layout_x} x ${layout_y}"
    echo "   Original masked domains:       ${n_mask}"
    echo "   Compatible masked domains:     ${compatible_n_mask}"
    echo "   Original active PEs:           ${original_active_pes}"
    echo "   Compatible active PEs:         ${compatible_active_pes}"
    echo "   Extra retained land-only PEs:  ${extra_land_pes}"
    echo "   Compatible unmasked layout:    ${compatible_unmasked_x} x ${compatible_unmasked_y}"
    echo "   Approx. processor overhead:     ${processor_overhead} %"
    echo
    echo "   NOTE: The processor overhead only represents the additional"
    echo "   allocation from retaining land-only PEs. It is not an estimate"
    echo "   of wall-clock performance loss."
    echo

    # -------------------------------------------------------------------------
    # Validation-only mode
    # -------------------------------------------------------------------------

    if [[ "${AUTO_ADJUST}" != true ]]; then
        echo "   No adjusted mask table generated because -a was not specified."
        echo

        add_provenance "${masktable}" \
            "MOM6 compatibility: FAILED; re-run with -a to generate a compatible mask table"

        continue
    fi

    # -------------------------------------------------------------------------
    # Auto-adjust
    # -------------------------------------------------------------------------

    adjusted_masktable=$(
        write_adjusted_masktable \
            "${masktable}" \
            "${compatible_n_mask}"
    )

    echo "-- Generated adjusted MOM6-compatible mask table"
    echo "   check_mask result: ${masktable}"
    echo "   Final mask table:  ${adjusted_masktable}"
    echo

    if validate_masktable \
        "${adjusted_masktable}" \
        "${NX}" \
        "${NY}"
    then
        add_provenance "${masktable}" \
            "MOM6 compatibility: FAILED; superseded by ${adjusted_masktable}"
        add_provenance "${adjusted_masktable}" \
            "Derived from ${masktable} (masked domains ${n_mask} -> ${compatible_n_mask}) for MOM6 compatibility"

        N_ADJUSTED=$((N_ADJUSTED + 1))
    else
        echo "ERROR: Adjusted mask table unexpectedly failed validation:" >&2
        echo "       ${adjusted_masktable}" >&2

        rm -f "${adjusted_masktable}"

        add_provenance "${masktable}" \
            "MOM6 compatibility: FAILED; automatic adjustment also failed validation"

        N_UNADJUSTABLE=$((N_UNADJUSTABLE + 1))
    fi

done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo
echo "-- Mask-table generation complete (target model: MOM6)"
echo "   Generated by check_mask:   ${#GENERATED_MASKTABLES[@]}"
echo "   Already MOM6-compatible:   ${N_COMPATIBLE}"
echo "   Initially incompatible:    ${N_INCOMPATIBLE}"

if [[ "${AUTO_ADJUST}" == true ]]; then
    echo "   Automatically adjusted:    ${N_ADJUSTED}"
fi

if (( N_UNADJUSTABLE > 0 )); then
    echo "   Could not adjust:          ${N_UNADJUSTABLE}"
fi


# Validation-only mode should return failure if incompatible tables were found,
# but only after checking the full range.
if [[ "${AUTO_ADJUST}" != true ]] && (( N_INCOMPATIBLE > 0 )); then
    echo
    echo "One or more generated mask tables are not compatible with MOM6."
    echo "Re-run with -a to generate adjusted tables."

    if [[ "${MODE}" == "layout" ]]; then
        echo
        echo "  $(basename "$0") \\"
        echo "      -g ${OCEAN_HGRID} \\"
        echo "      -t ${OCEAN_TOPOG} \\"
        echo "      -l ${LAYOUT_X} ${LAYOUT_Y} -a"
    elif [[ "${MODE}" == "range" ]]; then
        echo
        echo "  $(basename "$0") \\"
        echo "      -g ${OCEAN_HGRID} \\"
        echo "      -t ${OCEAN_TOPOG} \\"
        echo "      -r ${MIN_PROCESSORS} ${MAX_PROCESSORS} -a"
    fi

    echo
    echo "If these tables are intended for MOM5, re-run with -m mom5: the"
    echo "constraint above is specific to MOM6 and the tables are already valid."

    exit 1
fi


# Auto-adjust was requested but some candidates still could not be repaired.
if [[ "${AUTO_ADJUST}" == true ]] && (( N_UNADJUSTABLE > 0 )); then
    exit 1
fi
