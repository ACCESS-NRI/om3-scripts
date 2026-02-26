#!/usr/bin/env bash
set -euo pipefail


# Help
Help() {
    echo
    echo "Submit bottom roughness generation PBS job (Gadi)."
    echo
    echo "./submit_bottom_roughness.sh -s PATH_TO_SAVE -r RESOLUTION -p true|false -g HGRID -t TOPOG -j PBS_SCRIPT"
    echo
    echo "  -h   Show this help message"
    echo
    echo "Required arguments:"
    echo "  -s   Path to save intermediate and final outputs"
    echo "  -r   Resolution to generate (eg, '25km' or any user-defined name)"
    echo "  -p   Periodic mode: true (global) or false (regional)"
    echo "       (Sets BOTH PERIODIC_REGRID and PERIODIC_LON_LAPLACE)"
    echo "  -g   Path to ocean_hgrid.nc"
    echo "  -t   Path to ocean_topog.nc"
    echo "  -j   Path to PBS script"
    echo
    echo "Examples:"
    echo "  # Global (periodic) mode"
    echo "  ./submit_bottom_roughness.sh -s /path/to/save -r 25km -p true -g /path/to/hgrid.nc -t /path/to/topog.nc -j /path/to/pbs_bottom_roughness.pbs"
    echo
    exit 0
}

# default settings
PATH_TO_SAVE=""
RESOLUTION=""
PERIODIC=""
HGRID_PATH=""
TOPOG_PATH=""
PBS_SCRIPT=""

while getopts ":hs:r:p:g:t:j:" option; do
    case "${option}" in
        h)
            Help
            ;;
        s)
            PATH_TO_SAVE="${OPTARG}"
            ;;
        r)
            RESOLUTION="${OPTARG}"
            ;;
        p)
            PERIODIC="${OPTARG}"
            ;;
        g)
            HGRID_PATH="${OPTARG}"
            ;;
        t)
            TOPOG_PATH="${OPTARG}"
            ;;
        j)
            PBS_SCRIPT="${OPTARG}"
            ;;
        *)
            echo "Error: Invalid option -${OPTARG}"
            Help
            ;;
    esac
done


echo "Submitting PBS job:"
echo "  save dir: $PATH_TO_SAVE"
echo "  resolution: $RESOLUTION"
echo "  periodic: $PERIODIC"
echo "  hgrid path: $HGRID_PATH"
echo "  topog path: $TOPOG_PATH"
echo "  PBS script: $PBS_SCRIPT"

qsub \
-o "${PATH_TO_SAVE}/bottom_roughness_${RESOLUTION}.out" \
-e "${PATH_TO_SAVE}/bottom_roughness_${RESOLUTION}.error" \
-v \
PATH_TO_SAVE="${PATH_TO_SAVE}",\
RESOLUTION="${RESOLUTION}",\
PERIODIC_REGRID="${PERIODIC}",\
PERIODIC_LON_LAPLACE="${PERIODIC}",\
HGRID_PATH="${HGRID_PATH}",\
TOPOG_PATH="${TOPOG_PATH}" \
"${PBS_SCRIPT}"
