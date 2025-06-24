source $(dirname "$0")/archive_scripts/concat_ice_daily.sh
source $(dirname "$0")/archive_scripts/standardise_mom6_filenames.sh

qsub -lstorage=gdata/xp65+${PBS_NCI_STORAGE} -v SCRIPTS_DIR=$(dirname "$0"),PROJECT $(dirname "$0")/archive_scripts/build_intake_ds.sh