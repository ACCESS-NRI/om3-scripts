#!/usr/bin/env bash
set -e  # exit on error

source /etc/profile.d/modules.sh
module use /g/data/vk83/prerelease/modules && module load payu/dev

DEST="diagnostic_profiles/source_yaml_files/make_diag_table"

if [ ! -d "$DEST" ]; then
    git clone https://github.com/COSIMA/make_diag_table "$DEST"
    mv "$DEST/make_diag_table.py" "$DEST/.."
fi

cd "$DEST/.."
echo "-- Cloned make_diag_table repo to $DEST"

cp -f diag_table_standard_source.yaml diag_table_source.yaml
echo "-- Copied diag_table_standard_source.yaml to diag_table_source.yaml"

python3 make_diag_table.py

cp -f diag_table ../diag_table_standard

echo "-- diag_table is ready for use!"
