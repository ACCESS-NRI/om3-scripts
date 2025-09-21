#!/usr/bin/env bash
set -e  # exit on error

source /etc/profile.d/modules.sh
module use /g/data/vk83/prerelease/modules && module load payu/dev

# detect model
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    exit 1
fi
MODEL=$(grep '^model:' config.yaml | awk '{print $2}')
echo "-- Detected model: $MODEL"

case $MODEL in
    access-om3)
        DEST="diagnostic_profiles/source_yaml_files/make_diag_table"
        ;;
    access-om2)
        DEST="tools/make_diag_table"
        ;;
    access | access-esm1.6)
        DEST="ocean/diagnostic_profiles/source_yaml_files/make_diag_table"
        ;;
    *)
        echo "Error: Unknown model '$MODEL' (expected access-om3 | access-om2 | access-esm1.6 | access)" >&2
        exit 1
        ;;
esac

BASE=$(dirname $DEST)

mkdir -p $BASE

rm -rf $DEST
git clone https://github.com/COSIMA/make_diag_table $DEST
mv $DEST/make_diag_table.py $BASE/

cd $BASE
echo "-- Cloned make_diag_table repo to $DEST"

if [ $MODEL = "access-om2" ]; then
  python3 make_diag_table.py
else
  cp -f diag_table_standard_source.yaml diag_table_source.yaml
  echo "-- Copied diag_table_standard_source.yaml to diag_table_source.yaml"
  python3 make_diag_table.py
  cp -f diag_table ../diag_table_standard
fi

rm -rf "$DEST" "$BASE/make_diag_table.py" || true

echo "-- diag_table is ready for use! (model: $MODEL)"
