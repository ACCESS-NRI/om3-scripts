from os import chdir

import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from payu_config.postscript.build_intake_ds import description


def test_no_metadata_file(tmp_path):
    chdir(tmp_path)
    desc = description()
    assert desc.startswith("intake-esm datastore for experiment in folder")


def test_empty_metadata_file(tmp_path):
    chdir(tmp_path)
    open("metadata,yaml", "a").close()
    desc = description()
    assert desc.startswith("intake-esm datastore for experiment in folder")


def test_metadata_file(tmp_path):
    chdir(tmp_path)
    f = open(f"{tmp_path}/metadata.yaml", "w")
    print("experiment_uuid: ccedea3c-b42a-4d98-82a1-6a3255549fc6", file=f)
    f.close()
    desc = description()
    assert desc.startswith("intake-esm datastore for experiment ccedea3c")
