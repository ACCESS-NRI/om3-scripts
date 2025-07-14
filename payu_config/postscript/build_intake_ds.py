#!python3
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
# modules:
# use:
#     - /g/data/xp65/public/modules
# load:
#     - conda/analysis3-25.05

from access_nri_intake.source import builders
import os
import sys
from pathlib import Path
from warnings import warn
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata, md5sum

METADATA_FILENAME = "metadata.yaml"
UUID_FIELD = "experiment_uuid"
ARCHIVE_PATH = "archive"


def description():

    # Get experiment uuid
    # follows https://github.com/payu-org/payu/blob/ef55e93fe23fcde19024479c0dc4112dcdf6603f/payu/metadata.py#L90
    metadata_filename = Path(METADATA_FILENAME)
    if metadata_filename.exists():
        metadata = CommentedMap()
        metadata = YAML().load(metadata_filename)
        uuid = metadata.get(UUID_FIELD, None)
    else:
        warn(f"{METADATA_FILENAME} not found in config folder")
        uuid = False

    # Check git status of this .py file
    this_file = os.path.normpath(__file__)

    runcmd = f"python3 {os.path.basename(this_file)}"

    # Get string "Created using $file: $command"
    provenance = get_provenance_metadata(this_file, runcmd)

    if uuid:
        description = f"intake-esm datastore for experiment {uuid}, in folder {os.getcwd()}. {provenance}. (md5 hash: {md5sum(this_file)})"
    else:
        description = f"intake-esm datastore for experiment in folder {os.getcwd()}. {provenance}. (md5 hash: {md5sum(this_file)})"

    return description


if __name__ == "__main__":

    builder = builders.AccessOm3Builder(path=ARCHIVE_PATH)

    print("LOG: Building intake-esm datastore")

    builder.build()

    # Log invalid assets
    builder.invalid_assets

    builder.save(
        name="experiment_datastore",
        description=description(),
        directory=ARCHIVE_PATH,
    )
