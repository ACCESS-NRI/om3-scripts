#!/usr/bin/env python3
# Copyright 2024 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.
#
# Standardise file naming for MOM output files in access-om by removing the underscore
# before date/time suffixes and replacing subsequent underscores with hyphens, e.g.,
# replacing '_YYYY_MM' with 'YYYY-MM'.
# This was written assuming it would be used as a payu "userscript" at the "archive" stage,
# but alternatively a path to an "archive" directory can be provided.
# For more details, see https://github.com/COSIMA/om3-scripts/issues/32

import argparse
import glob
import os
import re
import sys


def standardised_filename(path):
    """Return the standardised filename for a MOM output file, or None if no rename is needed.

    Files ending in *_<digits>[_<digits>...].nc[...] are renamed so that the first
    underscore before the digit groups is removed and subsequent ones are replaced with '-'.
    For example:
        file._2023.nc       -> file.2023.nc
        file._2023_01.nc    -> file.2023-01.nc
        file._2023_01_15.nc -> file.2023-01-15.nc
    """
    dirpath = os.path.dirname(path)
    basename = os.path.basename(path)

    # Split the .nc extension
    ext_match = re.match(r"^(.*?)(\.nc.*)$", basename)
    if not ext_match:
        return None
    stem, ext = ext_match.groups()

    # Match the trailing block of underscore-separated digit groups
    suffix_match = re.match(r"^(.*?)(_\d+(?:_\d+)*)$", stem)
    if not suffix_match:
        return None
    base, suffix = suffix_match.groups()

    # Remove the leading underscore and replace any remaining ones with '-'
    new_suffix = suffix[1:].replace("_", "-")

    new_basename = base + new_suffix + ext
    if new_basename == basename:
        return None

    return os.path.join(dirpath, new_basename)


def main():
    parser = argparse.ArgumentParser(
        description="Standardise file naming for MOM output files."
    )
    parser.add_argument(
        "-d",
        metavar="DIRECTORY",
        dest="out_dir",
        help="Process files in the specified experiment 'DIRECTORY'.",
    )
    args = parser.parse_args()

    if args.out_dir:
        if not os.path.isdir(args.out_dir):
            print(f"{args.out_dir} does not exist")
            sys.exit(1)
        out_dirs = [args.out_dir]
    else:
        out_dirs = sorted(glob.glob("archive/output*[0-9]"), reverse=True)

    # Support ACCESS-OM3 and ACCESS-OM2 output files
    file_patterns = [
        "access-om3.mom6.*.nc*",
        "ocean/access-om2.mom5.*.nc*",
    ]

    for dir_path in out_dirs:
        for pattern in file_patterns:
            for current_file in glob.glob(f"{dir_path}/{pattern}"):
                if not os.path.isfile(current_file):
                    continue
                new_file = standardised_filename(current_file)
                if new_file is None:
                    continue
                if os.path.exists(new_file):
                    print(f"Skipping {current_file}: {new_file} already exists")
                    continue
                os.rename(current_file, new_file)


if __name__ == "__main__":
    main()
