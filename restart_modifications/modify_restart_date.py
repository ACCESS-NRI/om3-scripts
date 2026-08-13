# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import re
import argparse
import shutil
from datetime import date
from glob import glob
from pathlib import Path

import netCDF4 as nc

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata

"""
Change dates in OM3 restart files

Example usage:
python3 modify_restart_date.py --input_dir /path/to/archive/restart000 \
                                --output_dir /path/to/new/restart000 \
                                --new_date 2005-01-01
"""

RESTART_DATE_RE = re.compile(r"\.r\.(\d{4}-\d{2}-\d{2})-\d+")


def find_restart_files(input_dir):
    """
    Return (all_files, date_string) where all_files is every non-rpointer file in
    input_dir (collated .nc restarts, per-tile decomposed *.nc.NNNN fragments, and
    any other undated restart file such as FMS-style *.res.nc.NNNN tracer restarts
    all included, so nothing is silently dropped) and date_string is the restart
    date found among them. Errors out if no dated restart files, or more than one
    distinct date, are found.
    """
    all_files = sorted(
        f
        for f in glob(os.path.join(input_dir, "*"))
        if os.path.isfile(f) and not os.path.basename(f).startswith("rpointer.")
    )
    dates = set()
    for f in all_files:
        m = RESTART_DATE_RE.search(os.path.basename(f))
        if m:
            dates.add(m.group(1))

    if not dates:
        raise ValueError(f"No dated restart files found in {input_dir}")
    if len(dates) > 1:
        raise ValueError(
            f"Multiple restart dates found in {input_dir}: {sorted(dates)}"
        )
    return all_files, dates.pop()


def rewrite_cpl_restart(path, new_date, provenance):
    new_ymd = int(new_date.strftime("%Y%m%d"))
    with nc.Dataset(path, "r+") as f:
        f.variables["start_ymd"][:] = new_ymd
        f.variables["start_tod"][:] = 0
        f.variables["curr_ymd"][:] = new_ymd
        f.variables["curr_tod"][:] = 0
        f.setncatts(provenance)


def rewrite_cice_restart(path, new_date, provenance):
    with nc.Dataset(path, "r+") as f:
        f.myear = new_date.year
        f.mmonth = new_date.month
        f.mday = new_date.day
        f.msec = 0
        f.setncatts(provenance)


def main():
    parser = argparse.ArgumentParser(
        description="Modify ACCESS-OM3 restart files (renaming + internal date "
        "metadata) so they can be reused to start a new run on a different date."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing the source restart files (e.g. archive/restart000)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write the modified restart files to",
    )
    parser.add_argument(
        "--new_date",
        required=True,
        type=date.fromisoformat,
        help="New restart date, e.g. 2005-01-01",
    )
    args = parser.parse_args()

    restart_files, old_str = find_restart_files(args.input_dir)
    new_str = args.new_date.isoformat()

    rpointer_files = sorted(glob(os.path.join(args.input_dir, "rpointer.*")))
    ww3_files = [f for f in restart_files if os.path.basename(f).split(".")[1] == "ww3"]
    has_rpointer_wav = any(
        os.path.basename(f) == "rpointer.wav" for f in rpointer_files
    )
    if ww3_files or has_rpointer_wav:
        raise RuntimeError(
            f"ww3 restart file(s) found in {args.input_dir}: "
            f"{[os.path.basename(f) for f in ww3_files] or ['rpointer.wav']} - "
            "WW3 restarts are not handled by this script"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    for src in restart_files:
        fname = os.path.basename(src)
        new_fname = fname.replace(old_str, new_str)
        dst = os.path.join(args.output_dir, new_fname)

        # e.g. access-om3.cice.r.1959-01-01-00000.nc.0000 -> component "cice"
        component = fname.split(".")[1]

        shutil.copy2(src, dst)
        match component:
            case "cpl":
                rewrite_cpl_restart(
                    dst,
                    args.new_date,
                    get_provenance_metadata(input_files=[src], output_filename=dst),
                )
            case "cice":
                rewrite_cice_restart(
                    dst,
                    args.new_date,
                    get_provenance_metadata(input_files=[src], output_filename=dst),
                )

    for src in rpointer_files:
        fname = os.path.basename(src)
        dst = os.path.join(args.output_dir, fname)
        with open(src) as f:
            content = f.read()
        with open(dst, "w") as f:
            f.write(content.replace(old_str, new_str))

    print(f"\nModified restarts for {old_str} -> {new_str} in {args.output_dir}.")


if __name__ == "__main__":
    sys.exit(main())
