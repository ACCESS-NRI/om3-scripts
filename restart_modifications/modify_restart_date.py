# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import re
import argparse
import shutil
import subprocess
from datetime import date
from glob import glob
from pathlib import Path
from warnings import warn

import netCDF4 as nc

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata

"""
Change dates in OM3 restart files.

Run from inside a payu configuration directory (i.e. the directory
containing config.yaml). Reads the existing restart set to use from
config.yaml's `restart:` field, writes the rebranded restart set to an
`initial_restart` subdirectory of the configuration directory, points
`restart:` at it, and commits the config.yaml change (with provenance
details in the commit message) to the configuration directory's git repo.
The original restart set (config.yaml's old `restart:` value) is never
modified.

Example usage:
cd /path/to/your/access-om3-configs
python3 modify_restart_date.py --new_date 2005-01-01
"""

RESTART_DATE_RE = re.compile(r"\.r\.(\d{4}-\d{2}-\d{2})-\d+")
# Matches the top-level `restart:` key, with its value either on the same
# line (`restart: /path`) or folded onto the next, indented line (`restart:
# \n  /path`) - both are valid YAML and payu clone itself writes the latter.
RESTART_KEY_RE = re.compile(r"^restart:[ \t]*(.*)$")


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


def read_config_restart_path(config_path):
    """
    Return the value of the top-level `restart:` field in config.yaml.
    """
    with open(config_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        m = RESTART_KEY_RE.match(line)
        if m:
            value = m.group(1).strip()
            if value:
                return value
            if i + 1 < len(lines) and lines[i + 1].strip() and lines[i + 1][0] in " \t":
                return lines[i + 1].strip()
            raise ValueError(f"Could not parse 'restart:' field value in {config_path}")
    raise ValueError(f"No top-level 'restart:' field found in {config_path}")


def write_config_restart_path(config_path, new_restart_path):
    """
    Replace the value of the top-level `restart:` field in config.yaml with a
    single-line `restart: <new_restart_path>`, leaving the rest of the file
    untouched (collapsing the folded two-line form payu clone writes, if
    present).
    """
    with open(config_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        m = RESTART_KEY_RE.match(line)
        if m:
            end = i + 1
            if (
                not m.group(1).strip()
                and end < len(lines)
                and lines[end].strip()
                and lines[end][0] in " \t"
            ):
                end += 1  # also remove the folded continuation line
            lines[i:end] = [f"restart: {new_restart_path}\n"]
            break
    else:
        raise ValueError(f"No top-level 'restart:' field found in {config_path}")

    with open(config_path, "w") as f:
        f.writelines(lines)


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
        "--config_dir",
        default=".",
        help="Path to the payu configuration directory (default: current directory)",
    )
    parser.add_argument(
        "--new_date",
        default=date(1958, 1, 1),
        type=date.fromisoformat,
        help="New restart date, e.g. 2005-01-01 (default: 1958-01-01)",
    )
    args = parser.parse_args()

    config_dir = os.path.abspath(args.config_dir)
    config_path = os.path.join(config_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise ValueError(f"No config.yaml found in {config_dir}")

    input_dir = read_config_restart_path(config_path)
    if not os.path.isabs(input_dir):
        # Relative to config_dir, not the current working directory
        input_dir = os.path.join(config_dir, input_dir)
    output_dir = os.path.join(config_dir, "initial_restart")
    if os.path.exists(output_dir):
        raise ValueError(f"{output_dir} already exists - remove it before re-running")

    restart_files, old_str = find_restart_files(input_dir)
    new_str = args.new_date.isoformat()

    if old_str == new_str:
        raise ValueError(
            "Current restart date and new_date are the same, nothing to do"
        )

    rpointer_files = sorted(glob(os.path.join(input_dir, "rpointer.*")))
    ww3_files = [f for f in restart_files if os.path.basename(f).split(".")[1] == "ww3"]
    has_rpointer_wav = any(
        os.path.basename(f) == "rpointer.wav" for f in rpointer_files
    )
    if ww3_files or has_rpointer_wav:
        raise RuntimeError(
            f"ww3 restart file(s) found in {input_dir}: "
            f"{[os.path.basename(f) for f in ww3_files] or ['rpointer.wav']} - "
            "WW3 restarts are not handled by this script"
        )

    os.makedirs(output_dir)

    output_files = []
    key_files = []  # cpl/cice source files actually rewritten, for commit provenance
    for src in restart_files:
        fname = os.path.basename(src)
        new_fname = fname.replace(old_str, new_str)
        dst = os.path.join(output_dir, new_fname)
        output_files.append(dst)

        # e.g. access-om3.cice.r.1959-01-01-00000.nc.0000 -> component "cice"
        component = fname.split(".")[1]

        shutil.copy2(src, dst)
        match component:
            case "cpl":
                rewrite_cpl_restart(
                    dst,
                    args.new_date,
                    get_provenance_metadata(input_files=[src], write_readme_file=False),
                )
                key_files.append(src)
            case "cice":
                rewrite_cice_restart(
                    dst,
                    args.new_date,
                    get_provenance_metadata(input_files=[src], write_readme_file=False),
                )
                key_files.append(src)

    for src in rpointer_files:
        fname = os.path.basename(src)
        dst = os.path.join(output_dir, fname)
        output_files.append(dst)
        with open(src) as f:
            content = f.read()
        with open(dst, "w") as f:
            f.write(content.replace(old_str, new_str))

    # Write a single README describing the whole restart folder, rather than one
    # per restart file.
    get_provenance_metadata(
        input_files=restart_files,
        output_dir=output_dir,
        output_filename=output_files,
    )

    write_config_restart_path(config_path, output_dir)

    provenance = get_provenance_metadata(input_files=key_files, write_readme_file=False)
    commit_message = "\n".join(
        [
            f"Rebrand restart from {old_str} to {new_str}",
            "",
            f"Previous restart: {input_dir}",
            f"New restart: {output_dir}",
            "",
            provenance["history"],
            provenance.get("inputFile", ""),
        ]
    ).strip()

    try:
        subprocess.run(["git", "-C", config_dir, "add", "config.yaml"], check=True)
        subprocess.run(
            ["git", "-C", config_dir, "commit", "-m", commit_message], check=True
        )
    except subprocess.CalledProcessError:
        warn(f"Could not commit in {config_dir}")

    print(
        f"\nModified restarts for {old_str} -> {new_str} in {output_dir}, "
        f"and updated config.yaml"
    )


if __name__ == "__main__":
    sys.exit(main())
