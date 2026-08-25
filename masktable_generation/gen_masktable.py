#!/usr/bin/env python3
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
#
# =========================================================================================
# This script generates masktables for mom5/mom6 with FRE-NCtools.
#
# Usage:
# 1. Exact layout specification (mom5):
#   python3 gen_masktable.py -g path/to/hgrid.nc -t path/to/topog.nc -l X Y -m mom5
#
# 2. Exact layout specification (mom6):
#   python3 gen_masktable.py -g path/to/hgrid.nc -t path/to/topog.nc -l X Y -m mom6 -a
#
# 3. Processor range specification (mom5):
#   python3 gen_masktable.py -g path/to/hgrid.nc -t path/to/topog.nc -r MIN MAX -m mom5
#
# 4. Processor range specification (mom6):
#   python3 gen_masktable.py -g path/to/hgrid.nc -t path/to/topog.nc -r MIN MAX -m mom6 -a
#
# For more details, see
#   1. https://github.com/COSIMA/mom6-panan/wiki/Preparing-inputs-for-a-new-configuration
#   2. https://github.com/ACCESS-NRI/mom6/issues/42#issuecomment-5337278337
#
# Contact:
#   Minghang Li <minghang.li1@anu.edu.au>
#   This script was originally developed by Angus Gibson and modified and enhanced by Minghang Li.

import sys
import argparse
import re
import math
from dataclasses import dataclass
from pathlib import Path
import subprocess
import shutil
import netCDF4 as nc
import shlex
from datetime import datetime

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata

MASKTABLE_PATTERN = re.compile(r"masked=(\d+),\s*layout=(\d+),\s*(\d+)")

MODULE_COMMANDS = [
    "module use /g/data/xp65/public/modules",
    "module load conda/analysis3",
    "module use /g/data/vk83/modules",
    "module load model-tools/fre-nctools/2024.05-1",
]


class MasktableError(RuntimeError):
    """An error that should be reported to the user without a traceback."""


@dataclass(frozen=True)
class Masktable:
    n_mask: int
    layout_x: int
    layout_y: int

    @property
    def total_domain(self):
        return self.layout_x * self.layout_y

    @property
    def active_pes(self):
        return self.total_domain - self.n_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mom5/mom6 mask tables with FRE-NCtools."
    )
    parser.add_argument(
        "-g",
        "--hgrid",
        type=Path,
        required=True,
        help="Path to the hgrid file.",
    )

    parser.add_argument(
        "-t",
        "--topog",
        type=Path,
        required=True,
        help="Path to the topog file.",
    )

    # https://docs.python.org/3/library/argparse.html#:~:text=ArgumentParser.add_mutually_exclusive_group(required%3DFalse)%C2%B6
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-l",
        "--layout",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="Layout of the domain to use for the mask masktable generation.",
    )

    group.add_argument(
        "-r",
        "--processor-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Range of processors to use for the mask masktable generation.",
    )

    parser.add_argument(
        "-m",
        "--model",
        choices=["mom5", "mom6"],
        required=True,
        help="Model type (mom5 or mom6).",
    )

    parser.add_argument(
        "-a",
        "--auto-adjust",
        action="store_true",
        help="Adjust incompatible mask tables only for mom6",
    )

    parser.add_argument(
        "-x",
        "--periodx",
        type=float,
        default=360.0,
        help="Periodicity in the X direction (default: 360.0).",
    )

    parser.add_argument(
        "-y",
        "--periody",
        type=float,
        help="Periodicity in the Y direction (default: 180.0).",
    )

    args = parser.parse_args()
    if args.model == "mom5" and args.auto_adjust:
        parser.error("Auto-adjust '-a' option is not applicable for mom5.")
    return args


def copy_input(source: Path | str) -> Path:
    source = source.resolve()
    target = Path.cwd() / source.name

    if target.exists() and target.resolve() == source:
        print(f"-- Using {source} in place")
        return target

    print(f"-- Copying {source} to {target}")
    shutil.copy2(source, target)
    return target


def run(command: list, capture_output=False):
    command = [str(c) for c in command]
    print(f"-- Running: {shlex.join(command)}")

    try:
        res = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.STDOUT if capture_output else None,
        )
    except subprocess.CalledProcessError as exc:
        # With capture_output the tool's diagnostics are in exc.output, which
        # would otherwise be swallowed along with the reason for the failure.
        if exc.output:
            print(exc.output, end="")
        raise MasktableError(
            f"{command[0]} failed with exit code {exc.returncode}."
        ) from exc

    if capture_output:
        print(res.stdout, end="")
        return res.stdout

    return ""


def grid_size(topog: Path | str) -> tuple[int, int]:
    with nc.Dataset(topog) as ds:
        return len(ds.dimensions["nx"]), len(ds.dimensions["ny"])


def make_mosaics(
    hgrid: Path | str, topog: Path | str, periodx: float, periody: float | None
):
    command = [
        "make_solo_mosaic",
        "--num_tiles",
        "1",
        "--dir",
        ".",
        "--mosaic_name",
        "ocean_mosaic",
        "--tile_file",
        hgrid.name,
        "--periodx",
        periodx,
    ]
    if periody is not None:
        command.extend(["--periody", periody])
    run(command)


def check_mask(args):
    command = [
        "check_mask",
        "--grid_file",
        "ocean_mosaic.nc",
        "--ocean_topog",
        args.topog,
    ]

    if args.layout:
        layout_x, layout_y = args.layout
        print(f"Checking mask masktable for layout: {layout_x} x {layout_y}")
        command.extend(["--layout", f"{layout_x},{layout_y}"])
    elif args.processor_range:
        min_proc, max_proc = args.processor_range
        print(f"Checking mask masktable for processor range: {min_proc} to {max_proc}")
        command.extend(["--min_pe", str(min_proc), "--max_pe", str(max_proc)])

    output = run(command, capture_output=True)

    filenames = {
        Path(f"mask_table.{n_mask}.{layout_x}x{layout_y}")
        for n_mask, layout_x, layout_y in MASKTABLE_PATTERN.findall(output)
    }

    if not filenames:
        raise MasktableError("check_mask did not generate any mask tables.")

    return filenames


def read_masktable(filepath: Path | str) -> Masktable:
    lines = filepath.read_text().splitlines()
    layout_x, layout_y = (int(value.strip()) for value in lines[1].split(","))
    return Masktable(int(lines[0]), layout_x, layout_y)


def mom_define_layout(nx: int, ny: int, npes: int) -> tuple[int, int]:
    """
    Reproduce mom6's MOM_define_layout() for a positive PE count.
    https://github.com/ACCESS-NRI/MOM6/blob/6432010e3ab29df43994adabb413b69fe718d94c/src/framework/MOM_domains.F90#L466-L486
    """
    idiv = max(math.floor(math.sqrt(npes * nx / ny) + 0.5), 1)
    while npes % idiv:
        idiv -= 1
    return idiv, npes // idiv


def mom6_layout_is_compatible(nx: int, ny: int, active_pes: int) -> bool:
    if active_pes <= 0:
        return False
    unmasked_x, unmasked_y = mom_define_layout(nx, ny, active_pes)
    return unmasked_x <= nx // 2 and unmasked_y <= ny // 2


def is_compatible_masktable(masktable_path: Path | str, nx: int, ny: int) -> bool:
    masktable = read_masktable(masktable_path)
    unmasked_x, unmasked_y = mom_define_layout(nx, ny, masktable.active_pes)
    coarse_nx, coarse_ny = nx // 2, ny // 2
    compatible = unmasked_x <= coarse_nx and unmasked_y <= coarse_ny

    print(f"\n-- mom6 compatibility check: {masktable_path}")
    print(f"   Logical layout: {masktable.layout_x} x {masktable.layout_y}")
    print(f"   Total logical domains: {masktable.total_domain}")
    print(f"   Masked land domains: {masktable.n_mask}")
    print(f"   Active PEs: {masktable.active_pes}")
    print(f"   mom6 unmasked layout: {unmasked_x} x {unmasked_y}")
    print(f"   Coarsened global domain: {coarse_nx} x {coarse_ny}")
    print(f"   mom6 compatibility: {'OK' if compatible else 'FAILED'}")
    return compatible


def find_masktable_mask_count(nx: int, ny: int, masktable: Masktable) -> int | None:
    for n_mask in range(masktable.n_mask, -1, -1):
        active_pes = masktable.total_domain - n_mask
        if mom6_layout_is_compatible(nx, ny, active_pes):
            return n_mask
    return None


def write_adjusted_masktable(masktable_path: Path | str, new_n_mask: int):
    masktable = read_masktable(masktable_path)
    lines = masktable_path.read_text().splitlines()
    mask_entries = [line for line in lines[2:] if line and not line.startswith("#")]
    output = Path(f"mask_table.{new_n_mask}.{masktable.layout_x}x{masktable.layout_y}")
    output.write_text(
        "\n".join(
            [str(new_n_mask), f"{masktable.layout_x},{masktable.layout_y}"]
            + mask_entries[:new_n_mask]
        )
        + "\n"
    )
    return output


def add_provenance(
    masktable_path: Path | str,
    common_provenance: list[str],
    message: str,
):
    lines = masktable_path.read_text().splitlines()
    comments = [f"# {line}" for line in common_provenance + [message]]
    masktable_path.write_text("\n".join(lines[:2] + comments + lines[2:]) + "\n")


def provenance(args, nx: int, ny: int):
    command = shlex.join([sys.executable, *sys.argv])
    history = get_provenance_metadata(
        runcmd=command,
        write_readme_file=False,
    )["history"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        history,
        "",
        "Environment modules:",
        *MODULE_COMMANDS,
        "",
        f"Date: {timestamp}",
        f"hgrid: {args.hgrid}",
        f"topog: {args.topog}",
        f"Grid size: {nx} x {ny}",
        f"Target model: {args.model}",
    ]


def process_mom6_masktables(
    masktables: list[Path | str],
    nx: int,
    ny: int,
    auto_adjust: bool,
    common_provenance: list[str],
):
    compatible_count = 0
    incompatible_count = 0
    adjusted_count = 0
    unadjusted_count = 0

    for path in masktables:
        if is_compatible_masktable(path, nx, ny):
            add_provenance(path, common_provenance, "mom6 compatibility check: OK")
            compatible_count += 1
            continue

        incompatible_count += 1
        masktable = read_masktable(path)
        compatible_n_mask = find_masktable_mask_count(nx, ny, masktable)
        if compatible_n_mask is None:
            add_provenance(
                path,
                common_provenance,
                "mom6 compatibility check: FAILED (no compatible mask count found)",
            )
            unadjusted_count += 1
            continue

        compatible_active_pes = masktable.total_domain - compatible_n_mask
        unmasked_x, unmasked_y = mom_define_layout(nx, ny, compatible_active_pes)
        overhead = (1 - masktable.active_pes / compatible_active_pes) * 100

        print("\n   -- Nearest mom6-compatible mask configuration")
        print(f"      Original masked domains: {masktable.n_mask}")
        print(f"      Compatible masked domains: {compatible_n_mask}")
        print(f"      Original active PEs: {masktable.active_pes}")
        print(f"      Compatible active PEs: {compatible_active_pes}")
        print(
            f"      Extra retained land-only PEs: {compatible_active_pes - masktable.active_pes}"
        )
        print(f"      Compatible unmasked layout: {unmasked_x} x {unmasked_y}")
        print(f"      Approximate processor overhead: {overhead:.3f}%")

        if not auto_adjust:
            add_provenance(
                path,
                common_provenance,
                "mom6 compatibility check: FAILED (auto-adjust not enabled)",
            )
            continue

        adjusted_path = write_adjusted_masktable(path, compatible_n_mask)
        print(
            f"      *Generated adjusted mom6-compatible mask masktable: {adjusted_path}"
        )

        if is_compatible_masktable(adjusted_path, nx, ny):
            add_provenance(
                adjusted_path,
                common_provenance,
                "mom6 compatibility check: OK (auto-adjusted)",
            )
            adjusted_count += 1
        else:
            adjusted_path.unlink()
            unadjusted_count += 1

    print("\n-- Mask-masktable generation complete (target model: mom6)")
    print(f"Generated by check_mask: {len(masktables)}")
    print(f"Already mom6-compatible: {compatible_count}")
    print(f"Initially incompatible: {incompatible_count}")
    if auto_adjust:
        print(f"Automatically adjusted: {adjusted_count}")
    if unadjusted_count:
        print(f"Could not adjust: {unadjusted_count}")

    if (not auto_adjust and incompatible_count) or unadjusted_count:
        raise MasktableError(
            "Some mask tables are incompatible with mom6. "
            "Please enable the auto-adjust option (-a)."
        )


def main():
    args = parse_args()
    nx, ny = grid_size(args.topog)

    local_hgrid = copy_input(args.hgrid)
    local_topog = copy_input(args.topog)

    make_mosaics(local_hgrid, local_topog, args.periodx, args.periody)
    masktables = check_mask(args)
    common_provenance = provenance(args, nx, ny)

    if args.model == "mom5":
        for path in masktables:
            add_provenance(
                path,
                common_provenance,
                "mom6 compatibility check skipped (target model: mom5)",
            )

        print("-- Mask-masktable generation complete (target model: mom5)")
        for path in masktables:
            print(f"Generated mask masktable: {path}")
        return

    return process_mom6_masktables(
        masktables, nx, ny, args.auto_adjust, common_provenance
    )


if __name__ == "__main__":
    try:
        main()
    except MasktableError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
