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
# Requires FRE-NCtools (make_solo_mosaic, check_mask) on $PATH; see MODULE_COMMANDS below.
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
import os
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

# mom6 builds a second, coarsened domain for downsampled diagnostics
# (create_MOM_domain -> clone_MD_to_d2D(..., coarsen=2)). A layout that does not
# fit inside the coarsened domain produces zero-sized FMS domains and fails with
# "MPP_DEFINE_DOMAINS(mpp_compute_extent): domain extents must be positive definite".
COARSEN_FACTOR = 2

# FRE-NCtools executables this script drives.
REQUIRED_TOOLS = ("make_solo_mosaic", "check_mask")

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


@dataclass(frozen=True)
class Compatibility:
    """The outcome of the mom6 compatibility test for a given active PE count."""

    active_pes: int
    unmasked_x: int
    unmasked_y: int
    coarse_nx: int
    coarse_ny: int

    @property
    def compatible(self) -> bool:
        return self.unmasked_x <= self.coarse_nx and self.unmasked_y <= self.coarse_ny


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
        help="Periodicity in the Y direction (default: unset, i.e. aperiodic in Y).",
    )

    args = parser.parse_args()

    if args.model == "mom5" and args.auto_adjust:
        parser.error("Auto-adjust '-a' option is not applicable for mom5.")

    for name in ("hgrid", "topog"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"--{name} is not an existing file: {path}")
        setattr(args, name, path.resolve())

    if args.layout and any(value <= 0 for value in args.layout):
        parser.error("--layout X Y requires positive integers.")

    if args.processor_range:
        min_proc, max_proc = args.processor_range
        if min_proc <= 0 or max_proc <= 0:
            parser.error("--processor-range MIN MAX requires positive integers.")
        if min_proc > max_proc:
            parser.error("--processor-range MIN must not exceed MAX.")

    return args


def check_tools():
    """Fail early, and helpfully, when FRE-NCtools is not on $PATH."""
    missing = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    if missing:
        raise MasktableError(
            "Could not find the FRE-NCtools executable(s): "
            + ", ".join(missing)
            + "\nLoad them with:\n"
            + "\n".join(f"  {command}" for command in MODULE_COMMANDS)
        )


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


def grid_size(topog: Path) -> tuple:
    with nc.Dataset(topog) as ds:
        missing = [dim for dim in ("nx", "ny") if dim not in ds.dimensions]
        if missing:
            raise MasktableError(
                f"{topog} has no {' or '.join(missing)} dimension; is it a topog file?"
            )
        return len(ds.dimensions["nx"]), len(ds.dimensions["ny"])


def make_ocean_mosaic(hgrid: Path, periodx: float, periody: float | None) -> Path:
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
    return Path("ocean_mosaic.nc")


def check_mask(args, mosaic: Path) -> list:
    command = [
        "check_mask",
        "--grid_file",
        mosaic.name,
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

    # Sorted so that the reporting below is reproducible from run to run.
    reported = sorted(
        {
            (int(n_mask), int(layout_x), int(layout_y))
            for n_mask, layout_x, layout_y in MASKTABLE_PATTERN.findall(output)
        },
        key=lambda entry: (entry[1] * entry[2], entry[1], entry[0]),
    )

    if not reported:
        raise MasktableError(
            "check_mask did not generate any mask tables. No land-only domains "
            "exist for the requested layout or processor range."
        )

    masktables = [Path(f"mask_table.{n}.{x}x{y}") for n, x, y in reported]

    missing = [str(path) for path in masktables if not path.is_file()]
    if missing:
        raise MasktableError(
            "check_mask reported mask tables that do not exist: " + ", ".join(missing)
        )

    return masktables


def read_masktable(filepath: Path) -> Masktable:
    filepath = Path(filepath)
    lines = filepath.read_text().splitlines()

    if len(lines) < 2:
        raise MasktableError(f"{filepath} has fewer than two lines; not a mask table.")

    try:
        n_mask = int(lines[0].strip())
        layout_x, layout_y = (int(value.strip()) for value in lines[1].split(","))
    except ValueError as exc:
        raise MasktableError(
            f"Could not parse the header of {filepath}: {exc}"
        ) from exc

    return Masktable(n_mask, layout_x, layout_y)


def read_mask_entries(filepath: Path) -> list:
    """
    Return the masked-domain entries, skipping the two header lines and any
    provenance comments a previous run added.
    """
    lines = Path(filepath).read_text().splitlines()
    return [line for line in lines[2:] if line.strip() and not line.startswith("#")]


def mom_define_layout(nx: int, ny: int, npes: int) -> tuple[int, int]:
    """
    Reproduce mom6's MOM_define_layout() for a positive PE count.
    https://github.com/ACCESS-NRI/MOM6/blob/6432010e3ab29df43994adabb413b69fe718d94c/src/framework/MOM_domains.F90#L466-L486
    """
    if npes < 1:
        raise ValueError(f"npes must be positive, got {npes}.")
    idiv = max(math.floor(math.sqrt(npes * nx / ny) + 0.5), 1)
    while npes % idiv:
        idiv -= 1
    return idiv, npes // idiv


def mom6_compatibility(nx: int, ny: int, active_pes: int):
    """
    Test whether the layout mom6 derives from active_pes fits inside the
    factor-2 coarsened global domain. Returns None if active_pes is not a
    usable PE count.

    This assumes mom6 runs with LAYOUT = 0, 0 so that it calls
    MOM_define_layout(n_global, PEs_used, layout) itself. A configuration that
    sets LAYOUT explicitly to the mask table's layout uses that layout instead.
    """
    if active_pes <= 0:
        return None
    unmasked_x, unmasked_y = mom_define_layout(nx, ny, active_pes)
    return Compatibility(
        active_pes,
        unmasked_x,
        unmasked_y,
        nx // COARSEN_FACTOR,
        ny // COARSEN_FACTOR,
    )


def report_compatibility(masktable_path: Path, masktable: Masktable, check) -> bool:
    print(f"\n-- mom6 compatibility check: {masktable_path}")
    print(f"   Logical layout: {masktable.layout_x} x {masktable.layout_y}")
    print(f"   Total logical domains: {masktable.total_domain}")
    print(f"   Masked land domains: {masktable.n_mask}")
    print(f"   Active PEs: {masktable.active_pes}")

    if check is None:
        print("   mom6 compatibility: FAILED (no active PEs remain)")
        return False

    print(f"   mom6 unmasked layout: {check.unmasked_x} x {check.unmasked_y}")
    print(f"   Coarsened global domain: {check.coarse_nx} x {check.coarse_ny}")
    print(f"   mom6 compatibility: {'OK' if check.compatible else 'FAILED'}")
    return check.compatible


def is_compatible_masktable(masktable_path: Path, nx: int, ny: int) -> bool:
    masktable = read_masktable(masktable_path)
    check = mom6_compatibility(nx, ny, masktable.active_pes)
    return report_compatibility(masktable_path, masktable, check)


def find_masktable_mask_count(nx: int, ny: int, masktable: Masktable) -> int | None:
    """
    Largest number of masked domains, at most masktable.n_mask, whose active PE
    count gives a mom6-compatible layout. Masking fewer domains keeps more
    land-only PEs, so the search runs downwards to minimise that overhead.

    Stops at one masked domain: a mask table that masks nothing is pointless,
    and check_mask never writes one.
    """
    for n_mask in range(masktable.n_mask, 0, -1):
        check = mom6_compatibility(nx, ny, masktable.total_domain - n_mask)
        if check is not None and check.compatible:
            return n_mask
    return None


def write_adjusted_masktable(
    masktable_path: Path, new_n_mask: int, protected: set
) -> Path:
    masktable = read_masktable(masktable_path)
    entries = read_mask_entries(masktable_path)

    if len(entries) < new_n_mask:
        raise MasktableError(
            f"{masktable_path} lists {len(entries)} masked domains, "
            f"but {new_n_mask} are needed for the adjusted table."
        )

    output = Path(f"mask_table.{new_n_mask}.{masktable.layout_x}x{masktable.layout_y}")

    # A different PE count in the same run can produce this same filename.
    if output in protected:
        raise MasktableError(
            f"The adjusted mask table {output} would overwrite a mask table "
            f"generated by check_mask in this run."
        )

    # Any subset of the land-only domains is a valid mask, so keeping the first
    # new_n_mask entries is sufficient.
    output.write_text(
        "\n".join(
            [str(new_n_mask), f"{masktable.layout_x},{masktable.layout_y}"]
            + entries[:new_n_mask]
        )
        + "\n"
    )
    return output


def add_provenance(
    masktable_path: Path,
    common_provenance: list[str],
    message: str,
):
    """
    Insert provenance as comments after the two header lines. FMS's
    parse_mask_table skips records whose first character is '#', so these are
    ignored by the model (fms/fms_io.F90, parse_mask_table_2d).
    """
    masktable_path = Path(masktable_path)
    lines = masktable_path.read_text().splitlines()
    comments = [f"# {line}".rstrip() for line in common_provenance + [message]]
    masktable_path.write_text("\n".join(lines[:2] + comments + lines[2:]) + "\n")


def module_provenance() -> list[str]:
    """
    Record the modules actually loaded, rather than asserting the documented
    ones were used.
    """
    loaded = [m for m in os.environ.get("LOADEDMODULES", "").split(":") if m]
    if loaded:
        return ["Loaded environment modules:"] + [f"  {m}" for m in loaded]
    return ["Environment modules (documented; none detected at run time):"] + [
        f"  {command}" for command in MODULE_COMMANDS
    ]


def provenance(args, nx: int, ny: int):
    command = shlex.join([sys.executable, *sys.argv])
    history = get_provenance_metadata(
        runcmd=command,
        write_readme_file=False,
    )["history"]
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return [
        history,
        "",
        *module_provenance(),
        "",
        f"Date: {timestamp}",
        f"hgrid: {args.hgrid}",
        f"topog: {args.topog}",
        f"Grid size: {nx} x {ny}",
        f"periodx: {args.periodx}",
        f"periody: {args.periody if args.periody is not None else 'unset (aperiodic)'}",
        f"Target model: {args.model}",
    ]


def rerun_hint(args) -> str:
    selection = (
        f"-l {args.layout[0]} {args.layout[1]}"
        if args.layout
        else f"-r {args.processor_range[0]} {args.processor_range[1]}"
    )
    return (
        f"  {Path(sys.argv[0]).name} \\\n"
        f"      -g {args.hgrid} \\\n"
        f"      -t {args.topog} \\\n"
        f"      {selection} -m {args.model} -a"
    )


def process_mom6_masktables(
    masktables: list,
    nx: int,
    ny: int,
    args,
    common_provenance: list[str],
):
    auto_adjust = args.auto_adjust
    protected = set(masktables)

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
        print(
            "      NOTE: the overhead is the extra allocation from retaining\n"
            "      land-only PEs, not an estimate of wall-clock performance loss."
        )

        if not auto_adjust:
            add_provenance(
                path,
                common_provenance,
                "mom6 compatibility check: FAILED (auto-adjust not enabled)",
            )
            continue

        adjusted_path = write_adjusted_masktable(path, compatible_n_mask, protected)
        print(
            f"      *Generated adjusted mom6-compatible mask masktable: {adjusted_path}"
        )

        if is_compatible_masktable(adjusted_path, nx, ny):
            add_provenance(
                adjusted_path,
                common_provenance,
                "mom6 compatibility check: OK (auto-adjusted)",
            )
            # Stamp the superseded check_mask output so it cannot be picked up
            # by mistake; it is kept as a record of what check_mask produced.
            add_provenance(
                path,
                common_provenance,
                f"mom6 compatibility check: FAILED (superseded by {adjusted_path})",
            )
            protected.add(adjusted_path)
            adjusted_count += 1
        else:
            adjusted_path.unlink()
            add_provenance(
                path,
                common_provenance,
                "mom6 compatibility check: FAILED (adjusted table failed validation)",
            )
            unadjusted_count += 1

    print("\n-- Mask-masktable generation complete (target model: mom6)")
    print(f"Generated by check_mask: {len(masktables)}")
    print(f"Already mom6-compatible: {compatible_count}")
    print(f"Initially incompatible: {incompatible_count}")
    if auto_adjust:
        print(f"Automatically adjusted: {adjusted_count}")
    if unadjusted_count:
        print(f"Could not adjust: {unadjusted_count}")

    if not auto_adjust and incompatible_count:
        raise MasktableError(
            "Some mask tables are incompatible with mom6.\n"
            "Re-run with the auto-adjust option (-a) to generate adjusted tables:\n"
            + rerun_hint(args)
        )

    if unadjusted_count:
        raise MasktableError(
            f"{unadjusted_count} mask table(s) could not be made compatible with mom6. "
            "Choose a different layout or processor range."
        )


def main():
    args = parse_args()
    check_tools()

    nx, ny = grid_size(args.topog)
    print(f"-- mom grid size: {nx} x {ny}")

    # check_mask reads the topog directly, so only the hgrid needs staging:
    # make_solo_mosaic resolves --tile_file relative to --dir.
    local_hgrid = copy_input(args.hgrid)

    mosaic = make_ocean_mosaic(local_hgrid, args.periodx, args.periody)
    masktables = check_mask(args, mosaic)
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

    process_mom6_masktables(masktables, nx, ny, args, common_provenance)


if __name__ == "__main__":
    try:
        main()
    except MasktableError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
