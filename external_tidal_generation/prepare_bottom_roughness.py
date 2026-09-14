#!/usr/bin/env python3
# Copyright 2026 ACCESS-NRI and contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Prepare the shared WOA / SYNBATH bottom roughness intermediate.
#
# The intermediate bottom roughness depends on the WOA temperature and salinity, and SYNBATH bathymetry.
# It is independent of the model grid.
#
# This script checks the provenance stored in an existing intermediate file and reuses it when the inputs
# are unchanged. If the intermediate is missing or out of date, it runs `generate_bottom_roughness_intermediate_woa.py`
# with MPI and only publishes the new file afer generation and provenance checks.
#
# This script must be run inside a PBS job. It uses PBS_NCPUS as the MPI rank count and it does not submit a job itself.
#
# After the shared intermediate has been prepared, run `generate_bottom_roughness_regrid.py` separately for each target MOM6 grid.
#
# After review, publish the intermediate bottom roughness file and its associated README with provenance info through model-config-inputs.
# =========================================================================================
import argparse
import os
from pathlib import Path
import subprocess
import sys

from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_common import get_provenance_input_files

GENERATOR = (
    Path(__file__).resolve().parent / "generate_bottom_roughness_intermediate_woa.py"
)


def provenance(inputs):
    """
    Recreate the inputfile attribute written by get_provenance_metadata.
    """
    return get_provenance_input_files([str(path) for path in inputs])


def is_current(output, expected):
    if not output.is_file():
        return False
    with Dataset(output) as dataset:
        return dataset.getncattr("inputFile") == expected


def generate(output, inputs, expected):
    """
    Run the MPI calculation inside PBS, publishing only a complete result.
    """
    ncpus = os.environ.get("PBS_NCPUS")
    if ncpus is None:
        raise RuntimeError(
            "PBS_NCPUS is not set; this script must be run inside a PBS job."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "mpirun",
            "-n",
            ncpus,
            sys.executable,
            str(GENERATOR),
            "--woa_temp_file",
            str(inputs[0]),
            "--woa_salt_file",
            str(inputs[1]),
            "--synbath_file",
            str(inputs[2]),
            "--woa_intermediate_file",
            str(output),
        ],
        check=True,
    )

    with Dataset(output) as dataset:
        generated = dataset.getncattr("inputFile")
    if generated != expected or provenance(inputs) != expected:
        raise RuntimeError(
            "An input changed during generation. The generated intermediate "
            f"must not be published:\n  {output}"
        )

    print(f"Generated {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the shared WOA / SYNBATH bottom roughness intermediate file."
    )
    parser.add_argument("--woa-temp-file", type=Path, required=True)
    parser.add_argument("--woa-salt-file", type=Path, required=True)
    parser.add_argument("--synbath-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Generate a new intermediate at this staging path.",
    )
    parser.add_argument(
        "--existing-intermediate",
        type=Path,
        help="Check a previously published (read-only) intermediate and exit 0 if it is current.",
    )

    args = parser.parse_args()

    if args.output is None and args.existing_intermediate is None:
        parser.error("Must specify either --output or --existing-intermediate")
    if args.output is not None and args.existing_intermediate is not None:
        parser.error("Cannot specify both --output and --existing-intermediate")

    inputs = [
        args.woa_temp_file.expanduser().resolve(),
        args.woa_salt_file.expanduser().resolve(),
        args.synbath_file.expanduser().resolve(),
    ]

    expected = provenance(inputs)

    if args.existing_intermediate is not None:
        existing = args.existing_intermediate.expanduser().resolve()

        if is_current(existing, expected):
            print(f"{existing} is current")
            return 0

        print(
            "The published (read-only) bottom roughness intermediate is missing or out of date: "
            f"{existing}",
            file=sys.stderr,
        )
        return 1

    output = args.output.expanduser().resolve()

    if output.exists():
        print(
            f"Stop overwriting an existing bottom roughness intermediate: {output}",
            file=sys.stderr,
        )
        return 2

    generate(output, inputs, expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
