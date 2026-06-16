# Copyright 2024 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.

import pytest
from os import chdir
from pathlib import Path
from subprocess import run

scripts_base = Path(__file__).parents[2]
run_cmd = [
    "python3",
    str(scripts_base / "payu_config/archive_scripts/standardise_mom_filenames.py"),
]

MOM6_DIAG_BASE = "access-om3.mom6.test"
MOM5_DIAG_BASE = "access-om2.mom5.test"


def assert_file_exists(p):
    if not Path(p).resolve().is_file():
        raise AssertionError(f"File does not exist: {p}")


def assert_file_not_exists(p):
    if Path(p).resolve().is_file():
        raise AssertionError(f"File exists and should not: {p}")


def make_files(out_dir, diag_base, suffixes, splits=0):
    """Create empty test files and return their paths.

    Creates '{diag_base}.{suffix}.nc' for each suffix. If splits > 0, creates split
    files '{diag_base}.{suffix}.nc.0001' ... '.nc.{splits:04d}' instead.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in suffixes:
        if splits:
            for i in range(1, splits + 1):
                paths.append(out_dir / f"{diag_base}.{suffix}.nc.{i:04d}")
        else:
            paths.append(out_dir / f"{diag_base}.{suffix}.nc")
    for p in paths:
        p.touch()
    return [str(p) for p in paths]


def standardised_path(orig_path, suffix):
    """Return the expected path after standardisation of the given suffix.

    The first underscore in the suffix is removed and any remaining ones replaced
    with '-', e.g. suffix '_2010_01' becomes '2010-01'.
    """
    p = Path(orig_path)
    new_suffix = suffix[1:].replace("_", "-")
    new_name = p.name.replace(f".{suffix}.", f".{new_suffix}.", 1)
    return str(p.parent / new_name)


@pytest.mark.parametrize(
    "hist_dir, diag_base, file_subdir, use_dir, suffixes, splits",
    [
        # ACCESS-OM3
        (
            "archive/output000",
            MOM6_DIAG_BASE,
            "",
            False,
            ["_2010", "_2011_01", "_2012_01_01", "_2012_01_01_000000"],
            0,
        ),
        (
            "archive/output9999",
            MOM6_DIAG_BASE,
            "",
            True,
            ["_2010", "_2011_01", "_2012_01_01", "_2012_01_01_000000"],
            3,
        ),
        # ACCESS-OM2
        (
            "archive/output000",
            MOM5_DIAG_BASE,
            "ocean",
            False,
            ["_2010", "_2011_01", "_2012_01_01", "_2012_01_01_000000"],
            0,
        ),
        (
            "archive/output9999",
            MOM5_DIAG_BASE,
            "ocean",
            True,
            ["_2010", "_2011_01", "_2012_01_01", "_2012_01_01_000000"],
            3,
        ),
    ],
)
def test_rename(hist_dir, diag_base, file_subdir, use_dir, suffixes, splits, tmp_path):
    output_dir = tmp_path / hist_dir
    file_dir = output_dir / file_subdir if file_subdir else output_dir

    original_paths = make_files(str(file_dir), diag_base, suffixes, splits)
    chdir(tmp_path)

    if use_dir:
        run(run_cmd + ["-d", str(output_dir)])
    else:
        run(run_cmd)

    for orig_path in original_paths:
        name = Path(orig_path).name
        suffix = next(s for s in suffixes if f".{s}." in name)
        assert_file_exists(standardised_path(orig_path, suffix))
        assert_file_not_exists(orig_path)


@pytest.mark.parametrize(
    "hist_dir, diag_base, file_subdir, suffixes",
    [
        ("archive/output000", MOM6_DIAG_BASE, "", ["_2010", "_2011_01", "_2012_01_01"]),
        (
            "archive/output000",
            MOM5_DIAG_BASE,
            "ocean",
            ["_2010", "_2011_01", "_2012_01_01"],
        ),
    ],
)
def test_dont_override(hist_dir, diag_base, file_subdir, suffixes, tmp_path):
    """Check that existing target files are not overwritten."""
    output_dir = tmp_path / hist_dir
    file_dir = output_dir / file_subdir if file_subdir else output_dir

    original_paths = make_files(str(file_dir), diag_base, suffixes)
    chdir(tmp_path)

    # Create files at the expected destination paths so there is something to protect
    expected_paths = [standardised_path(p, s) for p, s in zip(original_paths, suffixes)]
    for p in expected_paths:
        Path(p).touch()

    run(run_cmd)

    for p in expected_paths:
        assert_file_exists(p)
    for p in original_paths:
        assert_file_exists(p)
