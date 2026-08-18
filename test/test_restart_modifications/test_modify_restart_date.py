# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.

import sys
import subprocess
from pathlib import Path

import pytest
import netCDF4 as nc

from restart_modifications.modify_restart_date import (
    find_restart_files,
    read_config_restart_path,
    write_config_restart_path,
    ensure_gitignore_entry,
    rewrite_cpl_restart,
    rewrite_cice_restart,
)

SCRIPT_PATH = (
    Path(__file__).parents[2] / "restart_modifications" / "modify_restart_date.py"
)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def make_cpl_restart(path, ymd=19580101, tod=0):
    with nc.Dataset(path, "w") as ds:
        for name in ("start_ymd", "start_tod", "curr_ymd", "curr_tod"):
            ds.createVariable(name, "i4")
        ds.variables["start_ymd"][:] = ymd
        ds.variables["start_tod"][:] = tod
        ds.variables["curr_ymd"][:] = ymd
        ds.variables["curr_tod"][:] = tod


def make_cice_restart(path, myear=1958, mmonth=1, mday=1, msec=0):
    with nc.Dataset(path, "w") as ds:
        ds.myear = myear
        ds.mmonth = mmonth
        ds.mday = mday
        ds.msec = msec


def make_restart_set(restart_dir, date_str="1958-01-01"):
    """
    Create a minimal but representative ACCESS-OM3 restart set (real cpl/cice
    netCDFs, dummy datm/drof/mom6 files including an uncollated per-tile
    fragment, an undated tracer restart, and rpointer files).
    """
    restart_dir.mkdir(parents=True, exist_ok=True)

    make_cpl_restart(restart_dir / f"access-om3.cpl.r.{date_str}-00000.nc")
    make_cice_restart(restart_dir / f"access-om3.cice.r.{date_str}-00000.nc")
    touch(restart_dir / f"access-om3.datm.r.{date_str}-00000.nc")
    touch(restart_dir / f"access-om3.drof.r.{date_str}-00000.nc")
    touch(restart_dir / f"access-om3.mom6.r.{date_str}-00000.nc.0000")
    touch(restart_dir / f"access-om3.mom6.r.{date_str}-00000.nc.0001")
    touch(restart_dir / "ocean_wombatlite_airsea_flux.res.nc.0000")

    (restart_dir / "rpointer.cpl").write_text(f"access-om3.cpl.r.{date_str}-00000.nc\n")
    (restart_dir / "rpointer.ice").write_text(
        f"./access-om3.cice.r.{date_str}-00000.nc\n"
    )
    (restart_dir / "rpointer.atm").write_text(
        f"access-om3.datm.r.{date_str}-00000.nc\n"
    )
    (restart_dir / "rpointer.rof").write_text(
        f"access-om3.drof.r.{date_str}-00000.nc\n"
    )
    (restart_dir / "rpointer.ocn").write_text(
        f"access-om3.mom6.r.{date_str}-00000.nc\n"
    )

    return restart_dir


def init_config_repo(config_dir, restart_path):
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        "model: access-om3\n" f"restart: {restart_path}\n" "restart_freq: 1YS\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=config_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=config_dir, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=config_dir, check=True)
    subprocess.run(["git", "add", "config.yaml"], cwd=config_dir, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial config"], cwd=config_dir, check=True
    )


def run_script(config_dir, new_date):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config_dir",
            str(config_dir),
            "--new_date",
            new_date,
        ],
        capture_output=True,
        text=True,
    )


# --- find_restart_files -------------------------------------------------


def test_find_restart_files_basic(tmp_path):
    restart_dir = make_restart_set(tmp_path / "restart065")

    all_files, old_str = find_restart_files(str(restart_dir))

    assert old_str == "1958-01-01"
    names = {p.name for p in map(Path, all_files)}
    assert "rpointer.cpl" not in names  # rpointer.* excluded
    assert "access-om3.cpl.r.1958-01-01-00000.nc" in names
    assert "access-om3.mom6.r.1958-01-01-00000.nc.0000" in names  # uncollated tile
    assert "ocean_wombatlite_airsea_flux.res.nc.0000" in names  # undated file


def test_find_restart_files_multiple_dates_raises(tmp_path):
    restart_dir = tmp_path / "restart"
    restart_dir.mkdir()
    touch(restart_dir / "access-om3.cice.r.1958-01-01-00000.nc")
    touch(restart_dir / "access-om3.cpl.r.1959-01-01-00000.nc")

    with pytest.raises(ValueError, match="Multiple restart dates"):
        find_restart_files(str(restart_dir))


def test_find_restart_files_no_dates_raises(tmp_path):
    restart_dir = tmp_path / "restart"
    restart_dir.mkdir()
    touch(restart_dir / "ocean_wombatlite_airsea_flux.res.nc.0000")

    with pytest.raises(ValueError, match="No dated restart files"):
        find_restart_files(str(restart_dir))


# --- config.yaml restart: field ------------------------------------------


def test_read_write_config_restart_path(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: access-om3\nrestart: /a/b/c\nrestart_freq: 1YS\n")

    assert read_config_restart_path(str(config_path)) == "/a/b/c"

    write_config_restart_path(str(config_path), "/x/y/z")

    assert read_config_restart_path(str(config_path)) == "/x/y/z"
    # other lines untouched
    content = config_path.read_text()
    assert "model: access-om3" in content
    assert "restart_freq: 1YS" in content


def test_read_write_config_restart_path_folded_value(tmp_path):
    # payu clone writes long restart: values folded onto the next, indented
    # line rather than on the same line - both are valid YAML.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model: access-om3\n" "restart_freq: 1YS\n" "restart: \n" "  /a/b/c\n"
    )

    assert read_config_restart_path(str(config_path)) == "/a/b/c"

    write_config_restart_path(str(config_path), "/x/y/z")

    # folded form is collapsed to a single line on write
    assert read_config_restart_path(str(config_path)) == "/x/y/z"
    content = config_path.read_text()
    assert "restart: /x/y/z\n" in content
    assert "model: access-om3" in content
    assert "restart_freq: 1YS" in content


def test_read_config_restart_path_missing_raises(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: access-om3\n")

    with pytest.raises(ValueError, match="No top-level 'restart:' field"):
        read_config_restart_path(str(config_path))


def test_write_config_restart_path_missing_raises(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: access-om3\n")

    with pytest.raises(ValueError, match="No top-level 'restart:' field"):
        write_config_restart_path(str(config_path), "/x/y/z")


# --- .gitignore -----------------------------------------------------------


def test_ensure_gitignore_entry_creates_and_is_idempotent(tmp_path):
    ensure_gitignore_entry(str(tmp_path), "initial_restart/")
    ensure_gitignore_entry(str(tmp_path), "initial_restart/")

    lines = (tmp_path / ".gitignore").read_text().splitlines()
    assert lines.count("initial_restart/") == 1


# --- rewrite_cpl_restart / rewrite_cice_restart ----------------------------


def test_rewrite_cpl_restart(tmp_path):
    from datetime import date

    path = tmp_path / "cpl.nc"
    make_cpl_restart(path, ymd=19580101)

    rewrite_cpl_restart(str(path), date(2005, 1, 1), {"history": "test provenance"})

    with nc.Dataset(path) as ds:
        assert int(ds.variables["start_ymd"][...]) == 20050101
        assert int(ds.variables["curr_ymd"][...]) == 20050101
        assert int(ds.variables["start_tod"][...]) == 0
        assert int(ds.variables["curr_tod"][...]) == 0
        assert ds.history == "test provenance"


def test_rewrite_cice_restart(tmp_path):
    from datetime import date

    path = tmp_path / "cice.nc"
    make_cice_restart(path, myear=1958, mmonth=1, mday=1)

    rewrite_cice_restart(str(path), date(2005, 3, 17), {"history": "test provenance"})

    with nc.Dataset(path) as ds:
        assert ds.myear == 2005
        assert ds.mmonth == 3
        assert ds.mday == 17
        assert ds.msec == 0
        assert ds.history == "test provenance"


# --- end-to-end CLI ---------------------------------------------------


def test_main_end_to_end(tmp_path):
    restart_dir = make_restart_set(tmp_path / "source" / "restart065")
    config_dir = tmp_path / "config"
    init_config_repo(config_dir, restart_dir)

    result = run_script(config_dir, "2005-01-01")
    assert result.returncode == 0, result.stderr

    initial_restart = config_dir / "initial_restart"
    assert (initial_restart / "access-om3.cpl.r.2005-01-01-00000.nc").is_file()
    assert (initial_restart / "access-om3.cice.r.2005-01-01-00000.nc").is_file()
    assert (initial_restart / "access-om3.mom6.r.2005-01-01-00000.nc.0000").is_file()
    assert (initial_restart / "ocean_wombatlite_airsea_flux.res.nc.0000").is_file()
    assert (
        initial_restart / "rpointer.ocn"
    ).read_text().strip() == "access-om3.mom6.r.2005-01-01-00000.nc"

    with nc.Dataset(initial_restart / "access-om3.cpl.r.2005-01-01-00000.nc") as ds:
        assert int(ds.variables["curr_ymd"][...]) == 20050101
    with nc.Dataset(initial_restart / "access-om3.cice.r.2005-01-01-00000.nc") as ds:
        assert ds.myear == 2005

    # a single consolidated README.md describing the whole restart folder was
    # written (not one per restart file), and it's inside the gitignored
    # initial_restart/ dir so it doesn't get committed
    readmes = list(initial_restart.glob("*.README.md"))
    assert len(readmes) == 1

    # config.yaml updated and committed
    assert read_config_restart_path(str(config_dir / "config.yaml")) == str(
        initial_restart
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=config_dir, capture_output=True, text=True
    )
    assert status.stdout.strip() == ""  # working tree clean - restart: change committed

    log = subprocess.run(
        ["git", "log", "-1", "--format=%s"],
        cwd=config_dir,
        capture_output=True,
        text=True,
    )
    assert "1958-01-01" in log.stdout and "2005-01-01" in log.stdout

    # initial_restart/ is gitignored, not left untracked
    check_ignore = subprocess.run(
        ["git", "check-ignore", "initial_restart"],
        cwd=config_dir,
        capture_output=True,
        text=True,
    )
    assert check_ignore.returncode == 0


def test_main_ww3_present_raises(tmp_path):
    restart_dir = make_restart_set(tmp_path / "source" / "restart065")
    touch(restart_dir / "access-om3.ww3.r.1958-01-01-00000.nc")
    config_dir = tmp_path / "config"
    init_config_repo(config_dir, restart_dir)

    result = run_script(config_dir, "2005-01-01")

    assert result.returncode != 0
    assert "ww3" in result.stderr.lower()
    assert not (config_dir / "initial_restart").exists()


def test_main_rerun_raises(tmp_path):
    restart_dir = make_restart_set(tmp_path / "source" / "restart065")
    config_dir = tmp_path / "config"
    init_config_repo(config_dir, restart_dir)

    first = run_script(config_dir, "2005-01-01")
    assert first.returncode == 0, first.stderr

    second = run_script(config_dir, "1979-01-01")
    assert second.returncode != 0
    assert "already exists" in second.stderr
