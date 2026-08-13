# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.

import os
import subprocess
import sys
from pathlib import Path

import pytest

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import username, get_email, write_readme, md5sum


@pytest.fixture(autouse=True)
def isolated_git_config(monkeypatch):
    """
    Prevent tests from picking up the developer's real global/system git config,
    so "no git identity configured" behaviour is deterministic regardless of the
    machine the tests run on. Still exercises real git commands, just against an
    empty config.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


def _init_repo(path, name=None, email=None):
    """Turn path into a real git repo, optionally setting user.name/user.email."""
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    if name is not None:
        subprocess.run(["git", "config", "user.name", name], cwd=path, check=True)
    if email is not None:
        subprocess.run(["git", "config", "user.email", email], cwd=path, check=True)


@pytest.fixture
def script_file(tmp_path):
    script_file = tmp_path / "generate_something.py"
    script_file.write_text("# dummy script\n")
    return script_file


# ----------------------------------------------------------------------------
# username()
# ----------------------------------------------------------------------------


def test_username_falls_back_to_user_env_without_git(script_file, monkeypatch):
    monkeypatch.setenv("USER", "testuser")
    assert username(str(script_file)) == "testuser"


def test_username_includes_git_name_when_available(tmp_path, script_file, monkeypatch):
    monkeypatch.setenv("USER", "testuser")
    _init_repo(tmp_path, name="Git Name")
    assert username(str(script_file)) == "testuser (Git Name)"


# ----------------------------------------------------------------------------
# get_email()
# ----------------------------------------------------------------------------


def test_get_email_returns_none_without_git(script_file):
    assert get_email(str(script_file)) is None


def test_get_email_returns_none_without_configured_email(tmp_path, script_file):
    _init_repo(tmp_path, name="Git Name")  # no email configured
    assert get_email(str(script_file)) is None


def test_get_email_returns_git_user_email(tmp_path, script_file):
    _init_repo(tmp_path, name="Git Name", email="someone@example.com")
    assert get_email(str(script_file)) == "someone@example.com"


# ----------------------------------------------------------------------------
# write_readme()
# ----------------------------------------------------------------------------


def test_write_readme_basic(tmp_path, script_file, monkeypatch):
    monkeypatch.setenv("USER", "testuser")
    _init_repo(tmp_path, name="Git Name", email="test@example.com")
    input_file = tmp_path / "input.nc"
    input_file.write_text("input data")
    output_file = tmp_path / "output.nc"
    output_file.write_text("output data")

    readme_path = Path(
        write_readme(
            str(tmp_path),
            str(script_file),
            str(output_file),
            [str(input_file)],
            "Created by someone on 2026-01-01, using https://example.com: python3 generate_something.py",
        )
    )

    # README is named after the output file, not a fixed "README.md"
    assert readme_path.name == "output.README.md"

    content = readme_path.read_text()
    assert "# output.nc" in content
    assert "Name: testuser (Git Name)" in content
    assert "Email: test@example.com" in content
    assert f"- {output_file}" in content
    assert "Created by someone on 2026-01-01" in content

    # input files are listed with their md5 hash
    assert f"- {input_file} (md5 hash: {md5sum(str(input_file))})" in content

    # licence defaults to a placeholder, not an assumed value
    assert "[List any licenses and/or restrictions placed on the data]" in content


def test_write_readme_licence(tmp_path, script_file, monkeypatch):
    monkeypatch.setenv("USER", "testuser")
    _init_repo(tmp_path, name="Git Name", email="test@example.com")
    output_file = tmp_path / "output.nc"
    output_file.write_text("output data")

    readme_path = Path(
        write_readme(
            str(tmp_path),
            str(script_file),
            str(output_file),
            None,
            "hist",
            licence="CC-BY-4.0",
        )
    )

    content = readme_path.read_text()
    assert "### Licensing/restrictions\n\nCC-BY-4.0" in content


def test_write_readme_no_input_files(tmp_path, script_file, monkeypatch):
    monkeypatch.setenv("USER", "testuser")
    _init_repo(tmp_path, name="Git Name", email="test@example.com")
    output_file = tmp_path / "output.nc"
    output_file.write_text("output data")

    readme_path = Path(
        write_readme(str(tmp_path), str(script_file), str(output_file), None, "hist")
    )

    content = readme_path.read_text()
    assert (
        "### Links/relationships\n\nOther files used to prepare this data:\n\nN/A"
        in content
    )


def test_write_readme_no_email_uses_placeholder(tmp_path, script_file, monkeypatch):
    monkeypatch.setenv("USER", "testuser")
    _init_repo(tmp_path, name="Git Name")  # no email configured
    output_file = tmp_path / "output.nc"
    output_file.write_text("output data")

    readme_path = Path(
        write_readme(str(tmp_path), str(script_file), str(output_file), None, "hist")
    )

    content = readme_path.read_text()
    assert "Email: [Email of data custodian]" in content


def test_write_readme_multiple_output_files(tmp_path, script_file, monkeypatch):
    monkeypatch.setenv("USER", "testuser")
    _init_repo(tmp_path, name="Git Name", email="test@example.com")
    output_a = tmp_path / "a.nc"
    output_a.write_text("a")
    output_b = tmp_path / "b.nc"
    output_b.write_text("b")

    readme_path = Path(
        write_readme(
            str(tmp_path),
            str(script_file),
            [str(output_a), str(output_b)],
            None,
            "hist",
        )
    )

    # README filename is based on the first output file
    assert readme_path.name == "a.README.md"

    content = readme_path.read_text()
    assert "# a.nc, b.nc" in content
    assert f"- {output_a}" in content
    assert f"- {output_b}" in content


def test_write_readme_multiple_input_files_hashed(tmp_path, script_file, monkeypatch):
    monkeypatch.setenv("USER", "testuser")
    _init_repo(tmp_path, name="Git Name", email="test@example.com")
    output_file = tmp_path / "output.nc"
    output_file.write_text("output data")
    input_a = tmp_path / "a.nc"
    input_a.write_text("aaa")
    input_b = tmp_path / "b.nc"
    input_b.write_text("bbb")

    readme_path = Path(
        write_readme(
            str(tmp_path),
            str(script_file),
            str(output_file),
            [str(input_a), str(input_b)],
            "hist",
        )
    )

    content = readme_path.read_text()
    assert f"- {input_a} (md5 hash: {md5sum(str(input_a))})" in content
    assert f"- {input_b} (md5 hash: {md5sum(str(input_b))})" in content
