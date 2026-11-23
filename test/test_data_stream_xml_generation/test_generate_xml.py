# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.

import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

SCRIPT_DIR = Path(__file__).parents[2] / "data_stream_xml_generation"

JRA55_FLUX_STREAMS = {
    "JRA55do.PRSN",
    "JRA55do.PRRN",
    "JRA55do.LWDN",
    "JRA55do.SWDN",
}
JRA55_STREAMS = JRA55_FLUX_STREAMS | {
    "JRA55do.Q_10",
    "JRA55do.SLP_10",
    "JRA55do.T_10",
    "JRA55do.U_10",
    "JRA55do.V_10",
}

ERA5_FLUX_PREFIXES = ["cp", "lsp", "csf", "lsf", "strd", "ssrd", "ssr"]
ERA5_STATE_PREFIXES = ["msl", "2t", "2d", "10u", "10v"]


def run_script(script, args, cwd):
    """Run a generate_xml script in cwd, returning the CompletedProcess."""
    return subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def read_streams(xml_file):
    """Return {stream name: stream_info element} from a streams xml file."""
    root = ElementTree.parse(xml_file).getroot()
    return {s.get("name"): s for s in root.findall("stream_info")}


def mapalgos(streams):
    return {name: s.findtext("mapalgo") for name, s in streams.items()}


@pytest.fixture
def era5_root(tmp_path):
    """Empty ERA5-like tree: two monthly files per variable for 1990 and 1991."""
    root = tmp_path / "era5"
    for prefix in ERA5_FLUX_PREFIXES + ERA5_STATE_PREFIXES:
        for year in [1990, 1991]:
            year_dir = root / prefix / str(year)
            year_dir.mkdir(parents=True)
            for month in ["01", "02"]:
                (
                    year_dir
                    / f"{prefix}_era5_oper_sfc_{year}{month}01-{year}{month}28.nc"
                ).touch()
    return root


@pytest.mark.parametrize("years", [["1990", "1990"], ["1958", "2024"]])
@pytest.mark.parametrize("flux_mapalgo", [None, "consf", "patch", "bilinear"])
def test_datm_jra55_mapalgo(tmp_path, years, flux_mapalgo):
    """
    Generate the JRA55-do datm streams (RYF and IAF) with and without --flux-mapalgo.
    All nine streams should be present, the flux streams should use the requested
    mapalgo (consf by default) and all other streams should use patch.
    """
    args = years if flux_mapalgo is None else [*years, "--flux-mapalgo", flux_mapalgo]
    result = run_script("generate_xml_datm_jra55.py", args, tmp_path)
    assert result.returncode == 0, result.stderr

    streams = read_streams(tmp_path / "datm.streams.xml")
    assert set(streams) == JRA55_STREAMS

    expected_flux = flux_mapalgo or "consf"
    for name, algo in mapalgos(streams).items():
        assert algo == (expected_flux if name in JRA55_FLUX_STREAMS else "patch"), name


@pytest.mark.parametrize("years", [["1990", "1990"], ["1990", "1991"]])
@pytest.mark.parametrize("flux_mapalgo", [None, "consf", "patch", "bilinear"])
def test_datm_era5_mapalgo(tmp_path, era5_root, years, flux_mapalgo):
    """
    Generate the ERA5 datm streams (RYF and IAF) from a synthetic input tree, with and
    without --flux-mapalgo. Every stream should list the input files for each year,
    the flux streams should use the requested mapalgo (consf by default) and the
    state streams should use patch.
    """
    args = [*years, "--input-base", str(era5_root)]
    if flux_mapalgo is not None:
        args += ["--flux-mapalgo", flux_mapalgo]
    result = run_script("generate_xml_datm_era5.py", args, tmp_path)
    assert result.returncode == 0, result.stderr

    streams = read_streams(tmp_path / "datm.streams.xml")
    assert len(streams) == len(ERA5_FLUX_PREFIXES) + len(ERA5_STATE_PREFIXES)

    expected_flux = flux_mapalgo or "consf"
    n_years = int(years[1]) - int(years[0]) + 1
    for name, stream in streams.items():
        files = [f.text for f in stream.find("datafiles").findall("file")]
        prefix = Path(files[0]).relative_to("INPUT").parts[0]
        assert len(files) == 2 * n_years, name

        if prefix in ERA5_FLUX_PREFIXES:
            assert stream.findtext("mapalgo") == expected_flux, name
        else:
            assert prefix in ERA5_STATE_PREFIXES, name
            assert stream.findtext("mapalgo") == "patch", name


@pytest.mark.parametrize(
    "script, extra_args",
    [
        ("generate_xml_datm_jra55.py", []),
        ("generate_xml_datm_era5.py", ["--input-base", "."]),
    ],
)
def test_datm_invalid_mapalgo(tmp_path, script, extra_args):
    """
    An unsupported --flux-mapalgo should be rejected by argparse for both datm
    scripts, without writing a streams file.
    """
    args = ["1990", "1990", *extra_args, "--flux-mapalgo", "nn"]
    result = run_script(script, args, tmp_path)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert not (tmp_path / "datm.streams.xml").exists()


@pytest.mark.parametrize("years", [["1990", "1990"], ["1958", "2024"]])
@pytest.mark.parametrize(
    "streams_arg, expected",
    [
        (None, ["JRA55do.FRIVER", "JRA55do.LICALVF"]),
        (["friver", "licalvf"], ["JRA55do.FRIVER", "JRA55do.LICALVF"]),
        (["licalvf", "friver"], ["JRA55do.FRIVER", "JRA55do.LICALVF"]),
        (["friver"], ["JRA55do.FRIVER"]),
        (["licalvf"], ["JRA55do.LICALVF"]),
    ],
)
def test_drof_jra55_streams(tmp_path, years, streams_arg, expected):
    """
    Generate the JRA55-do drof streams (RYF and IAF) with and without --streams.
    Only the requested streams should be written, in a fixed order regardless of the
    order given, each with the matching datavar, consd mapalgo and one input file
    per year.
    """
    args = years if streams_arg is None else [*years, "--streams", *streams_arg]
    result = run_script("generate_xml_drof_jra55.py", args, tmp_path)
    assert result.returncode == 0, result.stderr

    streams = read_streams(tmp_path / "drof.streams.xml")
    assert list(streams) == expected

    for name, stream in streams.items():
        var_prefix = name.split(".")[1].lower()
        assert stream.findtext("mapalgo") == "consd"
        assert stream.find("datavars").find("var").text.split()[0] == var_prefix
        files = [f.text for f in stream.find("datafiles").findall("file")]
        assert len(files) == int(years[1]) - int(years[0]) + 1
        assert all(f".{var_prefix}" in f or f"/{var_prefix}_" in f for f in files)


@pytest.mark.parametrize("streams_arg", [["foo"], []])
def test_drof_jra55_invalid_streams(tmp_path, streams_arg):
    """
    An unknown stream name, or --streams with no names, should be rejected without
    writing a streams file.
    """
    result = run_script(
        "generate_xml_drof_jra55.py",
        ["1990", "1990", "--streams", *streams_arg],
        tmp_path,
    )
    assert result.returncode != 0
    assert not (tmp_path / "drof.streams.xml").exists()
