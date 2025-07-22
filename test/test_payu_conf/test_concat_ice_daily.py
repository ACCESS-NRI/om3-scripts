import pytest
import xarray as xr
import numpy as np
import pandas as pd

from os import makedirs, chdir
from pathlib import Path

from payu_config.postscript.concat_ice_daily import concat_ice_daily


def assert_file_exists(p):
    if not Path(p).resolve().is_file():
        raise AssertionError("File does not exist: %s" % str(p))


def assert_f_not_exists(p):
    if Path(p).resolve().is_file():
        raise AssertionError("File exists and should not: %s" % str(p))


def dummy_files(
    dir_name, hist_base, n, tmp_path, ini_date="2010-01-01 12:00", freq="D"
):
    """
    Make `ndays` of fake data, starting at `ini_date`,
    and then write to daily files.

    request = (path, ndays)
    e.g. request = ("archive/output000", "365")

    """

    if dir_name == "Default":
        dir_name = "archive/output000"

    nx = 30
    ny = 50

    da = xr.DataArray(
        np.random.rand(n, nx, ny),
        dims=[
            "time",
            "x",
            "y",
        ],
        coords={
            "time": xr.date_range(
                ini_date,
                freq=freq,
                periods=n,
                calendar="proleptic_gregorian",
                use_cftime=True,
            )
        },
    )
    ds = da.to_dataset(name="aice")

    ds.attrs["comment2"] = "Written by model on day xx"
    ds.attrs["comment3"] = "pytest made this"

    out_dir = str(tmp_path) + "/" + dir_name + "/"
    if freq == "D":
        paths = [f"{out_dir}{hist_base}.{str(t.values)[0:10]}.nc" for t in ds.time]
    elif freq == "ME":
        paths = [f"{out_dir}{hist_base}.{str(t.values)[0:7]}.nc" for t in ds.time]
    datasets = [ds.sel(time=slice(t, t)) for t in ds.time]

    makedirs(out_dir, exist_ok=True)
    xr.save_mfdataset(datasets, paths, unlimited_dims=["time"])

    return paths


@pytest.fixture(params=["access-om3.cice.1day.mean"])
def hist_base(request):
    return str(request.param)


@pytest.mark.parametrize(
    "hist_dir, ndays, use_dir, nmonths",
    [
        ("Default", 365, False, 12),
        ("archive/output999", 31, False, 1),
        ("archive/output9999", 31, False, 1),
        ("archive/output574", 365, True, 12),
    ],
)  # run this test with a several folder names and lengths, provide the directory as an argument sometimes
def test_true_case(hist_dir, ndays, use_dir, nmonths, hist_base, tmp_path):
    """
    Run the script to convert the daily data into monthly files, and check the monthly files and the daily files dont exist.
    """

    daily_paths = dummy_files(hist_dir, hist_base, ndays, tmp_path)
    chdir(tmp_path)
    output_dir = Path(daily_paths[0]).parents[0]

    if not use_dir:  # default path
        concat_ice_daily(assume_gadi=False)
    else:  # provide path
        concat_ice_daily(directory=output_dir, assume_gadi=False)

    expected_months = pd.date_range("2010-01-01", freq="ME", periods=nmonths + 1)

    # valid output filenames
    monthly_paths = [
        f"{output_dir}/{hist_base}.{str(t)[0:7]}.nc" for t in expected_months
    ]

    for p in monthly_paths[:nmonths]:
        assert_file_exists(p)

    for p in monthly_paths[nmonths]:
        assert_f_not_exists(p)

    for p in daily_paths:
        assert_f_not_exists(p)


@pytest.mark.parametrize("hist_dir, ndays", [("Default", 1), ("Default", 30)])
def test_incomplete_month(hist_dir, ndays, hist_base, tmp_path):
    """
    Run the script to convert the daily data into monthly files, with less than 28 days data, and check no things happen.
    """

    daily_paths = dummy_files(hist_dir, hist_base, ndays, tmp_path)

    chdir(tmp_path)
    output_dir = Path(daily_paths[0]).parents[0]

    with pytest.raises(Exception):
        concat_ice_daily(directory=output_dir, assume_gadi=False)

    expected_months = pd.date_range("2010-01-01", freq="ME", periods=1)
    monthly_paths = [
        f"{output_dir}/{hist_base}.{str(t)[0:7]}.nc" for t in expected_months
    ]

    for p in daily_paths:
        assert_file_exists(p)

    for p in monthly_paths:
        assert_f_not_exists(p)


@pytest.mark.parametrize(
    "hist_dir, ndays", [("Default", 27), ("Default", 31), ("Default", 59)]
)
def test_no_del_result_does_not_exist(hist_dir, ndays, hist_base, tmp_path):
    """
    Run the script to convert the daily data into monthly files, but the output filename already exists with different data.
    In this case, don't delete anything because the monthly and daily data are different.
    """

    daily_paths = dummy_files(hist_dir, hist_base, ndays, tmp_path)

    chdir(tmp_path)
    output_dir = Path(daily_paths[0]).parents[0]

    monthly_paths = dummy_files(hist_dir, hist_base, 2, tmp_path, freq="ME")

    assert len(monthly_paths) == 2

    # check test setup
    for p in daily_paths:
        assert_file_exists(p)

    for p in monthly_paths:
        assert_file_exists(p)

    # run function
    with pytest.raises(Exception):
        concat_ice_daily(directory=None, assume_gadi=False)

    for p in daily_paths:
        assert_file_exists(p)

    for p in monthly_paths:
        assert_file_exists(p)


@pytest.mark.parametrize(
    "year, ndays, nmonths, use_dir",
    [
        (2012, 366, 12, False),  # Leap year
        (2013, 365, 12, True),  # No leap year
    ],
)
def test_leap_year(year, ndays, nmonths, use_dir, hist_base, tmp_path):
    """
    Ensures correctly handles Feb 29 for the leap year and
    Feb 28 for the no-leap year.
    """
    hist_dir = "Default"

    daily_paths = dummy_files(
        hist_dir, hist_base, ndays, tmp_path, ini_date=f"{year}-01-01 12:00"
    )

    chdir(tmp_path)
    output_dir = Path(daily_paths[0]).parents[0]

    if not use_dir:
        concat_ice_daily(directory=None, assume_gadi=False)
    else:
        concat_ice_daily(directory=output_dir, assume_gadi=False)

    expected_months = pd.date_range(f"{year}-01-01", freq="ME", periods=nmonths + 1)
    monthly_paths = [
        f"{output_dir}/{hist_base}.{str(t)[:7]}.nc" for t in expected_months
    ]

    for p in monthly_paths[:nmonths]:
        assert_file_exists(p)

    for p in daily_paths:
        assert_f_not_exists(p)
