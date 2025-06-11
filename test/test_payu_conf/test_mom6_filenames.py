import pytest
import pandas as pd

from os import makedirs, chdir
from subprocess import run
from pathlib import Path

scripts_base = Path(__file__).parents[2]
run_str = f"{scripts_base}/payu_config/archive_scripts/standardise_mom6_filenames.sh"

DIAG_BASE = "access-om3.mom6.h.test"


def assert_file_exists(p):
    if not Path(p).resolve().is_file():
        raise AssertionError("File does not exist: %s" % str(p))


def assert_f_not_exists(p):
    if Path(p).resolve().is_file():
        raise AssertionError("File exists and should not: %s" % str(p))


def yearly_files(dir_name, n, tmp_path, splits=0):
    """
    Make empty data files with `splits` option which will create split files
    such as `access-om3.mom6.h.test._2010.nc.0001` ... `.nc.000N` for each year.

    if `splits` is 0, then it will create files like `access-om3.mom6.h.test._2010.nc`,
    otherwise, it will create files like `access-om3.mom6.h.test._2010.nc.0001`,
    `access-om3.mom6.h.test._2010.nc.0002`, etc.
    """

    times = pd.date_range("2010-01-01", freq="YE", periods=n)

    out_dir = str(tmp_path) + "/" + dir_name + "/"
    paths = []

    for t in times:
        year = t.year
        if splits:
            for i in range(1, splits + 1):
                paths.append(f"{out_dir}{DIAG_BASE}._{year}.nc.{str(i).zfill(4)}")
        else:
            paths.append(f"{out_dir}{DIAG_BASE}._{str(year)}.nc")

    makedirs(out_dir)

    for p in paths:
        with open(p, "w") as f:
            f.close()

    for p in paths:
        assert_file_exists(p)

    return paths


# Add `splits` parameter
@pytest.mark.parametrize(
    "hist_dir, use_dir, n, splits",
    [
        ("archive/output000", False, 12, 0),
        ("archive/output999", False, 1, 5),
        ("archive/output9999", False, 1, 2),
        ("archive/output574", True, 12, 3),
    ],
)  # run this test with a several folder names and lengths, provide the directory as an argument sometimes
def test_true_case(hist_dir, use_dir, n, tmp_path, splits):

    yearly_paths = yearly_files(hist_dir, n, tmp_path, splits)
    chdir(tmp_path)
    output_dir = Path(yearly_paths[0]).parents[0]

    if not use_dir:  # default path
        run([run_str])
    else:  # provide path
        run(
            [
                run_str,
                "-d",
                output_dir,
            ],
        )

    expected_years = pd.date_range("2010-01-01", freq="YE", periods=n + 1)

    expected_paths = []
    for t in expected_years:
        year = t.year
        if splits:
            for i in range(1, splits + 1):
                expected_paths.append(
                    f"{output_dir}/{DIAG_BASE}.{year}.nc.{str(i).zfill(4)}"
                )
        else:
            expected_paths.append(f"{output_dir}/{DIAG_BASE}.{str(year)}.nc")

    for p in expected_paths[0:n]:
        assert_file_exists(p)

    for p in expected_paths[n]:
        assert_f_not_exists(p)

    for p in yearly_paths:
        assert_f_not_exists(p)


@pytest.mark.parametrize(
    "hist_dir, use_dir, n",
    [
        ("archive/output000", False, 12),
    ],
)
def test_dont_override(hist_dir, use_dir, n, tmp_path):
    """
    make some empty data files, and make some files where the files should be renamed to,
     and confirm it doesn't delete any of them
    """

    yearly_paths = yearly_files(hist_dir, n, tmp_path)
    chdir(tmp_path)
    output_dir = Path(yearly_paths[0]).parents[0]

    # write the expected output too
    expected_years = pd.date_range("2010-01-01", freq="YE", periods=n)

    expected_paths = [
        f"{output_dir}/{DIAG_BASE}.{str(t)[0:4]}.nc" for t in expected_years
    ]

    for p in expected_paths:
        with open(p, "w") as f:
            f.close()

    if not use_dir:  # default path
        run([run_str])
    else:  # provide path
        run(
            [
                run_str,
                "-d",
                output_dir,
            ],
        )

    for p in expected_paths:
        assert_file_exists(p)

    for p in yearly_paths:
        assert_file_exists(p)


# @pytest.mark.parametrize("hist_dir, ndays", [("Default", 31), ("Default", 27)])
# def test_no_override(hist_dir, ndays, hist_base, tmp_path):
#     """
#     Run the script to convert the daily data into monthly files, but the output filename already exists, and check nothing happens.
#     """

#     daily_paths = daily_files(hist_dir, hist_base, ndays, tmp_path)

#     chdir(tmp_path)
#     output_dir = Path(daily_paths[0]).parents[0]

#     expected_months = pd.date_range("2010-01-01", freq="ME", periods=1)

#     monthly_paths = [
#         f"{output_dir}/{hist_base}.{str(t)[0:7]}.nc" for t in expected_months
#     ]
#     for p in monthly_paths:
#         Path(p).touch()

#     run([run_str])

#     for p in daily_paths:
#         assert_file_exists(p)

#     for p in monthly_paths:
#         assert_file_exists(p)
