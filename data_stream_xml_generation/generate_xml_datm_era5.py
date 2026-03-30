#!/usr/bin/env python3
# Copyright 2023 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# Generate a datm xml file that contains a time-series of input atmosphere data files where all the fields in the stream are located.

# To run:
#   python generate_xml_datm_era5.py <year_first> <year_last>
# To generate IAF xml file, set year_first and year_last to the forcing period
# To generate RYF xml file, set year_first==year_last


# Contact: Ezhilsabareesh Kannadasan <ezhilsabareesh.kannadasan@anu.edu.au>

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import sys
import os
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
import calendar

path_root = Path(__file__).parent.parent
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata

# stream_name, era5_prefix, [(source_var, cime_var), ...], tintalgo
STREAM_SPECS = [
    ("ERA5.RAINC", "cp", [("cp", "Faxa_rainc")], "linear"),
    ("ERA5.RAINL", "lsp", [("lsp", "Faxa_rainl")], "linear"),
    ("ERA5.SNOWC", "csf", [("csf", "Faxa_snowc")], "linear"),
    ("ERA5.SNOWL", "lsf", [("lsf", "Faxa_snowl")], "linear"),
    ("ERA5.LWDN", "strd", [("strd", "Faxa_lwdn")], "linear"),
    ("ERA5.SWDN", "ssrd", [("ssrd", "Faxa_swdn")], "coszen"),
    ("ERA5.SWNET", "ssr", [("ssr", "Faxa_swnet")], "coszen"),
    ("ERA5.SWVDR", "aluvp", [("aluvp", "Faxa_swvdr")], "coszen"),
    ("ERA5.SWVDF", "aluvd", [("aluvd", "Faxa_swvdf")], "coszen"),
    ("ERA5.SWNDR", "alnip", [("alnip", "Faxa_swndr")], "coszen"),
    ("ERA5.SWNDF", "alnid", [("alnid", "Faxa_swndf")], "coszen"),
    ("ERA5.SEN", "sshf", [("sshf", "Faxa_sen")], "linear"),
    ("ERA5.LAT", "slhf", [("slhf", "Faxa_lat")], "linear"),
    (
        "ERA5.SLP_10",
        "msl",
        [("msl", "Sa_pslv"), ("msl", "Sa_pbot")],
        "linear",
    ),
    (
        "ERA5.T_10",
        "2t",
        [("t2m", "Sa_t2m"), ("t2m", "Sa_tbot")],
        "linear",
    ),
    (
        "ERA5.TDEW",
        "2d",
        [("d2m", "Sa_tdew")],
        "linear",
    ),
    ("ERA5.U_10", "10u", [("u10", "Sa_u"), ("u10", "Sa_u10m")], "linear"),
    ("ERA5.V_10", "10v", [("v10", "Sa_v"), ("v10", "Sa_v10m")], "linear"),
]

DATE_GENERATED_ENV = "DATM_XML_DATE_GENERATED"
HISTORY_ENV = "DATM_XML_HISTORY"


def _strip_suffix(value, suffix):
    if suffix and value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def _strip_prefix(value, prefix):
    if prefix and value.startswith(prefix):
        return value[len(prefix) :]
    return value


def _git_output(args):
    return subprocess.check_output(args).decode("ascii").strip()


def _compatible_provenance_metadata(file_path, runcmd):
    dirname = os.path.dirname(file_path)
    created_by = os.environ.get("USER", "unknown")

    try:
        git_name = _git_output(["git", "-C", dirname, "config", "user.name"])
        created_by = f"{created_by} ({git_name})"
    except subprocess.CalledProcessError:
        pass

    provenance = (
        f"Created by {created_by} on {datetime.now().strftime('%Y-%m-%d')}, using "
    )

    try:
        url = _git_output(
            ["git", "-C", dirname, "config", "--get", "remote.origin.url"]
        )
        url = _strip_suffix(url, ".git")
        if url.startswith("git@github.com:"):
            url = f"https://github.com/{_strip_prefix(url, 'git@github.com:')}"

        top_level_dir = _git_output(
            ["git", "-C", dirname, "rev-parse", "--show-toplevel"]
        )
        rel_path = _strip_prefix(file_path, top_level_dir)
        git_hash = _git_output(["git", "-C", dirname, "rev-parse", "HEAD"])
        provenance += f"{url}/blob/{git_hash}{rel_path}: "
    except subprocess.CalledProcessError:
        warnings.warn(
            f"{file_path} not under git version control! "
            "Add your file to a repository before generating any production output."
        )
        provenance += f"{file_path}: "

    return provenance + runcmd


def get_metadata_history(file_path, runcmd):
    metadata_override = os.environ.get(HISTORY_ENV)
    if metadata_override:
        return metadata_override

    if hasattr(str, "removeprefix") and hasattr(str, "removesuffix"):
        return get_provenance_metadata(file_path, runcmd)

    return _compatible_provenance_metadata(file_path, runcmd)


def get_date_generated():
    return os.environ.get(DATE_GENERATED_ENV) or datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

if len(sys.argv) != 3:
    print("Usage: python generate_xml_datm_era.py year_first year_last")
    sys.exit(1)

try:
    year_first = int(sys.argv[1])
    year_last = int(sys.argv[2])

except ValueError:
    print("Year values must be integers")
    sys.exit(1)

year_align = year_first

# Create the root element
root = Element("file", id="stream", version="2.0")

# Obtain metadata
this_file = sys.argv[0]
runcmd = " ".join(sys.argv)
try:
    metadata_info = get_metadata_history(this_file, runcmd)
except Exception as err:
    print(
        f"WARNING: failed to collect provenance metadata ({err}); using fallback history string.",
        file=sys.stderr,
    )
    metadata_info = f"{Path(this_file).name} {runcmd}"

# Add metadata
metadata = SubElement(root, "metadata")
SubElement(metadata, "File_type").text = "DATM xml file provides forcing data"
SubElement(metadata, "date_generated").text = get_date_generated()
SubElement(metadata, "history").text = metadata_info

# Generate stream info elements with changing years
for stream_name, era5_prefix, datavar_pairs, tintalgo in STREAM_SPECS:
    stream_info = SubElement(root, "stream_info", name=stream_name)
    if year_first == year_last:
        SubElement(stream_info, "taxmode").text = "cycle"
    else:
        SubElement(stream_info, "taxmode").text = "extend"
    SubElement(stream_info, "readmode").text = "single"
    SubElement(stream_info, "mapalgo").text = "bilinear"
    SubElement(stream_info, "dtlimit").text = "1.e30"
    SubElement(stream_info, "year_first").text = str(year_first)
    SubElement(stream_info, "year_last").text = str(year_last)
    SubElement(stream_info, "year_align").text = str(year_align)
    SubElement(stream_info, "vectors").text = "null"
    SubElement(stream_info, "meshfile").text = "./INPUT/ERA5-datm-ESMFmesh.nc"
    SubElement(stream_info, "lev_dimname").text = "null"

    datafiles = SubElement(stream_info, "datafiles")
    datavars = SubElement(stream_info, "datavars")

    for src_var, cime_var in datavar_pairs:
        var_element = SubElement(datavars, "var")
        var_element.text = f"{src_var}  {cime_var}"

    SubElement(stream_info, "tintalgo").text = tintalgo

    # Use the first source variable for RYF file naming.
    driver_src_var = datavar_pairs[0][0]

    for year in range(year_first, year_last + 1):
        if year_first == year_last:
            file_element = SubElement(datafiles, "file")
            file_element.text = (
                f"./INPUT/RYF.{driver_src_var}.{year + 90}_{year + 91}.nc"
            )
        else:
            for month in range(1, 13):  # Loop through months (January to December)
                days_in_month = calendar.monthrange(year, month)[1]  # Get the number of days in the month
                file_element = SubElement(datafiles, "file")
                file_element.text = f"/g/data/rt52/era5/single-levels/reanalysis/{era5_prefix}/{year}/{era5_prefix}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{days_in_month}.nc"
# Convert the XML to a nicely formatted string
xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")

# Write the XML content to a file
with open("datm.streams.xml", "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)
