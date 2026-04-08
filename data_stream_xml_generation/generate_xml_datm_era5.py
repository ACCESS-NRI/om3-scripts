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
from datetime import datetime
from pathlib import Path
import calendar

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata

# ECMWF ERA5 documentation states that reanalysis accumulations and mean rates/fluxes are
# over the hour ending at the validity time. See the "Mean rates/fluxes and accumulations"
# section here:
# https://confluence.ecmwf.int/pages/viewpage.action?pageId=239349050
# For linear time interpolation, hourly interval fields are therefore shifted by -1800 s so the
# timestamp is at the midpoint of the represented hour. Instantaneous fields keep offset=0.
#
# stream_name, era5_prefix, [(source_var, cime_var), ...], tintalgo, offset_seconds
STREAM_SPECS = [
    ("ERA5.RAINC", "cp", [("cp", "Faxa_rainc")], "linear", -1800),
    ("ERA5.RAINL", "lsp", [("lsp", "Faxa_rainl")], "linear", -1800),
    ("ERA5.SNOWC", "csf", [("csf", "Faxa_snowc")], "linear", -1800),
    ("ERA5.SNOWL", "lsf", [("lsf", "Faxa_snowl")], "linear", -1800),
    ("ERA5.LWDN", "strd", [("strd", "Faxa_lwdn")], "linear", -1800),
    ("ERA5.SWDN", "ssrd", [("ssrd", "Faxa_swdn")], "linear", -1800),
    ("ERA5.SWNET", "ssr", [("ssr", "Faxa_swnet")], "linear", -1800),
    ("ERA5.SWVDR", "aluvp", [("aluvp", "Faxa_swvdr")], "linear", 0),
    ("ERA5.SWVDF", "aluvd", [("aluvd", "Faxa_swvdf")], "linear", 0),
    ("ERA5.SWNDR", "alnip", [("alnip", "Faxa_swndr")], "linear", 0),
    ("ERA5.SWNDF", "alnid", [("alnid", "Faxa_swndf")], "linear", 0),
    ("ERA5.SEN", "sshf", [("sshf", "Faxa_sen")], "linear", -1800),
    ("ERA5.LAT", "slhf", [("slhf", "Faxa_lat")], "linear", -1800),
    (
        "ERA5.SLP_10",
        "msl",
        [("msl", "Sa_pslv"), ("msl", "Sa_pbot")],
        "linear",
        0,
    ),
    (
        "ERA5.T_10",
        "2t",
        [("t2m", "Sa_t2m"), ("t2m", "Sa_tbot")],
        "linear",
        0,
    ),
    (
        "ERA5.TDEW",
        "2d",
        [("d2m", "Sa_tdew")],
        "linear",
        0,
    ),
    ("ERA5.U_10", "10u", [("u10", "Sa_u"), ("u10", "Sa_u10m")], "linear", 0),
    ("ERA5.V_10", "10v", [("v10", "Sa_v"), ("v10", "Sa_v10m")], "linear", 0),
]

if len(sys.argv) != 3:
    print("Usage: python generate_xml_datm_era5.py year_first year_last")
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
metadata_info = get_provenance_metadata(this_file, runcmd)

# Add metadata
metadata = SubElement(root, "metadata")
SubElement(metadata, "File_type").text = "DATM xml file provides forcing data"
SubElement(metadata, "date_generated").text = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)
SubElement(metadata, "history").text = metadata_info

# Generate stream info elements with changing years
for stream_name, era5_prefix, datavar_pairs, tintalgo, offset_seconds in STREAM_SPECS:
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

    SubElement(stream_info, "offset").text = str(offset_seconds)
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
                file_element.text = f"./INPUT/{year}/{era5_prefix}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{days_in_month}.nc"
# Convert the XML to a nicely formatted string
xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")

# Write the XML content to a file
with open("datm.streams.xml", "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_str)
