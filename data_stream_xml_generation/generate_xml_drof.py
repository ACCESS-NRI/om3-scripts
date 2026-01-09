#!/usr/bin/env python3
# Copyright 2023 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# Generate a drof xml file that contains a time-series of runoff data files where all the fields in the stream are located.

# To run:
#   python generate_xml_datm.py <year_first> <year_last>
# To generate IAF xml file, set year_first and year_last to the forcing period
# To generate RYF xml file, set year_first==year_last

# Contact: Ezhilsabareesh Kannadasan <ezhilsabareesh.kannadasan@anu.edu.au>

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import sys
from datetime import datetime
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from scripts_common import get_provenance_metadata

source_data = "jra55v1p6"
# source_data = "jra55v1p4"

if len(sys.argv) != 3:
    print("Usage: python generate_xml_drof.py year_first year_last")
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
SubElement(metadata, "File_type").text = "DROF xml file provides river runoff data"
SubElement(metadata, "date_generated").text = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)
SubElement(metadata, "history").text = metadata_info

# Define the stream info names and corresponding var names
stream_info_data = [
    ("JRA55do.FRIVER", "friver", "Forr_rofl"),
    ("JRA55do.LICALVF", "licalvf", "Forr_rofi"),
]

# Generate stream info elements with changing years
for stream_name, var_prefix, var_suffix in stream_info_data:
    stream_info = SubElement(root, "stream_info", name=stream_name)
    if year_first == year_last:
        SubElement(stream_info, "taxmode").text = "cycle"
        SubElement(stream_info, "dtlimit").text = "1.0"
    else:
        SubElement(stream_info, "taxmode").text = "extend"
        SubElement(stream_info, "dtlimit").text = "1.e30"
    SubElement(stream_info, "tintalgo").text = "linear"
    SubElement(stream_info, "offset").text = "0"
    SubElement(stream_info, "readmode").text = "single"
    SubElement(stream_info, "mapalgo").text = "consd"
    SubElement(stream_info, "year_first").text = str(year_first)
    SubElement(stream_info, "year_last").text = str(year_last)
    SubElement(stream_info, "year_align").text = str(year_align)
    SubElement(stream_info, "vectors").text = "null"
    SubElement(stream_info, "meshfile").text = "./INPUT/JRA55do-drof-ESMFmesh.nc"
    SubElement(stream_info, "lev_dimname").text = "null"

    datafiles = SubElement(stream_info, "datafiles")
    datavars = SubElement(stream_info, "datavars")

    var_element = SubElement(datavars, "var")
    var_element.text = f"{var_prefix} {var_suffix}"

    for year in range(year_first, year_last + 1):
        if year_first == year_last:
            file_element = SubElement(datafiles, "file")
            file_element.text = f"./INPUT/RYF.{var_prefix}.{year+90}_{year + 90 + 1}.nc"
        else:
            if var_prefix == "friver":
                f_prefix = f"./INPUT/land/day/"
            elif var_prefix == "licalvf":
                f_prefix = f"./INPUT/landIce/day/"

            if source_data == "jra55v1p4":
                f_prefix += f"{var_prefix}/gr/v20190429/{var_prefix}_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-4-0_gr_"
            elif source_data == "jra55v1p6":
                f_prefix += f"{var_prefix}/gr/v20240531/{var_prefix}_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-6-0_gr_"

            file_element = SubElement(datafiles, "file")
            if source_data == "jra55v1p4" and year == 2019:
                file_element.text = f"{f_prefix}{year}0101-{year}0105.nc"
            elif source_data == "jra55v1p6" and year == 2024:
                file_element.text = f"{f_prefix}{year}0101-{year}0201.nc"
            else:
                file_element.text = f"{f_prefix}{year}0101-{year}1231.nc"


# Convert the XML to a nicely formatted string
xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")

# Write the XML content to a file
with open("drof.streams.xml", "w") as xml_file:
    xml_file.write(xml_str)
