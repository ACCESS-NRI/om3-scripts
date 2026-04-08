#!/usr/bin/env python3
# Copyright 2023 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

# Generate a datm xml file that contains a time-series of input atmosphere data files where all the fields in the stream are located.

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

# in jra55do, these fields (fluxes) have timestamps in the middle of the average interval
STREAMS_AVE = [
    "JRA55do.PRSN",
    "JRA55do.PRRN",
    "JRA55do.LWDN",
    "JRA55do.SWDN",
]

# in jra55do, these fields (tracers) have timestamps at the time of the point measurement
STREAMS_PT = [
    "JRA55do.Q_10",
    "JRA55do.SLP_10",
    "JRA55do.T_10",
    "JRA55do.U_10",
    "JRA55do.V_10",
]

source_data = "jra55v1p6"
# source_data = "jra55v1p4"

if len(sys.argv) != 3:
    print("Usage: python generate_xml_datm.py year_first year_last")
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

# Define the stream info names and corresponding var names
stream_info_names = [*STREAMS_AVE, *STREAMS_PT]

var_names = {
    "JRA55do.PRSN": ("prsn", "Faxa_prsn"),
    "JRA55do.PRRN": ("prra", "Faxa_prrn"),
    "JRA55do.LWDN": ("rlds", "Faxa_lwdn"),
    "JRA55do.SWDN": ("rsds", "Faxa_swdn"),
    "JRA55do.Q_10": ("huss", "Sa_shum"),
    "JRA55do.SLP_10": ("psl", "Sa_pslv"),
    "JRA55do.T_10": ("tas", "Sa_tbot"),
    "JRA55do.U_10": ("uas", "Sa_u"),
    "JRA55do.V_10": ("vas", "Sa_v"),
}

# Generate stream info elements with changing years
for stream_name in stream_info_names:
    stream_info = SubElement(root, "stream_info", name=stream_name)
    if year_first == year_last:
        SubElement(stream_info, "taxmode").text = "cycle"
        SubElement(stream_info, "dtlimit").text = "1.0"
    else:
        if stream_name in STREAMS_AVE:
            # first measurement is at 1:30am, however experiments start at midnight, so allow extension
            SubElement(stream_info, "taxmode").text = "extend"
            SubElement(stream_info, "dtlimit").text = "1.e30"
        else:
            SubElement(stream_info, "taxmode").text = "limit"
            SubElement(stream_info, "dtlimit").text = "1.0"

    SubElement(stream_info, "readmode").text = "single"
    SubElement(stream_info, "mapalgo").text = "bilinear"
    SubElement(stream_info, "year_first").text = str(year_first)
    SubElement(stream_info, "year_last").text = str(year_last)
    SubElement(stream_info, "year_align").text = str(year_align)
    SubElement(stream_info, "vectors").text = "null"
    SubElement(stream_info, "meshfile").text = "./INPUT/JRA55do-datm-ESMFmesh.nc"
    SubElement(stream_info, "lev_dimname").text = "null"

    datafiles = SubElement(stream_info, "datafiles")
    datavars = SubElement(stream_info, "datavars")

    # for linear/nearest interpolation, timestamps need to be middle of time interval
    # for coszen, timestamps need to be start of time interval
    # see https://github.com/ACCESS-NRI/CDEPS/blob/2733fdcfaece8eb53798f5fe19bf91137744f21c/streams/dshr_tinterp_mod.F90#L36
    if stream_name == "JRA55do.SWDN":
        SubElement(stream_info, "offset").text = "-5400"
        SubElement(stream_info, "tintalgo").text = "coszen"
    else:
        SubElement(stream_info, "offset").text = "0"
        SubElement(stream_info, "tintalgo").text = "linear"

    var_name_parts = var_names.get(
        stream_name,
        (
            stream_name.split(".")[-1].lower(),
            f"Faxa_{stream_name.split('.')[-1].lower()}",
        ),
    )
    var_element = SubElement(datavars, "var")
    var_element.text = f"{var_name_parts[0]}  {var_name_parts[1]}"

    for year in range(year_first, year_last + 1):
        if year_first == year_last:
            file_element = SubElement(datafiles, "file")
            file_element.text = (
                f"./INPUT/RYF.{var_name_parts[0]}.{year+90}_{year + 90 + 1}.nc"
            )
        else:
            file_element = SubElement(datafiles, "file")

            if source_data == "jra55v1p4":
                file = f"{var_name_parts[0]}/gr/v20190429/{var_name_parts[0]}_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-4-0_gr_"
            elif source_data == "jra55v1p6":
                file = f"{var_name_parts[0]}/gr/v20240531/{var_name_parts[0]}_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-6-0_gr_"

            # figure out the final month of data
            if source_data == "jra55v1p4" and year == 2019:
                fin_md = "0105"  # aka 1 May 2019 (sic)
            elif source_data == "jra55v1p6" and year == 2024:
                fin_md = "0201"  # aka 1 Feb 2024
            else:
                fin_md = "1231"  # aka 31 Dec

            # account for point or average datestamps in filename
            if stream_name in STREAMS_AVE:
                file_element.text = (
                    f"./INPUT/atmos/3hr/{file}{year}01010130-{year}{fin_md}2230.nc"
                )
            else:
                file_element.text = (
                    f"./INPUT/atmos/3hrPt/{file}{year}01010000-{year}{fin_md}2100.nc"
                )


# Convert the XML to a nicely formatted string
xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")

# Write the XML content to a file
with open("datm.streams.xml", "w") as xml_file:
    xml_file.write(xml_str)
