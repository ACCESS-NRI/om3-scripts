#!/usr/bin/env python3
"""
ESMF Profiling tool - OM3
The ESMF Profiling tool is a Python-based tool designed to read and process 
performance profile data from ESMF profiling log files. It provides a 
structured way to extract hierachical timing and computational stats for 
various regions within ESMF runs, enabling detailed performance analysis.

 1. esmfFileParser.py: 
    - handles the input ESMF profile files.
    - constructs the hierarchical data structure.
    - outputs runtimes for specific regions.
 2. esmfRegion.py: 
    - defines the ESMFRegion class.
    - represents individual ESMF regions, capturing performance metrics.

The tool supports two profiling data formats:
 1. ESMF_Profile.xxxx:    (0. count, 1. total, 2. self_time, 3. mean    , 4. min_time, 5. max_time           )
 2. ESMF_Profile.summary: (0. count, 1. PETs , 2. mean     , 3. min_time, 4. max_time, 5. min_PET, 6. max_PET)

One example for demonstration:
----------
# Collect runtime info for specific regions
ESMF_path = ['/path/to/your/ESMF/output/files']
region_names = ['[ESMF]', '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[OCN] RunPhase1']
profile_prefix = 'ESMF_Profile.'
esmf_summary = True  # Set to True for summary profiling
index = 2  # Choose the metric to extract (e.g., mean time for summary profiling)

runtime_totals = collect_runtime_tot(ESMF_path, regionNames=region_names, profile_prefix=profile_prefix, esmf_summary=esmf_summary, index=index)
print(runtime_totals)

Latest version: xxx
Author: Minghang Li
Email: minghang.li1@anu.edu.au
License: Apache 2.0 License http://www.apache.org/licenses/LICENSE-2.0.txt
"""


# ===========================================================================
import os
import re
import warnings
from esmfRegion import ESMFRegion
import time


def collect_runtime_tot(
    ESMF_path,
    regionNames=["[ESMF]"],
    profile_prefix="ESMF_Profile.",
    esmf_summary=True,
    index=2,
):

    runtime_tot = []
    for i in range(len(ESMF_path)):
        subfiles_path = _list_esmf_files(
            ESMF_path[i],
            profile_prefix,
            esmf_summary,
            summary_profile="ESMF_Profile.summary",
        )
        esmf_region_all = []
        for subfile in subfiles_path:
            with open(subfile, "r") as f:
                esmf_region = ESMFProfileTrees(esmf_summary).build_ESMF_trees(f)
                esmf_region_all.append(esmf_region)

        runtime = _region_time_consumption(
            regionNames, esmf_region_all, index, esmf_summary
        )
        runtime_tot.append(runtime)
    return runtime_tot


def _list_esmf_files(dir_path, profile_prefix, esmf_summary, summary_profile):
    """Lists ESMF files based on a prefix."""
    try:
        files = os.listdir(dir_path)
        if not esmf_summary:
            # ESMF_Profile.xxxx
            matching_files = [
                file
                for file in files
                if file.startswith(profile_prefix) and file != summary_profile
            ]
            matching_files.sort(key=lambda x: int(x[len(profile_prefix) :]))
        else:
            # ESMF_Profile.summary
            matching_files = [summary_profile]

        matching_files_path = [
            os.path.join(dir_path, matching_file) for matching_file in matching_files
        ]
    except FileNotFoundError:
        warnings.warn(f"Directory {dir_path} does not exist! Skipping!")
        matching_files_path = []
    return matching_files_path


def _region_time_consumption(regionNames, esmf_region_all, index, esmf_summary):
    """Calculates time consumption for specific regions"""
    runtime = {}

    for regionName in regionNames:
        region_values = [
            _find_region_values(sub_ESMF_region, regionName, esmf_summary)
            for sub_ESMF_region in esmf_region_all
        ]

        # Process each region value
        for region_value in region_values:
            for key, val in region_value.items():
                if val and val[0] is not None:
                    if key not in runtime:
                        runtime[key] = []
                    runtime[key].append(val[index])
    return runtime


def _find_region_values(region, target_region, esmf_summary):
    """
    Searches for all region values based on a hierarchical path prefix.
    """
    target_parts = target_region.split("/")
    matches = {}
    default_nans = (None,) * 6

    def search_region(region, path, current_path):
        """Recursive helper function to search regions."""
        if region.name.startswith(path[0]):
            if len(path) == 1:
                matches[current_path] = _get_region_data(region, esmf_summary)
                return

            # Continue searching in children if more path components remain
            for child in region.children:
                search_region(child, path[1:], f"{current_path}/{child.name}")

    def _get_region_data(region, esmf_summary):
        """Returns the region data tuple based on esmf_summary setting."""
        return (
            (
                region.count,
                region.total,
                region.self_time,
                region.mean,
                region.min_time,
                region.max_time,
            )
            if not esmf_summary
            else (
                region.count,
                region.PETs,
                region.mean,
                region.min_time,
                region.max_time,
                region.min_PET,
                region.max_PET,
            )
        )

    # recursive search
    search_region(region, target_parts, region.name)

    if not matches:
        print(f"No matches found for target prefix: {target_region}")
    return matches if matches else {f"{target_region}": default_nans}


class ESMFProfileTrees(object):
    def __init__(self, esmf_summary=True):
        self.esmf_summary = esmf_summary

    def build_ESMF_trees(self, lines):
        """Builds a hierarchical tree of ESMF regions."""
        stack = []
        esmf_region = None
        skip = False
        for line in lines:
            if not line.strip():
                continue
            if line.startswith("  [ESMF]"):
                skip = True
            if skip:
                indent_level = len(line) - len(line.lstrip())
                collect_data = self._parse_line(line)
                region = ESMFRegion(collect_data, self.esmf_summary)
                if not stack:
                    # the first esmf_region
                    esmf_region = region
                else:
                    while stack and stack[-1][1] >= indent_level:
                        stack.pop()
                    if stack:
                        stack[-1][0].add_child(region)
                stack.append((region, indent_level))
        return esmf_region

    def _parse_line(self, line):
        """Parses a line from an ESMF file based on summary flag."""
        line = re.sub(r"(\d+)(\s+)(\d+\.\d+)", r"\1  \3", line)
        parts = re.split(r"\s{2,}", line.strip())

        if not self.esmf_summary:
            name = parts[0]
            count = int(parts[1])
            total = float(parts[2])
            self_time = float(parts[3])
            mean = float(parts[4])
            min_time = float(parts[5])
            max_time = float(parts[6])
            collect_data = (name, count, total, self_time, mean, min_time, max_time)
        else:
            name = parts[0]
            PETs = int(parts[1])
            PEs = int(parts[2])
            count = int(parts[3])
            mean = float(parts[4])
            min_time = float(parts[5])
            min_PET = int(parts[6])
            max_time = float(parts[7])
            max_PET = int(parts[8])
            collect_data = (
                name,
                count,
                PETs,
                PEs,
                mean,
                min_time,
                max_time,
                min_PET,
                max_PET,
            )
        return collect_data


if __name__ == "__main__":
    # index for different profiling files (ESMF_Profile.xxxx or ESMF_Profile.summary)
    # ESMF_Profile.xxxx   : (0. count, 1. total, 2. self_time, 3. mean    , 4. min_time, 5. max_time           )
    # ESMF_Profile.summary: (0. count, 1. PETs , 2. mean     , 3. min_time, 4. max_time, 5. min_PET, 6. max_PET)
    # It takes ~20 seconds to read and output runtimes for 1440 ESMF_Profile.xxxx files,
    # while reading the ESMF_Profile.summary file only takes about 0.02 seconds.
    time_start = time.time()
    ESMF_path = ["/g/data/tm70/ml0072/ESMF_Profile_test_example"]
    runtime_tot = collect_runtime_tot(
        ESMF_path,
        regionNames=[
            "[ESMF]",
            "[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[OCN] RunPhase1",
            "[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED]",
        ],
        profile_prefix="ESMF_Profile.",
        esmf_summary=True,
        index=2,
    )
    time_end = time.time()
    elaps = time_end - time_start
    print(elaps)
    print(runtime_tot)

    # The above example presents three items in `regionNames` variable,
    # where "[ESMF]" and "[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[OCN] RunPhase1" are the exact region names in ESMF profile files,
    # while "[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED]" is a prefix for:
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_aofluxes_run',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_diag_accum',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_diag_atm',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_diag_ice_ice2med',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_diag_ice_med2ice',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_diag_ocn',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_diag_print',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_diag_rof',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_history_write',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_ocnalb_run',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_post_atm',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_post_ice',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_post_ocn',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_post_rof',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_prep_ice',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_prep_ocn_accum',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_prep_ocn_avg',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_profile',
    # '[ESMF]/[ensemble] RunPhase1/[ESM0001] RunPhase1/[MED] med_phases_restart_write'
    ## Hence, this tool supports runtime collection for both exact match and prefix-based match.
