"""
ESMF Profiling tool
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
class ESMFRegion(object):
    def __init__(self, collect_data, esmf_summary=True):
        if not esmf_summary:
            self.name = collect_data[0]
            self.count = collect_data[1]
            self.total = collect_data[2]
            self.self_time = collect_data[3]
            self.mean = collect_data[4]
            self.min_time = collect_data[5]
            self.max_time = collect_data[6]
        else:
            self.name = collect_data[0]
            self.count = collect_data[1]
            self.PETs = collect_data[2]
            self.PEs = collect_data[3]
            self.mean = collect_data[4]
            self.min_time = collect_data[5]
            self.max_time = collect_data[6]
            self.min_PET = collect_data[7]
            self.max_PET = collect_data[8]

        self.children = []
        self.esmf_summary = esmf_summary

    def add_child(self, child):
        self.children.append(child)

    def to_dict(self):
        if not self.esmf_summary:
            profile_dict = {
                "name": self.name,
                "count": self.count,
                "total": self.total,
                "self_time": self.self_time,
                "mean": self.mean,
                "min_time": self.min_time,
                "max_time": self.max_time,
                "children": [child.to_dict() for child in self.children],
            }
        else:
            profile_dict = {
                "name": self.name,
                "count": self.count,
                "PETs": self.PETs,
                "PEs": self.PEs,
                "mean": self.mean,
                "min_time": self.min_time,
                "max_time": self.max_time,
                "min_PET": self.min_PET,
                "max_PET": self.max_PET,
                "children": [child.to_dict() for child in self.children],
            }
        return profile_dict
