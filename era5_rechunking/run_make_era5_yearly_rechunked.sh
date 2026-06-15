#!/usr/bin/bash
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

#PBS -N era5_yearly_rechunk
#PBS -P tm70
#PBS -q normal
#PBS -l ncpus=24
#PBS -l mem=96GB
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/tm70+gdata/xp65+gdata/rt52
#PBS -l wd
#PBS -j oe
#PBS -o /g/data/tm70/ek4684/era5_rechunked_1h/logs/yearly/

set -euo pipefail

SCRIPT_DIR="${PBS_O_WORKDIR}"

module use /g/data/xp65/public/modules
module load conda/analysis3-26.02

python3 "${SCRIPT_DIR}/make_era5_yearly_rechunked.py" --workers "${PBS_NCPUS:-24}"
