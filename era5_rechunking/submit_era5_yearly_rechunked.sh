#!/usr/bin/bash
# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
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

set -euo pipefail

# ── user configuration ─────────────────────────────────────────────────────────
# Directory where yearly rechunked files will be written:
#   {OUTPUT_DIR}/{stream}/{stream}_era5_oper_sfc_{YYYYMMDD}-{YYYYMMDD}.nc
OUTPUT_DIR="/scratch/${PROJECT}/${USER}/era5_rechunked_1h_yearly"
# ──────────────────────────────────────────────────────────────────────────────

module use /g/data/xp65/public/modules
module load conda/analysis3-26.02

python3 "${PBS_O_WORKDIR}/make_era5_yearly_rechunked.py" \
    --output-dir "${OUTPUT_DIR}"                       \
    --workers    "${PBS_NCPUS:-24}"
