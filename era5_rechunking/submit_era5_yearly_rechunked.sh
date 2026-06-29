#!/usr/bin/bash
# Copyright 2026 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

#PBS -N era5_yearly_rechunk
#PBS -P tm70
#PBS -q normal
#PBS -l ncpus=24
#PBS -l mem=190GB
#PBS -l walltime=48:00:00
#PBS -l storage=gdata/tm70+gdata/xp65+gdata/rt52
#PBS -l wd
#PBS -j oe

set -euo pipefail

# ── user configuration ─────────────────────────────────────────────────────────
# Directory where yearly rechunked files will be written:
#   {OUTPUT_DIR}/{stream}/{stream}_era5_oper_sfc_{YYYYMMDD}-{YYYYMMDD}.nc
OUTPUT_DIR="/scratch/${PROJECT}/${USER}/era5_rechunked_1h_yearly"

# Each worker can use several GB while decompressing and rechunking full-spatial
# ERA5 fields. On this 24 CPU / 96 GB job, 12 workers can hit the memory limit.
# Use 12 only with a larger node/allocation that has been tested.
WORKERS="${WORKERS:-6}"
# ──────────────────────────────────────────────────────────────────────────────

module use /g/data/xp65/public/modules
module load conda/analysis3-26.02

python3 "${PBS_O_WORKDIR}/make_era5_yearly_rechunked.py" \
    --output-dir "${OUTPUT_DIR}"                       \
    --workers    "${WORKERS}"
