#!/usr/bin/bash
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBE -l jobfs=100GB
#PBS -q expresssr
#PBS -l walltime=00:30:00
#PBS -l wd
#PBS -v PYTHONNOUSERSITE=True

module purge
module use /g/data/xp65/public/modules 
module load conda/analysis3-25.05

python3 $SCRIPTS_DIR/payu_config/postscript/build_intake_ds.py
