#!/usr/bin/bash
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

python3 $(dirname "$0")/archive_scripts/standardise_mom_filenames.py
python3 $(dirname "$0")/archive_scripts/archive_esmf_streams.py
