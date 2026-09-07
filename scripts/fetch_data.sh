#!/usr/bin/env bash
# Download the IMMREP23 dataset into data/ (MIT licensed, ~15 MB unpacked).
set -euo pipefail
cd "$(dirname "$0")/../data"
curl -sL -o immrep23.zip https://github.com/justin-barton/IMMREP23/archive/refs/heads/main.zip
unzip -oq immrep23.zip
rm immrep23.zip
ls -la IMMREP23-main/data

# VDJdb, for the Phase B evaluation set. AGPL-3.0; cite Goncharov et al. 2022.
VDJDB_RELEASE=2026-06-03
curl -sL -o "vdjdb-${VDJDB_RELEASE}.zip" \
  "https://github.com/antigenomics/vdjdb-db/releases/download/${VDJDB_RELEASE}-ZENODO/vdjdb-${VDJDB_RELEASE}.zip"
unzip -oq "vdjdb-${VDJDB_RELEASE}.zip" \
  "vdjdb-${VDJDB_RELEASE}/vdjdb.slim.txt" \
  "vdjdb-${VDJDB_RELEASE}/vdjdb.slim.meta.txt" \
  "vdjdb-${VDJDB_RELEASE}/LICENSE" \
  "vdjdb-${VDJDB_RELEASE}/latest-version.txt"
rm "vdjdb-${VDJDB_RELEASE}.zip"
ls -la "vdjdb-${VDJDB_RELEASE}"
