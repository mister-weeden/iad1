#!/usr/bin/env bash
set -euo pipefail
root="/lustre/scratch/rsna_ihd"
mkdir -p "$root"/weights "$root"/preds "$root"/logs
echo "RSNA IHD syncpoints created under $root"

