#!/bin/bash
# Run compare.py on every file in files.tsv (or only the ones given) and write results.
#
#   DATA_DIR=/folder/with/the/met/files ./validation/sampling/run.sh OUT_DIR [hysplit-values-dir]
#
# DATA_DIR holds the files of files.tsv under those names. OUT_DIR gets one JSON per file.
# With a second argument, HYSPLIT's values are read from that folder when a file is there
# (./validation/sampling/hysplit holds the stored ones, so nothing needs building) and
# otherwise the Fortran drivers run (build.sh first) and write it.
# Run it in a SLURM job, not on a login node. PYTHON defaults to `python`.
set -euo pipefail
: "${DATA_DIR:?set DATA_DIR to the folder holding the met files}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -x "$here/../build/hyorder" ] || [ -n "${2:-}" ] || { echo "build the drivers first (build.sh)" >&2; exit 1; }
out="${1:?usage: run.sh OUT_DIR [hysplit-values-dir]}"
hy="${2:-}"
py="${PYTHON:-python}"
mkdir -p "$out"; [ -z "$hy" ] || mkdir -p "$hy"
while IFS=$'\t' read -r prod name; do
  extra=()
  [ -z "$hy" ] || extra=(--hysplit "$hy/${prod}__${name}.npz")
  "$py" -I "$here/compare.py" "$DATA_DIR/$name" "$out/${prod}__${name}.json" "${extra[@]}" \
    > "$out/${prod}__${name}.log" 2>&1 || echo "FAILED $name (see $out/${prod}__${name}.log)"
done < "$here/files.tsv"
echo done
