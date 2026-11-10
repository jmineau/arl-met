#!/bin/bash
# Run compare.py on every file in files.tsv. Needs the drivers (build.sh).
#
#   DATA_DIR=/folder/with/the/met/files ./validation/files/run.sh OUT_DIR
#
# Run it in a SLURM job (30 files took 9 minutes on 4 cores). PYTHON defaults to `python`.
set -euo pipefail
: "${DATA_DIR:?set DATA_DIR to the folder holding the met files}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -x "$here/../build/hyref" ] || { echo "build the drivers first (build.sh)" >&2; exit 1; }
out="${1:?usage: run.sh OUT_DIR}"
py="${PYTHON:-python}"
mkdir -p "$out"
while IFS=$'\t' read -r prod name; do
  "$py" -I "$here/compare.py" "$DATA_DIR/$name" "$out/${prod}__${name}.json" --columns \
    > "$out/${prod}__${name}.log" 2>&1 || echo "FAILED $name (see $out/${prod}__${name}.log)"
done < "$here/files.tsv"
echo done
