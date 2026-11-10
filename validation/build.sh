#!/bin/bash
# Build the three Fortran drivers against a HYSPLIT library you built yourself.
#
#   HYSPLIT_LIB=/path/to/hysplit/library ./validation/build.sh
#
# HYSPLIT_LIB holds libhysplit.a and hysplit/ (the .mod and .inc files); both come from
# `make` in HYSPLIT's library/ folder. Write down the HYSPLIT version the library came from
# (see README.md, "HYSPLIT version"). The binaries go to validation/build/ (not tracked):
# the three drivers and HYSPLIT's own xtrct_grid, built from source/xtrct_grid.f beside
# library/ (override with HYSPLIT_SOURCE), which tests/test_subset_slow.py compares
# extract_subset with.
set -euo pipefail
: "${HYSPLIT_LIB:?set HYSPLIT_LIB to the library folder of your HYSPLIT build}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/build"
for name in hyref hyrun hyorder; do
  gfortran -O0 -ffree-form -fconvert=big-endian -I"$HYSPLIT_LIB/hysplit" \
    -o "$here/build/$name" "$here/fortran/$name.f90" "$HYSPLIT_LIB/libhysplit.a"
done
src="${HYSPLIT_SOURCE:-$HYSPLIT_LIB/../source}"
gfortran -O0 -ffree-form -fallow-argument-mismatch -fconvert=big-endian -I"$HYSPLIT_LIB/hysplit" \
  -o "$here/build/xtrct_grid" "$src/xtrct_grid.f" "$HYSPLIT_LIB/libhysplit.a"
echo "built $here/build/{hyref,hyrun,hyorder,xtrct_grid}"
