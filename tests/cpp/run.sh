#!/usr/bin/env bash
# Runs canopy/inference's pure C++ unit tests (header-only, no ORT, no device):
#   tilecover-test  — RestoreTiling.h: the fixed-window tiler's central spans partition the padded
#                     length exactly (no holes or seams in the stitched output);
#   colorconv-test  — RestoreColor.h: the sRGB <-> CIELAB round trip of the colorize path.
# Needs a C++17 compiler (CXX, default c++).   tests/cpp/run.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cxx="${CXX:-c++}"
out="$(mktemp -d)"; trap 'rm -rf "$out"' EXIT
status=0
for t in tilecover colorconv; do
  if "$cxx" -std=c++17 -Wall -I "$here/native/cpp" "$here/tests/cpp/$t-test.cpp" -o "$out/$t" 2>"$out/$t.log"; then
    "$out/$t" || status=1
  else
    echo "FAIL — $t-test did not compile"; cat "$out/$t.log"; status=1
  fi
done
exit "$status"
