#!/usr/bin/env bash
# Verify a change to the CUDA-Q Python bridge against a real cudaq runtime.
#
# Usage: verify_bridge_patch.sh <python-with-cudaq> <bridge.diff> <test.py>...
#
# cudaq/kernel/ast_bridge.py is pure Python, so a diff against it can be
# applied onto an installed wheel without building CUDA-Q. Each test file is
# run before and after, every test in its own process (see run_isolated.sh).
# The installed ast_bridge.py is restored on exit, whatever happens.
set -u
py="$1"; diff="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2
here="$(cd "$(dirname "$0")" && pwd)"
kdir="$("$py" -c 'import cudaq, os; print(os.path.dirname(cudaq.__file__))' 2>/dev/null)/kernel"
orig="$(mktemp)"
cp "$kdir/ast_bridge.py" "$orig"
restore() {
  cp "$orig" "$kdir/ast_bridge.py"
  find "$kdir" -name 'ast_bridge*.pyc' -delete
  rm -f "$orig"
}
trap restore EXIT

run_all() {
  for t in "$@"; do
    echo "-- $(basename "$t")"
    bash "$here/run_isolated.sh" "$py" "$t"
  done
}

echo "== before: cudaq as installed"
run_all "$@"

patch -s -d "$kdir" -p4 < "$diff" || { echo "patch did not apply"; exit 1; }
find "$kdir" -name 'ast_bridge*.pyc' -delete

echo "== after: with the patch"
run_all "$@"
