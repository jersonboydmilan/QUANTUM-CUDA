#!/usr/bin/env bash
# Verify NVIDIA/cuda-quantum#5649 against a real cudaq runtime.
#
# Usage: verify_pr5649.sh <python-with-cudaq> <test_custom_operations.py>
#
# The Python bridge (cudaq/kernel/ast_bridge.py) is pure Python, so the PR's
# bridge change can be applied onto an installed wheel without building
# CUDA-Q. patches/0004-pr5649-bridge-on-0.16.0.diff is the PR's bridge hunk
# with one context line re-anchored to the 0.16.0 source; its added and
# removed lines are byte-identical to the PR. The test file is the PR's own
# (fetch it from the PR head). The installed ast_bridge.py is restored on
# exit, whatever happens.
set -u
py="$1"; tests="$2"
here="$(cd "$(dirname "$0")" && pwd)"
diff="$here/../patches/0004-pr5649-bridge-on-0.16.0.diff"
kdir="$("$py" -c 'import cudaq, os; print(os.path.dirname(cudaq.__file__))' 2>/dev/null)/kernel"
orig="$(mktemp)"
cp "$kdir/ast_bridge.py" "$orig"
restore() {
  cp "$orig" "$kdir/ast_bridge.py"
  find "$kdir" -name 'ast_bridge*.pyc' -delete
  rm -f "$orig"
}
trap restore EXIT

skip="test_unknown_veq_size_incorrect_count"  # upstream skips it on macOS arm64

echo "== before: cudaq as installed"
bash "$here/run_isolated.sh" "$py" "$tests" $skip
"$py" "$here/check_adj_broadcast_phase.py" 2>/dev/null

patch -s -d "$kdir" -p4 < "$diff" || { echo "patch did not apply"; exit 1; }
find "$kdir" -name 'ast_bridge*.pyc' -delete

echo "== after: with the #5649 bridge change"
bash "$here/run_isolated.sh" "$py" "$tests" $skip
"$py" "$here/check_adj_broadcast_phase.py" 2>/dev/null
