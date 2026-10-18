#!/usr/bin/env bash
# Run each test of a pytest file in its own process, so a JIT-side abort
# (macOS arm64) is reported as one failed test instead of killing the run.
# Usage: run_isolated.sh <python> <test_file> [deselect_id ...]
set -u
py="$1"; file="$2"; shift 2
skip=" $* "
# pytest prints IDs relative to the file's directory; run from there.
cd "$(dirname "$file")" || exit 1
file="$(basename "$file")"
ids=$("$py" -m pytest "$file" --collect-only -q -p no:cacheprovider -W ignore 2>/dev/null | grep '::')
pass=0; fail=0; skipped=0
for id in $ids; do
  name="${id#*::}"
  if [[ "$skip" == *" $name "* ]]; then
    printf "  %-62s SKIP (macOS arm64 JIT)\n" "$name"; skipped=$((skipped+1)); continue
  fi
  if out=$("$py" -m pytest "$id" -q -rs -p no:cacheprovider -W ignore 2>&1); then
    if grep -q " skipped" <<<"$out"; then
      reason=$(grep -m1 "^SKIPPED" <<<"$out" | sed 's/.*: //')
      printf "  %-62s SKIP (%s)\n" "$name" "${reason:-skipped}"; skipped=$((skipped+1))
    else
      printf "  %-62s PASS\n" "$name"; pass=$((pass+1))
    fi
  else
    rc=$?
    note="FAIL"; [ "$rc" -ge 128 ] && note="FAIL (process aborted)"
    printf "  %-62s %s\n" "$name" "$note"; fail=$((fail+1))
  fi
done
echo "  -- $pass passed, $fail failed, $skipped skipped"
