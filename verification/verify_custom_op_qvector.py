#!/usr/bin/env python3
"""Execute every Python claim in the "Passing a qvector to a custom operation"
note against the installed cudaq.

Usage: verify_custom_op_qvector.py

Each case is a small kernel written to a temporary file and run in its own
subprocess, so a crash in one case cannot hide the result of another.
`custom_swap` is a 2-target operation; `custom_x` is a 1-target operation.
Starting from |10>, a SWAP gives |01>.
"""

import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

PRELUDE = """
import cudaq
import numpy as np

cudaq.register_operation("custom_swap",
                         np.array([1, 0, 0, 0,
                                   0, 0, 1, 0,
                                   0, 1, 0, 0,
                                   0, 0, 0, 1], dtype=complex))
cudaq.register_operation("custom_x", np.array([0, 1, 1, 0], dtype=complex))
"""

# (label, kernel source, call, expected outcome)
# expected: a bitstring that must be the most probable result, or
#           "reject:<substring>" for a kernel cudaq must refuse with an error
#           containing <substring>. Count checks that can only run once the
#           kernel executes are raised from JIT-compiled code; on macOS arm64
#           that terminates the process instead of raising (upstream skips
#           test_unknown_veq_size_incorrect_count there for the same reason),
#           so a termination carrying the same message also counts as refused.
CASES = [
    ("individual qubits:  custom_swap(q[0], q[1])", """
@cudaq.kernel
def k():
    q = cudaq.qvector(2)
    x(q[0])
    custom_swap(q[0], q[1])
""", "cudaq.sample(k)", "01"),
    ("whole qvector:      custom_swap(q)", """
@cudaq.kernel
def k():
    q = cudaq.qvector(2)
    x(q[0])
    custom_swap(q)
""", "cudaq.sample(k)", "01"),
    ("unpacked qvector:   custom_swap(*q)", """
@cudaq.kernel
def k():
    q = cudaq.qvector(2)
    x(q[0])
    custom_swap(*q)
""", "cudaq.sample(k)", "01"),
    ("mixed:              custom_swap(q, r)", """
@cudaq.kernel
def k():
    q = cudaq.qvector(1)
    r = cudaq.qubit()
    x(q[0])
    custom_swap(q, r)
""", "cudaq.sample(k)", "01"),
    ("too few qubits:     custom_swap(r)", """
@cudaq.kernel
def k():
    r = cudaq.qubit()
    custom_swap(r)
""", "cudaq.sample(k)",
     "reject:requires 2 qubit target(s), but 1 were provided"),
    ("runtime-sized, n=2: custom_swap(q)", """
@cudaq.kernel
def k(n: int):
    q = cudaq.qvector(n)
    x(q[0])
    custom_swap(q)
""", "cudaq.sample(k, 2)", "01"),
    ("1-target op on qvector(2): custom_x(q)", """
@cudaq.kernel
def k():
    q = cudaq.qvector(2)
    custom_x(q)
""", "cudaq.sample(k)",
     "reject:requires 1 qubit target(s), but 2 were provided"),
]

ANSI = re.compile(r"\x1b\[[0-9;]*m")

HARNESS = """
try:
    print("RESULT", {call}.most_probable())
except RuntimeError as e:
    print("RUNTIME_ERROR", str(e).replace("\\n", " "))
"""


def run_case(kernel, call):
    src = PRELUDE + textwrap.dedent(kernel) + HARNESS.format(call=call)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    p = subprocess.run([sys.executable, path],
                       capture_output=True,
                       text=True,
                       timeout=300)
    Path(path).unlink()
    stdout = ANSI.sub("", p.stdout)
    for line in stdout.splitlines():
        if "RESULT " in line:
            return "result", line.split("RESULT ", 1)[1]
        if "RUNTIME_ERROR " in line:
            return "error", line.split("RUNTIME_ERROR ", 1)[1]
    return "crash", f"exit {p.returncode}: " + ANSI.sub("", p.stderr)


def main():
    import cudaq
    version = re.search(r"Version (\S+)", cudaq.__version__)
    print(f"cudaq {version.group(1) if version else cudaq.__version__}\n")
    passed = True
    for label, kernel, call, expected in CASES:
        kind, detail = run_case(kernel, call)
        if expected.startswith("reject:"):
            ok = kind in ("error", "crash") and expected[7:] in detail
            shown = {"error": "RuntimeError", "crash": "terminated"}.get(
                kind, detail)
            shown = f"refused ({shown})" if ok else shown
        else:
            ok = kind == "result" and detail == expected
            shown = detail
        passed &= ok
        print(f"{label:<42} -> {shown:<14} {ok}")
    print("\nAll checks passed." if passed else "\nChecks FAILED.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
