#!/usr/bin/env python3
"""Compare how the Python bridge treats qvector arguments for built-in
operations, registered custom operations, and user kernels.

Usage: probe_bridge_consistency.py

This is a probe, not a pass/fail check: it prints what the installed cudaq
does for each call form. Its output is the evidence behind
NVIDIA/cuda-quantum#5646 and #5647; the "hermitian adj" rows show a separate
frontend/verifier mismatch for .adj on self-adjoint gates.

Each case runs in its own subprocess: on macOS arm64 a runtime count check
inside JIT code terminates the process instead of raising.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

PRELUDE = """
import cudaq
import numpy as np

SWAP = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=complex)
cudaq.register_operation("custom_x", np.array([0, 1, 1, 0], dtype=complex))
cudaq.register_operation("custom_swap", SWAP)

@cudaq.kernel
def two_qubit_kernel(a: cudaq.qubit, b: cudaq.qubit):
    swap(a, b)
"""

# (group, label, body, args). The kernel is `k`; body is its indented source.
CASES = [
    ("single-target", "x(q), qvector(3)", "q = cudaq.qvector(3)\n    x(q)", ""),
    ("single-target", "custom_x(q), qvector(1)",
     "q = cudaq.qvector(1)\n    custom_x(q)", ""),
    ("single-target", "custom_x(q), qvector(3)",
     "q = cudaq.qvector(3)\n    custom_x(q)", ""),
    ("single-target", "custom_x(q), qvector(n), n=3",
     "q = cudaq.qvector(n)\n    custom_x(q)", "3"),
    ("single-target", "t.adj(q), qvector(3)",
     "q = cudaq.qvector(3)\n    t.adj(q)", ""),
    ("single-target", "custom_x.adj(q), qvector(3)",
     "q = cudaq.qvector(3)\n    custom_x.adj(q)", ""),
    ("single-target", "x(a, b)",
     "a = cudaq.qubit()\n    b = cudaq.qubit()\n    x(a, b)", ""),
    ("single-target", "custom_x(a, b)",
     "a = cudaq.qubit()\n    b = cudaq.qubit()\n    custom_x(a, b)", ""),
    ("single-target", "custom_x.ctrl(c, t), c = qvector(2)",
     "c = cudaq.qvector(2)\n    t = cudaq.qubit()\n    x(c)\n    custom_x.ctrl(c, t)",
     ""),
    ("multi-target", "swap(q[0], q[1])",
     "q = cudaq.qvector(2)\n    x(q[0])\n    swap(q[0], q[1])", ""),
    ("multi-target", "swap(q), qvector(2)",
     "q = cudaq.qvector(2)\n    x(q[0])\n    swap(q)", ""),
    ("multi-target", "swap(*q), qvector(2)",
     "q = cudaq.qvector(2)\n    x(q[0])\n    swap(*q)", ""),
    ("multi-target", "custom_swap(q), qvector(2)",
     "q = cudaq.qvector(2)\n    x(q[0])\n    custom_swap(q)", ""),
    ("multi-target", "custom_swap(*q), qvector(2)",
     "q = cudaq.qvector(2)\n    x(q[0])\n    custom_swap(*q)", ""),
    ("multi-target", "custom_swap(q, r), qvector(1)",
     "q = cudaq.qvector(1)\n    r = cudaq.qubit()\n    x(q[0])\n    custom_swap(q, r)",
     ""),
    ("user kernel", "two_qubit_kernel(q), qvector(2)",
     "q = cudaq.qvector(2)\n    x(q[0])\n    two_qubit_kernel(q)", ""),
    ("user kernel", "two_qubit_kernel(*q), qvector(2)",
     "q = cudaq.qvector(2)\n    x(q[0])\n    two_qubit_kernel(*q)", ""),
    ("hermitian adj", "x.adj(q[0])", "q = cudaq.qvector(1)\n    x.adj(q[0])", ""),
    ("hermitian adj", "h.adj(q[0])", "q = cudaq.qvector(1)\n    h.adj(q[0])", ""),
    ("hermitian adj", "t.adj(q[0])", "q = cudaq.qvector(1)\n    t.adj(q[0])", ""),
]

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _most_specific(message):
    """Prefer an MLIR verifier diagnostic over the generic wrapper text."""
    for part in message.split(" | "):
        if "loc(" in part and "error: " in part:
            return part.split("error: ", 1)[1]
    return message.split(" | ")[0].split("error: ")[-1]


def run(body, args):
    sig = "n: int" if args else ""
    call = f"cudaq.sample(k, {args})" if args else "cudaq.sample(k)"
    src = (PRELUDE + f"\n@cudaq.kernel\ndef k({sig}):\n    {body}\n\n" +
           "try:\n"
           f"    print('RESULT', {call})\n"
           "except Exception as e:\n"
           "    print('ERROR', str(e).replace('\\n', ' | '))\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    p = subprocess.run([sys.executable, path],
                       capture_output=True,
                       text=True,
                       timeout=300)
    Path(path).unlink()
    out = ANSI.sub("", p.stdout)
    for line in out.splitlines():
        if line.startswith("RESULT "):
            return "ok   " + " ".join(line[7:].split())
        if line.startswith("ERROR "):
            return "error " + _most_specific(line[6:])[:90]
    err = ANSI.sub("", p.stderr).strip().splitlines()
    msg = next((l for l in reversed(err) if "terminating" in l or "rror" in l),
               f"exit {p.returncode}")
    return "abort " + msg.split("std::runtime_error: ")[-1][:90]


def main():
    group = None
    for g, label, body, args in CASES:
        if g != group:
            print(f"\n[{g}]")
            group = g
        print(f"  {label:<36} {run(body, args)}")


if __name__ == "__main__":
    main()
