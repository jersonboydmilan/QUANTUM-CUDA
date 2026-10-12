#!/usr/bin/env python3
"""Check the custom Kraus channel example under `cudaq.apply_noise` in
docs/sphinx/api/languages/python_api.rst.

Usage: verify_apply_noise_example.py <cuda-quantum checkout>

The block is extracted from the .rst file exactly as Sphinx renders it, then:
  1. checked structurally (is __init__ a method? does registration run?),
  2. compared against the CustomNoiseChannel that test_NoiseModel.py exercises,
  3. executed against the installed cudaq, each step in its own subprocess
     because the documented-as-is block aborts the interpreter.
"""

import ast
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

RST = "docs/sphinx/api/languages/python_api.rst"
TEST = "python/tests/builder/test_NoiseModel.py"


def extract_block(rst_text):
    lines = rst_text.splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.strip().startswith(".. function:: cudaq.apply_noise"))
    cb = next(i for i in range(start, len(lines))
              if lines[i].strip() == ".. code-block:: python")
    indent = len(lines[cb]) - len(lines[cb].lstrip())
    body = []
    for l in lines[cb + 1:]:
        if l.strip() and len(l) - len(l.lstrip()) <= indent:
            break
        body.append(l)
    return textwrap.dedent("\n".join(body)).strip("\n") + "\n"


def find_class(tree, name):
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == name)


def structural_checks(block, test_src):
    tree = ast.parse(block)
    cls = find_class(tree, "CustomNoiseChannel")
    has_init = any(
        isinstance(n, ast.FunctionDef) and n.name == "__init__"
        for n in cls.body)
    top_level_calls = [
        ast.unparse(n.value) for n in tree.body if isinstance(n, ast.Expr)
    ]
    registers = any("register_channel" in c for c in top_level_calls)
    tested = find_class(ast.parse(test_src), "CustomNoiseChannel")
    same_as_tested = ast.dump(cls) == ast.dump(tested)
    return has_init, registers, same_as_tested


CONSTRUCT = """
c = CustomNoiseChannel([0.1])
ops = [np.array(k) for k in c.get_ops()]
total = sum(k.conj().T @ k for k in ops)
print("KRAUS_COMPLETE", np.allclose(total, np.eye(2)))
"""

APPLY = """
if "noise" not in globals():
    noise = cudaq.NoiseModel()
    noise.register_channel(CustomNoiseChannel)

cudaq.set_target("density-matrix-cpu")

@cudaq.kernel
def flip():
    q = cudaq.qubit()
    x(q)
    cudaq.apply_noise(CustomNoiseChannel, 0.1, q)

counts = cudaq.sample(flip, shots_count=20000, noise_model=noise)
print("P0", counts.probability("0"))
"""

PRELUDE = "import cudaq\nimport numpy as np\n"


def run(block, harness):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(PRELUDE + block + harness)
        path = f.name
    p = subprocess.run([sys.executable, path],
                       capture_output=True,
                       text=True,
                       timeout=300)
    Path(path).unlink()
    out = p.stdout + p.stderr
    return p.returncode, out


def runtime_checks(block):
    rc, out = run(block, CONSTRUCT)
    construct_ok = rc == 0 and "KRAUS_COMPLETE True" in out
    construct_err = None if construct_ok else _last_error(out, rc)

    rc, out = run(block, APPLY)
    p0 = None
    for line in out.splitlines():
        if line.startswith("P0 "):
            p0 = float(line.split()[1])
    apply_err = None if p0 is not None else _last_error(out, rc)
    return construct_ok, construct_err, p0, apply_err


def _last_error(out, rc):
    for line in reversed(out.splitlines()):
        if "Error" in line or "terminating" in line:
            return f"{line.strip()} (exit {rc})"
    return f"exit {rc}"


def main():
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    block = extract_block((repo / RST).read_text())
    test_src = (repo / TEST).read_text()

    has_init, registers, same_as_tested = structural_checks(block, test_src)
    construct_ok, construct_err, p0, apply_err = runtime_checks(block)
    p0_ok = p0 is not None and abs(p0 - 0.1) < 0.01

    rows = [
        ("__init__ is a method of CustomNoiseChannel:", has_init),
        ("register_channel runs at module level:     ", registers),
        ("class matches test_NoiseModel.py:          ", same_as_tested),
        ("constructs; Kraus ops sum to identity:     ", construct_ok),
        ("apply_noise gives P(0) = 0.1 +/- 0.01:     ", p0_ok),
    ]
    for label, ok in rows:
        print(f"{label} {ok}")
    if construct_err:
        print(f"  construct: {construct_err}")
    if apply_err:
        print(f"  apply_noise: {apply_err}")
    if p0 is not None:
        print(f"  measured P(0) = {p0:.4f}")

    passed = all(ok for _, ok in rows)
    print("\nAll checks passed." if passed else "\nChecks FAILED.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
