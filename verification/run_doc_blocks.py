#!/usr/bin/env python3
"""Run every [Begin X] ... [End X] block of a docs example file on its own,
the way a reader copies one code cell, and report what fails.

Usage: run_doc_blocks.py <example.py> [block-name ...]
"""
import re, subprocess, sys, tempfile, textwrap
from pathlib import Path

src = Path(sys.argv[1]).read_text()
only = set(sys.argv[2:])
pat = re.compile(r"#\s*\[Begin ([^\]]+)\]\n(.*?)#\s*\[End \1\]", re.S)
for m in pat.finditer(src):
    name, body = m.group(1), m.group(2)
    if only and name.strip("`") not in only:
        continue
    code = "import cudaq\n" + textwrap.dedent(body)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    p = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=600)
    Path(f.name).unlink()
    err = [l for l in p.stderr.splitlines() if "Error" in l and "Warning" not in l]
    status = "OK" if p.returncode == 0 else f"FAIL: {err[-1] if err else 'exit ' + str(p.returncode)}"
    print(f"{name:32} {status}")
