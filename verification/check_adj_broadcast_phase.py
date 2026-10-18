#!/usr/bin/env python3
"""Check that broadcast `.adj` on a custom operation really applies the
adjoint, using an operation that is not its own inverse.

Usage: check_adj_broadcast_phase.py

With S = diag(1, i), H S^dagger S H = I, so every qubit returns to |0>.
If `.adj` were silently dropped, the circuit would be H S S H = H Z H = X
and every qubit would read 1. A Hermitian test operation (such as X) gives
the same counts either way, so it cannot tell the two apart.
"""

import re

import cudaq
import numpy as np

cudaq.register_operation("custom_s", np.array([1, 0, 0, 1j], dtype=complex))


@cudaq.kernel
def broadcast_adj():
    q = cudaq.qvector(3)
    h(q)
    custom_s.adj(q)
    s(q)
    h(q)


@cudaq.kernel
def single_qubit_adj():
    q = cudaq.qvector(1)
    h(q[0])
    custom_s.adj(q[0])
    s(q[0])
    h(q[0])


def main():
    single = cudaq.sample(single_qubit_adj, shots_count=1000)
    print("single-qubit custom_s.adj(q[0]):", single.most_probable())
    try:
        counts = cudaq.sample(broadcast_adj, shots_count=1000)
    except RuntimeError as e:
        message = re.sub(r"\x1b\[[0-9;]*m", "", str(e))
        print("broadcast custom_s.adj(q):      rejected:",
              message.split("error: ")[-1].splitlines()[0])
        return 1
    ok = counts.most_probable() == "000" and len(counts) == 1
    print("broadcast custom_s.adj(q):     ", counts.most_probable(),
          "(adjoint honored)" if ok else "(adjoint NOT honored)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
