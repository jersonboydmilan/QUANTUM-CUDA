#!/usr/bin/env python3
"""Numerical verification of the claims made in NVIDIA/cuda-quantum#5459.

Every matrix and sign convention documented in that PR is checked here against
the gate sequences that CUDA-Q actually executes, rather than transcribed from
the existing header comments.

Checks:
  1. The Givens rotation matrix, against the two `exp_pauli` calls in
     `runtime/cudaq/kernels/givens_rotation.h`.
  2. The fermionic SWAP matrix, against the full gate sequence in
     `runtime/cudaq/kernels/fermionic_swap.h`, global phase correction included.
  3. The sign convention of `CircuitSimulator::applyExpPauli`, against its
     basis-change / CNOT-ladder / `rz(-2*theta)` decomposition. This is the
     check that showed the doc comment on that method had the sign backwards.
  4. The r1 / rz relationship, which is why the two gates differ despite being
     described as the same rotation.

Only numpy is required; `expm` is avoided by using the closed form for Pauli
exponentials, which holds because a Pauli tensor product squares to identity.

Qubit ordering is big-endian throughout (q0 is the most significant bit), which
is the convention CUDA-Q uses for multi-qubit operator matrices.
"""

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1]).astype(complex)
H = (X + Z) / np.sqrt(2)

PAULIS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def pauli_word(word):
    """Tensor product of the Pauli characters in `word`, first character first."""
    out = np.array([[1]], dtype=complex)
    for char in word:
        out = np.kron(out, PAULIS[char])
    return out


def exp_pauli(theta, word):
    """exp(i * theta * P). Uses cos/sin closed form since P @ P == identity."""
    P = pauli_word(word)
    return np.cos(theta) * np.eye(P.shape[0]) + 1j * np.sin(theta) * P


def rz(angle):
    return np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])


def rx(angle):
    return np.cos(angle / 2) * I - 1j * np.sin(angle / 2) * X


def r1(angle):
    return np.diag([1, np.exp(1j * angle)]).astype(complex)


def on(gate, qubit, n=2):
    """Lift a single-qubit gate onto `qubit` of an n-qubit register."""
    ops = [I] * n
    ops[qubit] = gate
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


def cnot(control, target, n=2):
    dim = 2**n
    out = np.zeros((dim, dim), dtype=complex)
    for basis in range(dim):
        bits = [(basis >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[control]:
            bits[target] ^= 1
        flipped = sum(bit << (n - 1 - k) for k, bit in enumerate(bits))
        out[flipped, basis] = 1
    return out


def compose(*gates):
    """Circuit order: leftmost argument is applied first."""
    out = np.eye(gates[0].shape[0], dtype=complex)
    for gate in gates:
        out = gate @ out
    return out


def check_givens(theta=0.7321):
    """givens_rotation.h: exp_pauli(-theta/2, "YX") then exp_pauli(theta/2, "XY")."""
    actual = compose(exp_pauli(-0.5 * theta, 'YX'), exp_pauli(0.5 * theta, 'XY'))
    c, s = np.cos(theta), np.sin(theta)
    documented = np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=complex)
    return np.allclose(actual, documented), actual, documented


def check_fermionic_swap(phi=0.541):
    """fermionic_swap.h, including the r1(phi) rz(-phi) global phase correction."""
    actual = compose(
        on(H, 0), on(H, 1),
        cnot(0, 1), on(rz(phi / 2), 1), cnot(0, 1),
        on(H, 0), on(H, 1),
        on(rx(np.pi / 2), 0), on(rx(np.pi / 2), 1),
        cnot(0, 1), on(rz(phi / 2), 1), cnot(0, 1),
        on(rx(-np.pi / 2), 0), on(rx(-np.pi / 2), 1),
        on(rz(phi / 2), 0), on(rz(phi / 2), 1),
        on(r1(phi), 0), on(rz(-phi), 0),
    )
    e = np.exp(1j * phi / 2)
    c, s = np.cos(phi / 2), np.sin(phi / 2)
    documented = np.array([
        [1, 0, 0, 0],
        [0, e * c, -1j * e * s, 0],
        [0, -1j * e * s, e * c, 0],
        [0, 0, 0, np.exp(1j * phi)],
    ], dtype=complex)
    return np.allclose(actual, documented), actual, documented


def check_exp_pauli_sign(theta=0.33):
    """CircuitSimulator::applyExpPauli decomposition, for the word "XY".

    Basis change (H on an X target, rx(pi/2) on a Y target), CNOT ladder,
    rz(-2 * theta) on the last qubit in the support, then uncompute.
    """
    actual = compose(
        on(H, 0), on(rx(np.pi / 2), 1),
        cnot(0, 1), on(rz(-2 * theta), 1), cnot(0, 1),
        on(rx(-np.pi / 2), 1), on(H, 0),
    )
    return (np.allclose(actual, exp_pauli(theta, 'XY')),
            np.allclose(actual, exp_pauli(-theta, 'XY')))


def check_r1_vs_rz(angle=0.9):
    """r1 and rz are the same rotation up to a global phase of exp(i*angle/2)."""
    return np.allclose(r1(angle), np.exp(1j * angle / 2) * rz(angle))


def main():
    failures = 0

    ok, _, _ = check_givens()
    print(f"Givens matrix matches the exp_pauli pair:        {ok}")
    failures += not ok

    ok, _, _ = check_fermionic_swap()
    print(f"Fermionic SWAP matrix matches the gate sequence: {ok}")
    failures += not ok

    positive, negative = check_exp_pauli_sign()
    print(f"applyExpPauli decomposition == exp(+i theta P):  {positive}")
    print(f"applyExpPauli decomposition == exp(-i theta P):  {negative}")
    failures += not (positive and not negative)

    ok = check_r1_vs_rz()
    print(f"r1(x) == exp(i*x/2) * rz(x):                     {ok}")
    failures += not ok

    if failures:
        print(f"\n{failures} check(s) FAILED")
    else:
        print("\nAll checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
