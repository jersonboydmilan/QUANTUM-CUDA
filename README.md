# QUANTUM-CUDA

Working notes and artifacts from contributing to
[NVIDIA/cuda-quantum](https://github.com/NVIDIA/cuda-quantum), NVIDIA's C++ and
Python implementation of the CUDA-Q programming model for heterogeneous
quantum-classical workflows.

**Upstream contribution:** [NVIDIA/cuda-quantum#5459](https://github.com/NVIDIA/cuda-quantum/pull/5459)
— *Document `exp_pauli`, `givens_rotation` and `fermionic_swap`*, closing
[issue #2141](https://github.com/NVIDIA/cuda-quantum/issues/2141).

---

## Contents

| Path | What it is |
| --- | --- |
| `verification/verify_gates.py` | Standalone numerical check of every matrix and sign convention asserted in the PR |
| `patches/0001-document-exp-pauli-givens-fermionic-swap.patch` | The upstream commit, as a `git format-patch` export |

---

## Repository analysis

Read of `NVIDIA/cuda-quantum` as of 2026-09-21, at release 0.16.0
(published 2026-09-12): Apache-2.0, primary language C++, ~1.1k stars, ~455
forks, pushed to daily.

### Layout

The tree separates the compiler, the runtime, and the language frontends:

- `cudaq/` — the MLIR compiler. Quake and CC dialects, passes, codegen.
  `cudaq/include/cudaq/Optimizer/Dialect/Quake/QuakeOps.td` is the authoritative
  definition of each quantum operation's semantics.
- `runtime/` — the execution side. `runtime/nvqir/` holds the simulator
  interface (`CircuitSimulator.h`) and the cuStateVec / cuTensorNet backends;
  `runtime/cudaq/qis/qubit_qis.h` is the C++ gate-level user API.
- `python/` — the Python frontend. `python/cudaq/kernel/ast_bridge.py` lowers
  decorated Python kernels to Quake; `python/cudaq/kernel/kernel_builder.py` is
  the dynamic builder API.
- `docs/sphinx/` — user documentation. `api/default_ops.rst` is the per-gate
  reference page, `api/languages/python_api.rst` drives autodoc for the Python
  API.
- `unittests/`, `targettests/` — C++ unit tests and per-target integration tests.

### Contribution requirements

- **DCO sign-off is mandatory** on every commit (`git commit -s`). The repo's
  `.github/dco.yml` allows individual remediation commits but not third-party
  ones. The DCO app wants a sign-off per listed author, so an unsigned
  `Co-authored-by` trailer will block a PR.
- **pre-commit runs in CI**: clang-format 22 on C++, yapf (Google style) on
  Python, markdownlint on Markdown, pyspelling/aspell on Markdown, reST and HTML
  against `.github/pre-commit/spelling_allowlist.txt`, plus license-header and
  Markdown link checks at the `pre-push` stage.
- Features and bug fixes are expected to start from an issue, so that a
  maintainer can weigh in on the approach before code is written.

### Where a newcomer can start

Of the 13 open `good first issue` items, all are marked `stale-notified`. They
split cleanly by what hardware you need:

- **Needs a CUDA GPU and a full build** — the build-system, simulator and
  noise-model items (`#1571`, `#914`, `#2584`, `#1640`).
- **Verifiable without one** — the documentation items (`#2141`, `#2215`,
  `#710`, `#3103`) and some of the testing items.

Without local GPU access, the second group is where work can actually be checked
before submitting rather than guessed at. That is what drove the choice below.

---

## The contribution

Issue #2141 reports that `exp_pauli`, `givens` and `fermionic_swap` are
undocumented, and asks directly: *"In the first place, what is `givens`?"*

Confirmed still valid. `docs/sphinx/api/default_ops.rst` documents `x`, `y`, `z`,
`h`, `r1`, `rx`, `ry`, `rz`, `s`, `t`, `swap` and `u3`, and none of the three.
`givens_rotation` and `fermionic_swap` exist as `PyKernel` methods but were
absent from the `automethod` list in `python_api.rst`, so they never rendered at
all.

### What the PR changes

1. **`exp_pauli` section** added to the Quantum Operations page: the
   `exp(i theta P)` convention, the rule that Pauli characters are matched to
   target qubits in order (so word length must equal target count), and both
   supported call forms in Python and C++.
2. **`givens_rotation` and `fermionic_swap`** listed in the Python API
   reference, with docstrings expanded to carry the definition and unitary
   matrix of each rotation — the part that answers the question the issue
   actually asks.
3. **A sign error corrected** in the doc comment on
   `CircuitSimulator::applyExpPauli`.

### The sign error

The doc comment claimed the method applies `exp(-i theta P)`. The base
implementation performs the basis change, the CNOT ladder, and then
`rz(-2.0 * theta)`. Since `rz(a) = exp(-i*a*Z/2)`, substituting `a = -2*theta`
gives `exp(+i*theta*Z)` on the ladder target, so the operation as a whole is
`exp(+i theta P)`.

Four other descriptions in the tree already agreed on the positive sign:

| Location | States |
| --- | --- |
| `QuakeOps.td` | "applies exp(i theta P)" |
| `runtime/cudaq/builder/kernel_builder.h` | "exp(i theta P)" |
| `python/cudaq/kernel/kernel_builder.py` | "`exp(i theta P)`" |
| `python/cudaq/qis/qis.py` | "`exp(i theta P)`" |

The comment on `applyExpPauli` was the lone outlier, and it sits on the method
that every simulator backend overrides — so it is the description a backend
author is most likely to read. Comment only; no behavior change.

### Scope left to the maintainers

Two questions were raised on the issue rather than decided unilaterally:

- On the C++ side, `givens_rotation` and `fermionic_swap` live in
  `cudaq_internal::` under `runtime/cudaq/kernels/`, which
  `CppAPICodingStyle.md` designates as internal, and they are referenced only
  from `unittests/nvqpp/integration/gate_library_tester.cpp`. They were
  therefore not presented as public C++ operations.
- `cudaq.lib.givens` and `cudaq.lib.fermionic_swap` are reachable as
  `@cudaq.kernel` functions, but `cudaq.lib` is otherwise undocumented, so they
  were not promoted either.

---

## Verification

None of the matrices in the PR were transcribed from the existing header
comments. Each was recomputed from the gate sequence CUDA-Q actually executes
and compared:

```console
$ python3 verification/verify_gates.py
Givens matrix matches the exp_pauli pair:        True
Fermionic SWAP matrix matches the gate sequence: True
applyExpPauli decomposition == exp(+i theta P):  True
applyExpPauli decomposition == exp(-i theta P):  False
r1(x) == exp(i*x/2) * rz(x):                     True

All checks passed.
```

The third and fourth lines are the pair that caught the sign error: the
decomposition matches the positive convention and not the negative one.

Requires only `numpy`. Matrix exponentials use the closed form
`exp(i*t*P) = cos(t)*I + i*sin(t)*P`, valid because a Pauli tensor product
squares to the identity, so `scipy` is not needed. Qubit ordering is big-endian
(q0 most significant), matching the convention CUDA-Q uses for multi-qubit
operator matrices.

---

## Reproducing the patch

```bash
git clone https://github.com/NVIDIA/cuda-quantum.git
cd cuda-quantum
git am ../patches/0001-document-exp-pauli-givens-fermionic-swap.patch
```

The patch carries the original DCO sign-off.
