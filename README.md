<div align="center">

# QUANTUM-CUDA

**Contribution work for [NVIDIA CUDA-Q](https://github.com/NVIDIA/cuda-quantum)** — analysis, verification, and the patches behind upstream documentation fixes and bug reports.

[![#5459 merged](https://img.shields.io/badge/%235459-merged-8250df?logo=nvidia&logoColor=white)](https://github.com/NVIDIA/cuda-quantum/pull/5459)
[![#5609 merged](https://img.shields.io/badge/%235609-merged-8250df?logo=nvidia&logoColor=white)](https://github.com/NVIDIA/cuda-quantum/pull/5609)
[![#5608 open](https://img.shields.io/badge/%235608-open-76B900?logo=nvidia&logoColor=white)](https://github.com/NVIDIA/cuda-quantum/pull/5608)
[![#5649 open](https://img.shields.io/badge/%235649-open-76B900?logo=nvidia&logoColor=white)](https://github.com/NVIDIA/cuda-quantum/pull/5649)
[![Issues filed](https://img.shields.io/badge/issues-%235646%20%235647-1f6feb)](https://github.com/NVIDIA/cuda-quantum/issues/5646)
[![Verification](https://img.shields.io/badge/verification-all%20passing-2da44e)](#running-the-verification)
[![License](https://img.shields.io/badge/license-Apache--2.0-d4a72c)](LICENSE)

</div>

---

## Contributions

| Item | Status | Change | Verification |
| :--- | :--- | :--- | :--- |
| [#5459](https://github.com/NVIDIA/cuda-quantum/pull/5459) | Merged 2026-09-24 | Document `exp_pauli`, `givens_rotation` and `fermionic_swap`; fix the sign in the `applyExpPauli` doc comment. Closes [#2141](https://github.com/NVIDIA/cuda-quantum/issues/2141). | [`verify_gates.py`](verification/verify_gates.py), numerical, 4/4 |
| [#5608](https://github.com/NVIDIA/cuda-quantum/pull/5608) | Open | Fix the custom `KrausChannel` example under `cudaq.apply_noise`, which crashes as printed. First item of [#5181](https://github.com/NVIDIA/cuda-quantum/issues/5181). | [`verify_apply_noise_example.py`](verification/verify_apply_noise_example.py), runtime, 5/5 |
| [#5609](https://github.com/NVIDIA/cuda-quantum/pull/5609) | Merged 2026-10-16 | Document how a `qvector` can be passed to a custom operation in Python vs C++. Requested in [#4961](https://github.com/NVIDIA/cuda-quantum/issues/4961). | [`verify_custom_op_qvector.py`](verification/verify_custom_op_qvector.py), runtime, 7/7 |
| [#5646](https://github.com/NVIDIA/cuda-quantum/issues/5646) | Issue, open; fix in #5649 | Python bridge: single-target custom operations do not broadcast over a `qvector`, against spec §6. Filed at the CUDA-Q architect's request in review of #5609. | [`probe_bridge_consistency.py`](verification/probe_bridge_consistency.py), runtime probe |
| [#5647](https://github.com/NVIDIA/cuda-quantum/issues/5647) | Issue, open; direction settled | Python bridge: multi-target custom operations pull targets out of a `qvector`, which nothing else in the language does. Same request. The architect has since ruled #2484 should not be implemented, so the fix is to reject the call. | [`probe_bridge_consistency.py`](verification/probe_bridge_consistency.py), runtime probe |
| [#5649](https://github.com/NVIDIA/cuda-quantum/pull/5649) | Open | Fix for #5646: broadcast single-target custom operations, plain and `.adj`, over a `qvector` and over several qubits. | [`verify_pr5649.sh`](verification/verify_pr5649.sh), runtime, 26/30 → 30/30 |

> [!NOTE]
> The rule for every item here: nothing is transcribed from existing comments or
> issue text. Each claim is recomputed from what CUDA-Q actually does — the gate
> sequences it executes, or the installed runtime itself — and the script that
> does it is in [`verification/`](verification/). That rule caught a sign error
> in #5459, a mistake in the issue text behind #5608, and in #5609 an asymmetry
> that the CUDA-Q architect identified as a Python bridge bug.

## What's here

| Path | Contents |
| :--- | :--- |
| [`verification/verify_gates.py`](verification/verify_gates.py) | #5459 — standalone numerical check of every matrix and sign convention. Requires only `numpy`. |
| [`verification/verify_apply_noise_example.py`](verification/verify_apply_noise_example.py) | #5608 — extracts the docs example from the `.rst` source and runs it against the installed `cudaq`. |
| [`verification/verify_custom_op_qvector.py`](verification/verify_custom_op_qvector.py) | #5609 — runs every Python call form the new note describes against the installed `cudaq`. |
| [`verification/probe_bridge_consistency.py`](verification/probe_bridge_consistency.py) | #5646 / #5647 — prints how the bridge treats `qvector` arguments for built-ins, custom operations, and user kernels, side by side. |
| [`verification/verify_pr5649.sh`](verification/verify_pr5649.sh) | #5649 — applies the PR's bridge change onto an installed `cudaq` wheel, runs the PR's test file before and after, then restores the wheel. |
| [`verification/check_adj_broadcast_phase.py`](verification/check_adj_broadcast_phase.py) | #5649 — confirms broadcast `.adj` applies the adjoint, using a non-Hermitian operation the PR's own test cannot distinguish. |
| [`verification/run_isolated.sh`](verification/run_isolated.sh) | Runs each pytest test in its own process, so a JIT-side abort on macOS arm64 fails one test instead of the whole run. |
| [`patches/`](patches/) | Each upstream commit as a `git format-patch` export, DCO sign-off intact, plus the #5649 bridge change re-anchored to 0.16.0. |

---

## PR #5459 — `exp_pauli`, `givens_rotation` and `fermionic_swap`

> Merged 2026-09-24 by @sacpis. Closes #2141.

CUDA-Q ships three quantum operations that user code can reach but the
documentation never mentions: `exp_pauli`, `givens_rotation` and
`fermionic_swap`. [Issue #2141](https://github.com/NVIDIA/cuda-quantum/issues/2141)
reports the gap and asks directly — *"in the first place, what is `givens`?"*

[**PR #5459**](https://github.com/NVIDIA/cuda-quantum/pull/5459) documents all
three. While pinning down the convention to document, it also turned up a sign
error in the doc comment on the simulator method that every backend overrides.

### The bug

The doc comment on `CircuitSimulator::applyExpPauli` claimed the method applies
`exp(-i·θ·P)`. It does not.

The base implementation performs a basis change, a CNOT ladder, and then
`rz(-2.0 * theta)` on the ladder target. Since `rz(a) = exp(-i·a·Z/2)`,
substituting `a = -2θ` gives `exp(+i·θ·Z)` — so the operation as a whole applies
the **positive** convention.

Four other descriptions in the tree already said so:

| Location | States |
| :--- | :--- |
| `Optimizer/Dialect/Quake/QuakeOps.td` | `applies exp(i theta P)` |
| `runtime/cudaq/builder/kernel_builder.h` | `exp(i theta P)` |
| `python/cudaq/kernel/kernel_builder.py` | `exp(i theta P)` |
| `python/cudaq/qis/qis.py` | `exp(i theta P)` |
| **`runtime/nvqir/CircuitSimulator.h`** | **`exp(-i theta ...)`** ← the outlier |

It matters because `applyExpPauli` is the virtual method each simulator backend
overrides, so its comment is the description a backend author reads first. The
fix is comment-only; no behavior changes.

### Verification

```console
$ python3 verification/verify_gates.py
Givens matrix matches the exp_pauli pair:        True
Fermionic SWAP matrix matches the gate sequence: True
applyExpPauli decomposition == exp(+i theta P):  True
applyExpPauli decomposition == exp(-i theta P):  False
r1(x) == exp(i*x/2) * rz(x):                     True

All checks passed.
```

Lines three and four are the pair that caught the sign error — the decomposition
matches the positive convention and fails the negative one.

<details>
<summary><b>What each check does</b></summary>

<br>

1. **Givens** — rebuilds the matrix from the two `exp_pauli` calls in
   `runtime/cudaq/kernels/givens_rotation.h` and compares against the documented
   form.
2. **Fermionic SWAP** — replays the full eighteen-gate sequence from
   `runtime/cudaq/kernels/fermionic_swap.h`, global phase correction included.
3. **`exp_pauli` sign** — rebuilds the `applyExpPauli` decomposition and tests it
   against *both* conventions, so the result distinguishes them rather than
   merely confirming one.
4. **`r1` vs `rz`** — shows the two differ only by a global phase of
   `exp(i·x/2)`, which is why they are distinct gates despite identical verbal
   descriptions.

Matrix exponentials use the closed form `exp(i·t·P) = cos(t)·I + i·sin(t)·P`,
valid because a Pauli tensor product squares to the identity — so `scipy` is not
needed. Qubit ordering is big-endian (q0 most significant), matching the
convention CUDA-Q uses for multi-qubit operator matrices.

</details>

### The change

1. **`exp_pauli` section** added to the Quantum Operations reference page: the
   `exp(i·θ·P)` convention, the rule that Pauli characters map to target qubits
   in order — so word length must equal target count — and both supported call
   forms in Python and C++.
2. **`givens_rotation` and `fermionic_swap`** added to the Python API reference.
   Both are `PyKernel` methods, but neither appeared in the `automethod` list, so
   they never rendered at all. Their docstrings now carry the definition and
   unitary matrix of each rotation.
3. **The sign comment** on `applyExpPauli` corrected.

Four files, +72 / −5. Review feedback from @sacpis — Python's `exp_pauli`
dispatches its arguments by type, so only C++ requires the Pauli word to precede
the qubits — was addressed before merge.

#### Scope left to the maintainers

Two questions were raised on the issue rather than settled unilaterally:

- On the C++ side, `givens_rotation` and `fermionic_swap` live in
  `cudaq_internal::` under `runtime/cudaq/kernels/`, which `CppAPICodingStyle.md`
  designates as internal, and they are referenced only from
  `unittests/nvqpp/integration/gate_library_tester.cpp`. They were therefore not
  presented as public C++ operations.
- `cudaq.lib.givens` and `cudaq.lib.fermionic_swap` are reachable as
  `@cudaq.kernel` functions, but `cudaq.lib` is otherwise undocumented, so they
  were not promoted either.

---

## PR #5608 — custom `KrausChannel` example under `cudaq.apply_noise`

> Open. Fixes the first item of #5181.

The Python API reference shows how to register a custom noise channel for
`cudaq.apply_noise`. As printed, the example cannot be used: the constructor is
written at the same indentation as the class, so it is a module-level function
rather than a method.

```python
class CustomNoiseChannel(cudaq.KrausChannel):
    num_parameters = 1
    num_targets = 1

def __init__(self, params: list[float]):   # ← outside the class
    cudaq.KrausChannel.__init__(self)
    ...
```

The class silently falls back to the base constructor, so the failure only shows
up later and far from its cause:

- `CustomNoiseChannel([0.1])` raises `RuntimeError: std::bad_cast`;
- passing the channel to `cudaq.apply_noise` terminates the interpreter with
  `SIGABRT`.

### A correction to the issue

Issue #5181 also says the trailing `noise = cudaq.NoiseModel()` /
`noise.register_channel(...)` lines are indented inside `__init__` and never run.
Extracting the block from the `.rst` source rather than reading the rendered page
shows that they are already at module level. The only defect is the placement of
`__init__`, so the fix is smaller than the issue suggests.

### Verification

The script pulls the code block out of `python_api.rst` exactly as Sphinx renders
it, then runs it against the installed `cudaq` on `density-matrix-cpu`. Every
runtime step is a separate subprocess, because the unfixed block aborts the
interpreter.

```console
$ python3 verification/verify_apply_noise_example.py ../cuda-quantum   # main
__init__ is a method of CustomNoiseChannel: False
register_channel runs at module level:      True
class matches test_NoiseModel.py:           False
constructs; Kraus ops sum to identity:      False
apply_noise gives P(0) = 0.1 +/- 0.01:      False
  construct: RuntimeError: std::bad_cast (exit 1)
  apply_noise: libc++abi: terminating due to uncaught exception of type nanobind::python_error: RuntimeError: std::bad_cast (exit -6)

Checks FAILED.

$ python3 verification/verify_apply_noise_example.py ../cuda-quantum   # with the patch
__init__ is a method of CustomNoiseChannel: True
register_channel runs at module level:      True
class matches test_NoiseModel.py:           True
constructs; Kraus ops sum to identity:      True
apply_noise gives P(0) = 0.1 +/- 0.01:      True
  measured P(0) = 0.0985

All checks passed.
```

<details>
<summary><b>What each check does</b></summary>

<br>

1. **Method placement** — parses the block with `ast` and checks `__init__` is in
   the class body.
2. **Registration** — checks `register_channel` is a top-level statement. This is
   the check that disproved the second half of the issue text.
3. **Matches the tested class** — compares the documented class with the
   `CustomNoiseChannel` in `python/tests/builder/test_NoiseModel.py` by AST
   equality, so the docs now show exactly what CI exercises.
4. **Valid channel** — constructs the channel and checks the completeness
   relation `Σ Kᵢ†Kᵢ = I`.
5. **Physical behavior** — the channel is a bit flip with probability `p`. After
   `x(q)` and `apply_noise(CustomNoiseChannel, 0.1, q)`, 20 000 shots should
   measure `0` about 10% of the time. The standard error is about 0.0021, so
   the ±0.01 band is close to five standard errors wide.

</details>

### The change

`__init__` indented into the class, the `dtype=` continuation lines aligned, and
the two imports the block uses added so it runs as printed. One file, +15 / −12.

### Coordination

The other nine items of #5181 are in #5601, which deliberately leaves
`python_api.rst` untouched and defers this item to #2969. That PR would also fix
it, but has had no activity since June 2025. #5608 is the minimal standalone fix
and shares no files with #5601.

---

## PR #5609 — `qvector` targets for custom operations, Python vs C++

> Merged 2026-10-16 by @khalatepradnya. Documents the API difference requested
> in #4961. Its review produced issues #5646 and #5647 — see
> [In review](#in-review-the-asymmetry-is-a-bug).

Since #4173, a Python custom operation accepts its targets as a whole `qvector`.
#4961 reported that C++ does not, and asked for C++ to follow. A compiler
maintainer declined: C++ accepts a quantum sequence only where its meaning is
unambiguous, and further overloading would weaken type checking. The reporter
then asked for the difference to be documented, because a user had hit it. No
one had picked that up — the offers on the issue were all for the declined code
change.

### What the note says, and where each statement comes from

| Statement | Evidence |
| :--- | :--- |
| Python: targets may be individual qubits, a whole `qvector`, or a mix | Executed below; also covered by `python/tests/custom/test_custom_operations.py` |
| Python: the total must equal the operation's target count | Executed below; `__expandCustomOpTargets` in `python/cudaq/kernel/ast_bridge.py` |
| C++: targets of a multi-qubit custom operation must be passed individually | Maintainer comment on #4961; `cudaq/lib/Frontend/nvqpp/ConvertExpr.cpp` |
| C++: a `qvector` as the sole argument of a 1-qubit custom operation applies it to each qubit | `ConvertExpr.cpp`: the single-`veq` branch emits a loop of `CustomUnitaryCallOp` |
| C++: a `qvector` may be the controls of a controlled custom operation | `maybeUnpackOperands` in `ConvertExpr.cpp`: the last `targetCount` operands are targets, the rest controls |

### Verification

```console
$ python3 verification/verify_custom_op_qvector.py
cudaq 0.16.0

individual qubits:  custom_swap(q[0], q[1]) -> 01             True
whole qvector:      custom_swap(q)         -> 01             True
unpacked qvector:   custom_swap(*q)        -> 01             True
mixed:              custom_swap(q, r)      -> 01             True
too few qubits:     custom_swap(r)         -> refused (RuntimeError) True
runtime-sized, n=2: custom_swap(q)         -> 01             True
1-target op on qvector(2): custom_x(q)     -> refused (terminated) True

All checks passed.
```

`custom_swap` is a two-target operation applied to `|10>`, so every accepted form
must yield `01`.

### What was found along the way

The last row is a second asymmetry, in the opposite direction. C++ applies a
single-qubit custom operation to every qubit of a `qvector`; Python rejects the
same call through the target-count check, unlike built-in operations such as
`x(q)`. The note states only the count rule for Python, so it stays correct
whichever way that is meant to go, and the question is raised in the PR rather
than decided there.

On macOS arm64 that rejection terminates the process instead of raising, because
the check fires inside JIT-compiled code. Upstream already skips
`test_unknown_veq_size_incorrect_count` on that platform for the same reason, so
this is a known limitation rather than a new bug.

### Not verified locally

The C++ statements come from reading the bridge and from the maintainer's
comment; `nvq++` is not part of the `pip` wheel, so they were not compiled here.
The PR says so and asks a reviewer to confirm that tab.

### In review: the asymmetry is a bug

@schweitzpgi, CUDA-Q's architect, answered the open question directly. The
language rule is that an operation defined on a single target is overloaded to
broadcast over a sequence, and that passing a sequence to an operation with more
than one target is a hard error, because arguments are positional. Python
custom operations get both halves backwards. He asked for issues to be filed if
none existed; @khalatepradnya seconded it.

None existed, so the next step was to measure the whole surface rather than the
two cases already known. The probe runs each call form in its own subprocess
against `cudaq` 0.16.0:

```console
$ python3 verification/probe_bridge_consistency.py
[single-target]
  x(q), qvector(3)                     ok   { 111:1000 }
  custom_x(q), qvector(1)              ok   { 1:1000 }
  custom_x(q), qvector(3)              abort custom operation requires 1 qubit target(s), but 3 were provided
  custom_x(q), qvector(n), n=3         abort custom operation requires 1 qubit target(s), but 3 were provided
  t.adj(q), qvector(3)                 ok   { 000:1000 }
  custom_x.adj(q), qvector(3)          error invalid target operand - target must not be a qvector
  x(a, b)                              ok   { 11:1000 }
  custom_x(a, b)                       error custom operation requires 1 qubit target(s), but 2 were provided
  custom_x.ctrl(c, t), c = qvector(2)  ok   { 111:1000 }

[multi-target]
  swap(q[0], q[1])                     ok   { 01:1000 }
  swap(q), qvector(2)                  error missing value
  swap(*q), qvector(2)                 error CUDA-Q does not currently support Starred expressions
  custom_swap(q), qvector(2)           ok   { 01:1000 }
  custom_swap(*q), qvector(2)          ok   { 01:1000 }
  custom_swap(q, r), qvector(1)        ok   { 01:1000 }

[user kernel]
  two_qubit_kernel(q), qvector(2)      error missing value
  two_qubit_kernel(*q), qvector(2)     error CUDA-Q does not currently support Starred expressions

[hermitian adj]
  x.adj(q[0])                          error 'quake.x' op may not be adjoint
  h.adj(q[0])                          error 'quake.h' op may not be adjoint
  t.adj(q[0])                          ok   { 0:1000 }
```

Read top to bottom, it shows three things that the two known cases did not:

- **The single-target failure is broader than broadcast.** `custom_x(q)` works
  only when `len(q) == 1`, by coincidence of the count rule. `custom_x(a, b)`,
  which spec §4 covers, fails too. The `.adj` form fails with a different error,
  so plain and adjoint calls go through separate code paths with separate bugs.
- **Custom operations disagree with themselves.** In the multi-target section,
  only the plain custom-op call accepts `q`, `*q` and `q, r`. Built-in `swap`
  and user kernels reject all of them, and so do the custom op's own `.adj` and
  `.ctrl` paths. It is also the only place in the bridge that accepts a starred
  argument.
- **The history explains it.** #2452 began as the single-target broadcast case,
  and in that thread @schweitzpgi asked whether broadcast was intended. #4173
  later closed #2452 with the target-count check instead, and added `qvector`
  extraction for multi-target calls. Only the plain path got that change; its
  review had asked whether `.ctrl`/`.adj` needed the same logic.

The result was two issues, each with a reproducer, the full table, permalinks
to the root cause at `f0e69e9`, and the decisions it leaves to maintainers:

| Issue | Scope | Open decisions listed for maintainers |
| :--- | :--- | :--- |
| [#5646](https://github.com/NVIDIA/cuda-quantum/issues/5646) | Broadcast single-target custom ops in plain, `.adj`, and multi-`ref` forms, mirroring the C++ invariant loop | — (fix in [#5649](#pr-5649--broadcast-single-target-custom-operations)) |
| [#5647](https://github.com/NVIDIA/cuda-quantum/issues/5647) | Make a `qvector` target of a multi-target custom op a compile-time error with a clear message | ~~#2484 asks for the opposite~~ — settled in #5649's review: #2484 should not be implemented. Still open: six tests pin current behavior; the note from #5609 will need updating; whether `*q` stays |

The last probe section is outside both issues. `.adj` on a self-adjoint gate
gets through the Python frontend, sets `is_adj`, and is rejected later by the
`Hermitian` trait verifier — and the frontend's own typo hint suggests
`x.adj(...)`. Since `U† = U` there, the frontend could either drop the flag or
reject the call clearly. That was raised as a question in the PR thread rather
than filed unasked.

---

## PR #5649 — broadcast single-target custom operations

> Open. Fixes #5646.

Python now broadcasts a custom operation that has exactly one target, as
spec §6 and C++ already require. For `numTargets == 1`, a new helper emits one
`CustomUnitaryConstantOp` per qubit. Each individual qubit argument gets one
directly. Each `qvector` gets an invariant loop over `VeqSizeOp`, so a
runtime-sized vector works as well. The plain call and `.adj` both use it.
Every `numTargets > 1` call and every `.ctrl` call is left unchanged. The
docs note from #5609 is updated to match, and four regression tests are added.

### The scope note, and the architect's answer

The PR named two things it deliberately left alone: #5647, because #2484 asked
for the opposite, and broadcasting the target of a `.ctrl` call. @schweitzpgi
answered that section by stating the DSL's rules outright, so that Python and
C++ have one set of semantics:

| Rule (paraphrased) | What it settles |
| :--- | :--- |
| Every quantum operation, built-in or custom, takes one or more target qubits, each passed as its own positional argument. An operation is never defined over a `qvector` of targets. | #2484 is the wrong design and should not be implemented. #5647's direction is therefore no longer open: a `qvector` given as the targets of a multi-target operation must be rejected. |
| An operation with exactly one target also gets a broadcast overload, on its base form only, which applies it to each qubit of a `qvector`. | What #5649 implements. |
| The controlled form takes one or more controls, then exactly the base form's targets as the final positional arguments. Broadcast never combines with control, whatever the number of targets. | #5649 is right to leave `.ctrl` alone; there is no broadcast to add there. |
| Any change to these rules goes through specification review. | New overloads need a spec change first, not just a bridge change. |

Both points the PR marked as out of scope were therefore correct to leave out,
and the first one is now unblocked.

### Verification

The PR description says it could not be run, because `cudaq` supposedly ships
source-only on PyPI. That is not right: `pip install cudaq` installs a prebuilt
0.16.0 wheel on macOS arm64, and it is what every other script here runs
against. The Python bridge is pure Python, so the PR's change can go straight
onto that wheel without building CUDA-Q.

The PR's bridge hunk does not apply verbatim to 0.16.0, because one line of
trailing context changed on `main` since the release. In
[`patches/0004-pr5649-bridge-on-0.16.0.diff`](patches/0004-pr5649-bridge-on-0.16.0.diff)
only that context is re-anchored; its added and removed lines are
byte-identical to the PR. [`verify_pr5649.sh`](verification/verify_pr5649.sh)
applies it, runs the PR's own `test_custom_operations.py` with each test in a
separate process, and restores the original file on exit (checked by hash).

```console
$ bash verification/verify_pr5649.sh python3 test_custom_operations.py
== before: cudaq as installed
  ...26 existing tests                                           PASS
  test_single_target_broadcast_over_static_qvector               FAIL (process aborted)
  test_single_target_broadcast_over_dynamic_qvector              FAIL (process aborted)
  test_single_target_adjoint_broadcast_over_qvector              FAIL
  test_single_target_broadcast_over_individual_refs              FAIL
  -- 26 passed, 4 failed
single-qubit custom_s.adj(q[0]): 0
broadcast custom_s.adj(q):      rejected: invalid target operand - target must not be a qvector
== after: with the #5649 bridge change
  ...26 existing tests                                           PASS
  test_single_target_broadcast_over_static_qvector               PASS
  test_single_target_broadcast_over_dynamic_qvector              PASS
  test_single_target_adjoint_broadcast_over_qvector              PASS
  test_single_target_broadcast_over_individual_refs              PASS
  -- 30 passed, 0 failed
single-qubit custom_s.adj(q[0]): 0
broadcast custom_s.adj(q):      000 (adjoint honored)
```

All four new tests fail before the change and pass after it, and none of the 26
existing tests regress. `test_unknown_veq_size_incorrect_count` is skipped, as
upstream already does on macOS arm64. The two "process aborted" failures are
#5646's own symptom: the count check fires inside JIT code and terminates the
process instead of raising.

Rerunning [`probe_bridge_consistency.py`](verification/probe_bridge_consistency.py)
with the change applied, every single-target custom-operation row now matches
its built-in neighbour, and the multi-target rows are unchanged (#5647's scope):

```console
  custom_x(q), qvector(3)              ok   { 111:1000 }
  custom_x(q), qvector(n), n=3         ok   { 111:1000 }
  custom_x.adj(q), qvector(3)          ok   { 111:1000 }
  custom_x(a, b)                       ok   { 11:1000 }
```

### A gap in the PR's own tests

`test_single_target_adjoint_broadcast_over_qvector` registers `X`. Since
`X† = X`, it passes whether or not `.adj` is honored, so it cannot detect a
dropped adjoint. [`check_adj_broadcast_phase.py`](verification/check_adj_broadcast_phase.py)
uses `S = diag(1, i)` instead. `H·S†·S·H = I` returns every qubit to `0`; a
dropped adjoint would give `H·S·S·H = X` and every qubit would read `1`. With
the change it reads `000`, so the adjoint really is applied. Swapping the
PR's test operation for `S` with that sandwich would make the test meaningful.

### One point to confirm

The architect's rule puts the broadcast overload on an operation's base form
only, and rules out broadcast together with control. It does not explicitly
mention the adjoint form. #5649 broadcasts `.adj` too, which is consistent
with Python's built-ins (`t.adj(q)` broadcasts today), but whether "base form"
includes the adjoint is worth confirming in review.

---

## Repository analysis

Read of `NVIDIA/cuda-quantum` as of 2026-09-21, at release 0.16.0 — Apache-2.0,
primary language C++, ~1.1k stars, ~455 forks, pushed to daily.

```mermaid
flowchart LR
    subgraph frontend [Frontends]
        PY["python/<br/>ast_bridge · kernel_builder"]
        CPP["runtime/cudaq/qis/<br/>qubit_qis.h"]
    end
    subgraph compiler [Compiler]
        Q["cudaq/<br/>Quake + CC dialects"]
    end
    subgraph runtime [Runtime]
        SIM["runtime/nvqir/<br/>CircuitSimulator · cuStateVec"]
    end
    PY --> Q
    CPP --> Q
    Q --> SIM
```

| Area | Role |
| :--- | :--- |
| `cudaq/` | MLIR compiler. `QuakeOps.td` is the authoritative definition of each operation's semantics. |
| `runtime/nvqir/` | Simulator interface and the cuStateVec / cuTensorNet backends. |
| `runtime/cudaq/qis/` | C++ gate-level user API. |
| `python/cudaq/kernel/` | Python frontend — AST lowering to Quake, plus the dynamic builder. |
| `docs/sphinx/` | User documentation. `api/default_ops.rst` is the per-gate reference. |
| `unittests/`, `targettests/` | C++ unit tests and per-target integration tests. |

### Contribution requirements

- **DCO sign-off is mandatory** on every commit (`git commit -s`). The DCO app
  wants a sign-off per listed author, so an unsigned `Co-authored-by` trailer
  blocks a PR.
- **pre-commit runs in CI** — clang-format 22 (C++), yapf Google style (Python),
  markdownlint, pyspelling/aspell against a checked-in allowlist, plus
  license-header and Markdown link checks at the `pre-push` stage.
- Features and bug fixes are expected to **start from an issue**, so a maintainer
  can weigh in before code is written.

### Where a newcomer can start

All 13 open `good first issue` items are marked `stale-notified`. They split by
what hardware the work needs:

| Group | Issues | Needs |
| :--- | :--- | :--- |
| Build, simulator, noise models | `#1571` `#914` `#2584` `#1640` | CUDA GPU and a full build |
| Documentation and some testing | `#2141` `#2215` `#710` `#3103` | Nothing beyond a checkout |

Without local GPU access the second group is where work can be *verified* before
submitting rather than guessed at — which is what drove the choice here.

Since then, `pip install cudaq` has turned out to be enough to execute most
Python-facing claims for real, CPU simulators included. Every script after
`verify_gates.py` uses it.

## Running the verification

```bash
# PR #5459 — numpy only
python3 verification/verify_gates.py

# PR #5608, #5609 and issues #5646 / #5647 — the CUDA-Q wheel (Python 3.11+)
pip install cudaq numpy
git clone https://github.com/NVIDIA/cuda-quantum.git ../cuda-quantum
python3 verification/verify_apply_noise_example.py ../cuda-quantum
python3 verification/verify_custom_op_qvector.py
python3 verification/probe_bridge_consistency.py
```

`probe_bridge_consistency.py` is a probe rather than a pass/fail check. Once
#5646 and #5647 are fixed, its custom-operation rows should change to match the
built-in rows next to them.

```bash
# PR #5649 — the PR's own test file, before and after its bridge change
gh api "repos/jersonboydmilan/cuda-quantum/contents/python/tests/custom/test_custom_operations.py?ref=99730799367edd906e134ce45e4bfef1e5b72ea3" \
  --jq .content | base64 -d > /tmp/test_custom_operations.py
bash verification/verify_pr5649.sh python3 /tmp/test_custom_operations.py
```

`verify_pr5649.sh` edits the installed `cudaq/kernel/ast_bridge.py` for the
duration of the run and restores it on exit. Use a virtual environment.

`verify_apply_noise_example.py` reads the example from whichever checkout it is
given, so it reports the bug on `main` and passes once the patch is applied.

## Reproducing the patches

```bash
git clone https://github.com/NVIDIA/cuda-quantum.git
cd cuda-quantum
git am ../patches/0002-fix-apply-noise-custom-channel-example.patch
```

Each patch applies on its own. `0001` and `0003` are merged upstream; `0002` is
open and was cut from `main` at `be7b0b3`. All three carry their original DCO
sign-off. `0004` is different: it is the bridge part of #5649 re-anchored to
the 0.16.0 release, for verification only, so it targets an installed wheel
rather than a checkout.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jersonboydmilan/QUANTUM-CUDA&type=Date)](https://star-history.com/#jersonboydmilan/QUANTUM-CUDA&Date)

## License

[Apache-2.0](LICENSE).

The analysis and the scripts under `verification/` are original work in this
repository, released under Apache-2.0. The patches under `patches/` are diffs
against [NVIDIA/cuda-quantum](https://github.com/NVIDIA/cuda-quantum), which is
itself Apache-2.0 — so the same terms carry across both, and the upstream notice
governs the upstream-derived content.
