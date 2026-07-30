# SIAB Projected-Pi Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strictly pair ABACUS source sidecars with existing Sternheimer response-v1 targets, reconstruct the full-Coulomb symmetrized projected response, and determine whether that metric ranks the initial, joint, and guarded H-TZDP bases consistently with held-out SOS-RPA.

**Architecture:** A source reader mirrors the existing response reader and a pairing layer enforces primitive, overlap, occupation, and physical-provenance identity. A separate differentiable evaluator reconstructs `A(C)` and `Pi(C)` from complex source/response overlaps and real SIAB coefficients. A standalone analysis command evaluates three bases and three primitive-overlap rank thresholds; this plan does not connect the new metric to the optimizer.

**Tech Stack:** Python 3.10, PyTorch 2.1 complex128/float64, NumPy, Matplotlib, `unittest`, SLURM `normal` on `df_dcu`, LaTeX/Poppler for report verification.

---

## Dependency And Stop Rule

This plan starts only after
`2026-07-30-abacus-siab-source-sidecar.md` has produced successful H and H2
source sidecars plus zero-order comparison manifests. The existing response
targets remain immutable:

```text
/work1/ghj/sternheimer_abacus_tests/siab_greedy_targets_h2_channel_mpi_prod_v1_20260726/producer_atom/OUT.H_SIAB_GREEDY_ATOM_WFULL_NF16_E50/sternheimer_matrix.dat
/work1/ghj/sternheimer_abacus_tests/siab_greedy_targets_h2_channel_mpi_prod_v1_20260726/producer_h2/OUT.H2_SIAB_GREEDY_R074085_WFULL_NF16_E50/sternheimer_matrix.dat
```

If either optimized basis does not reduce the projected-Pi loss relative to
initial TZDP, or the result changes by more than 1% across overlap thresholds,
stop after documenting the negative result. Do not add the new loss to SIAB
optimization. The next design would export the primitive Hamiltonian and solve
a candidate-space Galerkin Sternheimer equation.

## File Map

Create:

- `SIAB/opt_orb_pytorch_dpsi/IO/sternheimer_text.py`: shared strict tagged-text parser helpers.
- `SIAB/opt_orb_pytorch_dpsi/sternheimer_source_data.py`: validated source-sidecar model.
- `SIAB/opt_orb_pytorch_dpsi/IO/read_sternheimer_source.py`: source-v1 parser.
- `SIAB/opt_orb_pytorch_dpsi/sternheimer_source_pair.py`: strict response/source pairing.
- `SIAB/opt_orb_pytorch_dpsi/projected_pi.py`: projected response and family loss.
- `SIAB/tests/fixtures/sternheimer_source_v1.dat`: canonical source fixture.
- `SIAB/tests/test_read_sternheimer_source.py`: reader/data-model tests.
- `SIAB/tests/test_sternheimer_source_pair.py`: pairing and provenance tests.
- `SIAB/tests/test_projected_pi.py`: direct complex algebra and gradient tests.
- `SIAB/tests/test_projected_pi_analysis.py`: analysis CLI contract test.
- `SIAB/example_H_sternheimer/projected_pi_loss/analyze_projected_pi.py`: three-basis analysis.
- `SIAB/example_H_sternheimer/projected_pi_loss/README.md`: immutable commands and gates.

Modify:

- `SIAB/opt_orb_pytorch_dpsi/IO/read_sternheimer.py`: use shared parser helpers without changing v1 behavior.
- `SIAB/tests/test_read_sternheimer.py`: prove parser extraction is compatible.
- `SIAB/example_H_sternheimer/fixed_dzp_tzdp_sos/README.md`: link the result.
- `/Users/ghj/同步空间/AITP_project/sternheimer_abacus/sternheimer_siab_project/main.tex`: formulas, provenance, ranking, and decision.

Do not modify `main.py`, `optimization_loss.py`,
`opt_orbital_converge.py`, the target-entry schema, or any production input in
this plan.

### Task 1: Extract Shared Tagged-Text Parsing Without Behavior Change

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/IO/sternheimer_text.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/IO/read_sternheimer.py`
- Modify: `SIAB/tests/test_read_sternheimer.py`

- [ ] **Step 1: Add an exact compatibility test**

Read `SIAB/tests/fixtures/sternheimer_matrix_v1.dat` before and after the
refactor and assert every scalar, tensor, block, provenance object, dtype, and
device is equal. Preserve representative error strings for unknown sections,
mismatched closing tags, duplicate provenance keys, non-finite complex values,
and non-Hermitian overlap.

- [ ] **Step 2: Run the existing reader suite as the characterization gate**

```bash
cd SIAB/tests
python -m unittest -v test_read_sternheimer
```

Expected before extraction: all tests pass. Save the test count and output;
this is a characterization gate, not a failing feature test.

- [ ] **Step 3: Move only format-neutral helpers**

Expose from `IO/sternheimer_text.py` the functions `read_sections(path,
required_sections)`, `parse_key_value_header(lines, allowed_keys, section)`,
`parse_blocks(lines, expected_count)`, `parse_complex_matrix(lines, section,
rows, columns)`, `parse_provenance(lines)`, `parse_count(value, field)`,
`parse_int(value, field)`, and `parse_float(value, field)`.

`read_sections` rejects unknown, repeated, missing, nested, mismatched, and
unterminated tags. Preserve the special one-line JSON handling for
`PROVENANCE_JSON`. `read_sternheimer.py` supplies its original six required
sections and header keys, so accepted files and error text remain unchanged.

- [ ] **Step 4: Prove exact reader compatibility**

```bash
cd SIAB/tests
python -m unittest -v test_read_sternheimer
```

Expected: the same test count passes and fixture tensors are bitwise equal.

- [ ] **Step 5: Commit parser extraction**

```bash
git add SIAB/opt_orb_pytorch_dpsi/IO/sternheimer_text.py \
        SIAB/opt_orb_pytorch_dpsi/IO/read_sternheimer.py \
        SIAB/tests/test_read_sternheimer.py
GIT_AUTHOR_NAME='Codex' GIT_AUTHOR_EMAIL='codex@openai.com' \
GIT_COMMITTER_NAME='AroundPeking' \
GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit -m 'refactor(siab): share Sternheimer text parsing'
```

### Task 2: Read And Validate Source-V1 Sidecars

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/sternheimer_source_data.py`
- Create: `SIAB/opt_orb_pytorch_dpsi/IO/read_sternheimer_source.py`
- Create: `SIAB/tests/fixtures/sternheimer_source_v1.dat`
- Create: `SIAB/tests/test_read_sternheimer_source.py`

- [ ] **Step 1: Write a failing canonical-reader test**

The fixture has two source rows, four primitive columns, the same primitive
blocks/overlap/provenance as `sternheimer_matrix_v1.dat`, and complex `D`.
Assert:

```python
data = read_sternheimer_source(FIXTURE)
self.assertEqual(data.format_version, 1)
self.assertEqual(data.d.shape, (2, 4))
self.assertEqual(data.overlap.shape, (4, 4))
self.assertEqual(data.occupied_state.dtype, torch.int64)
self.assertEqual(data.auxiliary_channel.dtype, torch.int64)
self.assertEqual(data.occupation.dtype, torch.float64)
self.assertEqual(data.norm.dtype, torch.float64)
self.assertEqual(data.d.dtype, torch.complex128)
self.assertEqual(data.d.device.type, "cpu")
```

Add failures for unsupported version, duplicate source key, negative index,
non-positive norm, zero/negative occupation, short `D`, non-finite values,
invalid block offsets, non-Hermitian `S`, and missing required provenance.

- [ ] **Step 2: Verify RED**

```bash
cd SIAB/tests
python -m unittest -v test_read_sternheimer_source
```

Expected: source reader module is absent.

- [ ] **Step 3: Implement the immutable data model**

Define:

```python
@dataclass(frozen=True)
class SternheimerSourceData:
    format_version: int
    grid_volume_bohr3: float
    blocks: tuple
    occupied_state: torch.Tensor
    auxiliary_channel: torch.Tensor
    occupation: torch.Tensor
    norm: torch.Tensor
    d: torch.Tensor
    overlap: torch.Tensor
    provenance: dict
```

`__post_init__` validates CPU-only dtypes, ranks, dimensions, finite values,
strictly positive occupation/norm/grid volume, contiguous primitive blocks,
unique non-negative `(occupied_state, auxiliary_channel)` keys, and Hermitian
overlap using the same absolute-plus-relative policy as response data.

The reader requires exactly:

```python
(
    "STERNHEIMER_SIAB_SOURCE_HEADER",
    "PRIMITIVE_BLOCKS",
    "SOURCE_METADATA",
    "OVERLAP_D",
    "OVERLAP_S",
    "PROVENANCE_JSON",
)
```

and header keys `format_version`, `n_source`, `n_primitive`, `n_blocks`, and
`grid_volume_bohr3`.

- [ ] **Step 4: Run source and response readers together**

```bash
cd SIAB/tests
python -m unittest -v test_read_sternheimer test_read_sternheimer_source
```

Expected: both modules pass.

- [ ] **Step 5: Commit source reader support**

```bash
git add SIAB/opt_orb_pytorch_dpsi/sternheimer_source_data.py \
        SIAB/opt_orb_pytorch_dpsi/IO/read_sternheimer_source.py \
        SIAB/tests/fixtures/sternheimer_source_v1.dat \
        SIAB/tests/test_read_sternheimer_source.py
GIT_AUTHOR_NAME='Codex' GIT_AUTHOR_EMAIL='codex@openai.com' \
GIT_COMMITTER_NAME='AroundPeking' \
GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit -m 'feat(siab): read Sternheimer source sidecars'
```

### Task 3: Enforce Strict Response/Source Pairing

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/sternheimer_source_pair.py`
- Create: `SIAB/tests/test_sternheimer_source_pair.py`

- [ ] **Step 1: Write failing successful-pair and mismatch tests**

Use the canonical fixtures and assert successful pairing maps every unique
response key `(occupied_state,auxiliary_channel)` to exactly one source row.
Parameterize failures for:

```text
primitive block or offset mismatch
relative S difference > 1e-13
maximum absolute S difference > 1e-14
cell, ecut, kernel, orbital, pseudo, ABS, PCA, whitening rank, or transform hash mismatch
missing or extra source key
duplicate source key
occupation difference > 1e-14
```

Changing only `abacus_commit`, `executable_sha256`, `mpi_ranks`, or
`omp_threads` must succeed and return those differences as warnings.

- [ ] **Step 2: Verify RED**

```bash
cd SIAB/tests
python -m unittest -v test_sternheimer_source_pair
```

Expected: pairing module is absent.

- [ ] **Step 3: Implement one explicit pairing object**

Define:

```python
@dataclass(frozen=True)
class SternheimerResponseSourcePair:
    response: SternheimerData
    source: SternheimerSourceData
    source_row_for_response_key: dict
    provenance_warnings: tuple
```

Implement `pair_response_and_source(response, source)` to validate the pair
and return this data class.

The physical provenance keys compared exactly are:

```python
(
    "auxiliary_basis_sha256", "cell_bohr", "ecut_ry", "kernel",
    "orbital_sha256", "pseudopotential_sha256", "spin_convention",
    "exx_pca_thr", "auxiliary_whitening", "raw_auxiliary_dimension",
    "whitened_auxiliary_rank", "discarded_auxiliary_rank",
    "coulomb_relative_threshold", "coulomb_transform_sha256",
)
```

Compare nested numeric lists elementwise. Require response channel IDs
contiguous from zero and equal to the whitened rank. Return warnings only for
the four allowed execution-provenance differences; every physical mismatch is
fatal.

- [ ] **Step 4: Pass pairing and reader tests**

```bash
cd SIAB/tests
python -m unittest -v test_read_sternheimer_source test_sternheimer_source_pair
```

Expected: all tests pass with explicit mismatch messages.

- [ ] **Step 5: Commit strict pairing**

```bash
git add SIAB/opt_orb_pytorch_dpsi/sternheimer_source_pair.py \
        SIAB/tests/test_sternheimer_source_pair.py
GIT_AUTHOR_NAME='Codex' GIT_AUTHOR_EMAIL='codex@openai.com' \
GIT_COMMITTER_NAME='AroundPeking' \
GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit -m 'feat(siab): pair response and source targets strictly'
```

### Task 4: Implement The Differentiable Projected-Pi Algebra

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/projected_pi.py`
- Create: `SIAB/tests/test_projected_pi.py`

- [ ] **Step 1: Write a failing direct-matrix test**

Construct one occupied state, two auxiliary channels, two frequencies, three
primitive functions, non-identity positive-definite complex Hermitian `S`,
complex `D` and `Q`, occupation `2.0`, and a real `3 x 2` candidate `C`.
Compute directly:

```python
g = C.mH @ S @ C
a = torch.zeros((2, 2), dtype=torch.complex128)
for occupied in range(1):
    a += 2.0 * (D[occupied] @ C) @ torch.linalg.inv(g) \
         @ (Q[frequency, occupied] @ C).mH
pi = a + a.mH
```

Assert candidate `A`, candidate `Pi`, primitive-reference `Pi`, per-frequency
squared error, and integrated relative loss match direct calculation to
`1e-13`.

Add tests proving:

- `Pi` is Hermitian;
- occupation is counted once, not squared;
- `A+A^dagger` cannot be replaced by `2*A` for complex non-Hermitian `A`;
- multiplying both `D` and `Q` for one occupied state by the same phase leaves
  all results unchanged;
- one coefficient gradient matches a centered `1e-6` finite difference;
- singular candidate `G_C`, incomplete response rows, inconsistent frequency
  weights, or non-positive primitive-reference norm is fatal.

- [ ] **Step 2: Verify RED**

```bash
cd SIAB/tests
python -m unittest -v test_projected_pi
```

Expected: projected-Pi module is absent.

- [ ] **Step 3: Implement tensor indexing and primitive pseudoinverse**

Define:

```python
@dataclass(frozen=True)
class ProjectedPiResult:
    loss: torch.Tensor
    frequency_ha: torch.Tensor
    frequency_weight: torch.Tensor
    frequency_loss: torch.Tensor
    candidate_pi: torch.Tensor
    reference_pi: torch.Tensor
    reference_rank: int
    max_candidate_condition: float

```

`ProjectedPiEvaluator.__init__` accepts `pair`,
`relative_rank_tolerance=1.0e-12`, and `condition_limit=1.0e12`.
`ProjectedPiEvaluator.evaluate(coefficients)` returns `ProjectedPiResult`.

At initialization, reshape source rows to `[occupied,a,e]` and response rows
to `[p,occupied,b,e]` only after proving complete rectangular key products.
Extract one occupation per occupied state and one GreenX weight per frequency
after consistency checks.

Build `S+` from `torch.linalg.eigh((S+S.mH)/2)` and retain
`lambda/lambda_max > relative_rank_tolerance`. For the candidate, reuse
`assemble_orbital_coefficients`, form `G_C=C.mH@S@C`, and use Cholesky solves.
Evaluate exactly:

```python
A_C[p] = sum_i f[i] * (D[i] @ C) @ solve(G_C, (Q[p,i] @ C).mH)
Pi_C[p] = A_C[p] + A_C[p].mH
A_B[p] = sum_i f[i] * D[i] @ S_plus @ Q[p,i].mH
Pi_B[p] = A_B[p] + A_B[p].mH
```

Compute the GreenX-weighted global relative Frobenius loss and local relative
losses. Keep all tensors complex128/float64 on CPU.

- [ ] **Step 4: Add equal-family aggregation**

Define `NormalizedPhysicalFamilyProjectedPi`. Its constructor accepts
`named_pairs`, `relative_rank_tolerance=1.0e-12`, and
`condition_limit=1.0e12`; `evaluate(coefficients)` returns the family sum and
the named per-family results.

Require nonempty unique physical names. Evaluate each family with its own
relative denominator and return their unweighted sum plus named results. There
is no ghost-family API.

- [ ] **Step 5: Pass algebra and legacy regressions**

```bash
cd SIAB/tests
python -m unittest -v \
  test_projected_pi test_sternheimer_spillage test_response_family_spillage
```

Expected: all pass and legacy spillage values are unchanged.

- [ ] **Step 6: Commit projected-Pi evaluation**

```bash
git add SIAB/opt_orb_pytorch_dpsi/projected_pi.py SIAB/tests/test_projected_pi.py
GIT_AUTHOR_NAME='Codex' GIT_AUTHOR_EMAIL='codex@openai.com' \
GIT_COMMITTER_NAME='AroundPeking' \
GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit -m 'feat(siab): evaluate source-aware projected Pi'
```

### Task 5: Add The Three-Basis Feasibility Analysis

**Files:**
- Create: `SIAB/example_H_sternheimer/projected_pi_loss/analyze_projected_pi.py`
- Create: `SIAB/example_H_sternheimer/projected_pi_loss/README.md`
- Create: `SIAB/tests/test_projected_pi_analysis.py`

- [ ] **Step 1: Write a failing synthetic CLI test**

Invoke the script on fixture pairs and three coefficient files. Require:

```text
projected_pi_ranking.json
projected_pi_ranking.md
projected_pi_frequency.pdf
projected_pi_frequency.png
```

The JSON contains schema version, every input SHA256, reader warnings,
zero-order-audit SHA256, thresholds `1e-10,1e-11,1e-12`, retained ranks,
per-family/total loss, frequency losses, gate booleans, and final decision
`pass` or `stop_galerkin_required`. A failed gate exits 2 after atomically
writing complete diagnostics.

- [ ] **Step 2: Verify RED**

```bash
cd SIAB/tests
python -m unittest -v test_projected_pi_analysis
```

Expected: analysis script is absent.

- [ ] **Step 3: Implement exact coefficient loading**

Construct H metadata with 25 radial primitives and:

```python
Nu = [3, 2, 0, 0, 0]
```

Use `IO.func_C.read_C_init`; do not parse coefficient text ad hoc. Require
exactly `3s2p` and no nonzero d/f/g columns. Verify fixed `1s,2s,1p` columns
of joint and guarded files match initial TZDP within `1e-12`.

- [ ] **Step 4: Implement ranking and threshold gates**

For every threshold evaluate H, H2, and the equal-family sum for:

```text
initial_tzdp
fixed_dzp_joint
low_frequency_guarded
```

The hard gate is:

```python
joint_total < initial_total
guarded_total < initial_total
max_relative_spread_for_each_basis_and_family <= 0.01
```

Report but do not gate joint-versus-guarded order. Plot one panel per family,
frequency in Hartree versus local relative Pi error, with distinct colors and
markers. Do not use SOS energy or H+ghost in the calculation.

- [ ] **Step 5: Require explicit zero-order audit files**

Require one H and one H2 JSON audit containing `status=pass`, exact grid and
state-count flags, eigenvalue/occupation/energy differences, and hashes of
the compared logs. Enforce producer-plan tolerances and reject missing, stale,
or failed audits.

- [ ] **Step 6: Pass the complete Python suite on `df_dcu`**

```bash
source /public/home/ghj/app/src/env_60_245_intel2021.sh
/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python -m unittest discover \
  -s SIAB/tests -p 'test_*.py' -v
```

Expected: all existing and new tests pass. Record the actual new total.

- [ ] **Step 7: Commit the analysis gate**

```bash
git add SIAB/example_H_sternheimer/projected_pi_loss/analyze_projected_pi.py \
        SIAB/example_H_sternheimer/projected_pi_loss/README.md \
        SIAB/tests/test_projected_pi_analysis.py
GIT_AUTHOR_NAME='Codex' GIT_AUTHOR_EMAIL='codex@openai.com' \
GIT_COMMITTER_NAME='AroundPeking' \
GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit -m 'analysis(siab): gate projected-Pi target ranking'
```

### Task 6: Run The Physical Ranking Gate And Record The Decision

**Files:**
- Create remotely: `/work1/ghj/sternheimer_abacus_tests/siab_projected_pi_feasibility_20260730/`
- Modify: `SIAB/example_H_sternheimer/fixed_dzp_tzdp_sos/README.md`
- Modify: `/Users/ghj/同步空间/AITP_project/sternheimer_abacus/sternheimer_siab_project/main.tex`

- [ ] **Step 1: Stage immutable inputs**

Use the two response paths at the top and source sidecars under:

```text
/work1/ghj/sternheimer_abacus_tests/siab_projected_pi_source_only_h_h2_20260730/H
/work1/ghj/sternheimer_abacus_tests/siab_projected_pi_source_only_h_h2_20260730/H2
```

Stage coefficients as:

```text
initial_tzdp_ORBITAL_RESULTS.txt
fixed_dzp_joint_ORBITAL_RESULTS.txt
low_frequency_guarded_ORBITAL_RESULTS.txt
```

Initial comes from
`siab_greedy_targets_source_h2_channel_mpi_prod_v1_20260726/Dojo-NC-SR/Orbitals_v2.0/H_TZDP/info/8/ORBITAL_RESULTS.txt`.
Old joint comes from
`siab_low_frequency_guard_old_joint_frequency_20260730/ORBITAL_RESULTS.txt`.
Sync guarded from
`/Users/ghj/同步空间/AITP_project/sternheimer_abacus/results/siab_h_low_frequency_guard_21440455_21440627_text/optimizer/ORBITAL_RESULTS.txt`.
Write a SHA256 manifest before execution.

- [ ] **Step 2: Submit one full-resource normal-node analysis job**

Use:

```text
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00
```

Set `OMP_NUM_THREADS=30`, `MKL_NUM_THREADS=30`, and run with
`/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python`. Print source commit,
Python/PyTorch versions, node, hashes, and `/usr/bin/time -v`.

- [ ] **Step 3: Inspect actual values, not only exit status**

Require finite losses, Hermitian errors below `1e-12`, candidate conditions
below `1e12`, stable ranks, all pairing checks passed, and exit code matching
the JSON decision. Report a table with H loss, H2 loss, equal-family total,
threshold spread, retained ranks, and existing held-out CP SOS binding for all
three bases.

- [ ] **Step 4: Follow exactly one decision branch**

If `decision=pass`, write only the next `pi_dpsi_joint` design. If
`decision=stop_galerkin_required`, record the failed ordering and begin a
separate Galerkin-Sternheimer design. Preserve raw JSON, Markdown, plots, job
output, source commit, and hashes in either case.

- [ ] **Step 5: Update and visually verify the research document**

Add the exact `D`, `Q`, `S`, `A(C)`, and `Pi(C)` equations; full-Coulomb and
no-extra-`V` convention; Hartree/Ry factor; pairing and zero-order audits; the
three-threshold table and frequency figure; and an explicit pass/stop
conclusion. Do not claim a basis improvement.

Compile `main.pdf`, use `pdftotext` to verify inserted numbers, render affected
pages, and inspect equations, table, legend, and captions for overflow.

- [ ] **Step 6: Commit the verified feasibility result**

Commit the example README, compact JSON/Markdown/plots under
`projected_pi_loss/results/`, and any analysis correction:

```bash
GIT_AUTHOR_NAME='Codex' GIT_AUTHOR_EMAIL='codex@openai.com' \
GIT_COMMITTER_NAME='AroundPeking' \
GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit -m 'analysis(siab): report projected-Pi feasibility gate'
```

Verify `git log -1` shows Codex as author and AroundPeking as committer.
