# SIAB Greedy Response-Shell Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a deterministic, H-only nested `s/p/d/f/g` response-basis sequence from Coulomb-orthonormal Delta-ST first-order-wavefunction spillage, then accept the smallest basis that reproduces matched Delta-ST RPA within `0.1 kcal/mol` while keeping counterpoise BSSE below `0.1 kcal/mol`.

**Architecture:** Extend the existing Sternheimer data and overlap-projector code with named target families, multi-target residual covariances, a fragment/ghost borrowing metric, and a deterministic one-shell greedy driver. Keep the fixed `1s,2s,1p` DZP core and the validated joint ST+dpsi optimizer. Freeze the complete spillage-ranked sequence before any H2 energy is evaluated; run all tests and physics producers on `df_dcu` `normal`.

**Tech Stack:** Python 3, PyTorch float64/complex128, NumPy, unittest, ABACUS Sternheimer-SIAB v1 producer, LibRPA, Slurm, JSON manifests, XeLaTeX.

---

### Task 1: Add named multi-target Sternheimer inputs

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/main.py`
- Create: `SIAB/opt_orb_pytorch_dpsi/sternheimer_targets.py`
- Create: `SIAB/tests/test_sternheimer_targets.py`
- Modify: `SIAB/tests/test_main_sternheimer.py`

- [ ] **Step 1: Write failing tests for backward-compatible target parsing**

Add tests that require a legacy one-path list to become family `default`, and
that require named entries to preserve family and role without accepting an H2
energy field:

```python
def test_parses_named_target_families(self):
    entries = parse_target_entries([
        {"path": "atom.dat", "family": "atom", "role": "physical"},
        {"path": "h3.dat", "family": "multicenter", "role": "physical"},
        {"path": "ghost.dat", "family": "fragment_ghost", "role": "ghost"},
    ])
    self.assertEqual(
        [(value.family, value.role) for value in entries],
        [("atom", "physical"), ("multicenter", "physical"),
         ("fragment_ghost", "ghost")],
    )

def test_rejects_energy_fields_in_target_entries(self):
    with self.assertRaisesRegex(ValueError, "RPA energy is not a selector input"):
        parse_target_entries([
            {"path": "h2.dat", "family": "atom", "rpa_binding": 108.72}
        ])
```

- [ ] **Step 2: Run the focused test remotely and verify RED**

Stage the worktree snapshot to df_dcu and run on one `normal` node:

```bash
python -m unittest discover -s SIAB/tests -p 'test_sternheimer_targets.py' -v
```

Expected: import failure for missing `sternheimer_targets`.

- [ ] **Step 3: Implement the immutable target-entry parser**

Create:

```python
@dataclass(frozen=True)
class SternheimerTargetEntry:
    path: Path
    family: str
    role: str

def parse_target_entries(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("sternheimer targets must be a nonempty list")
    entries = []
    for value in values:
        if isinstance(value, str):
            entries.append(SternheimerTargetEntry(Path(value), "default", "physical"))
            continue
        if not isinstance(value, dict):
            raise ValueError("sternheimer target entries must be paths or objects")
        forbidden = set(value) & {"rpa_binding", "h2_energy", "delta_st_energy"}
        if forbidden:
            raise ValueError("RPA energy is not a selector input")
        if set(value) != {"path", "family", "role"}:
            raise ValueError("named target requires path, family, and role")
        if value["role"] not in {"physical", "ghost"}:
            raise ValueError("target role must be physical or ghost")
        entries.append(SternheimerTargetEntry(
            Path(value["path"]), str(value["family"]), value["role"]
        ))
    return tuple(entries)
```

Update `_load_sternheimer_data` to return all parsed v1 target files while
preserving the one-string legacy behavior.

- [ ] **Step 4: Run focused and full SIAB tests remotely**

Run:

```bash
python -m unittest discover -s SIAB/tests -p 'test_sternheimer_targets.py' -v
python -m unittest discover -s SIAB/tests -p 'test_main_sternheimer.py' -v
python -m unittest discover -s SIAB/tests -v
```

Expected: all tests pass; record job ID and exact test count.

- [ ] **Step 5: Commit the target-input contract**

Commit with message:

```text
feat(siab): accept named Sternheimer target families
```

---

### Task 2: Generalize the residual spectrum to multiple targets and `l<=4`

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/sternheimer_spillage.py`
- Modify: `SIAB/tests/test_sternheimer_spillage.py`
- Create: `SIAB/tests/test_high_l_response_spectrum.py`

- [ ] **Step 1: Write failing high-l and multi-target spectrum tests**

Create synthetic complete f and g multiplets and two compatible target files.
Require summed covariance eigenvalues and reject one missing `m` channel:

```python
def test_many_target_spectrum_sums_coulomb_weighted_covariances(self):
    first = make_high_l_target(l=3, eigenvalues=[4.0, 1.0])
    second = make_high_l_target(l=3, eigenvalues=[2.0, 3.0])
    spectrum = radial_residual_spectrum_many(
        [first, second], c0(), fixed_specs(), "H", 3
    )
    torch.testing.assert_close(
        spectrum.eigenvalues,
        torch.tensor([6.0, 4.0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )

def test_rejects_incomplete_g_multiplet(self):
    data = make_high_l_target(l=4, omitted_m=4)
    with self.assertRaisesRegex(ValueError, "expected .* got"):
        radial_residual_spectrum_many([data], c0(), fixed_specs(), "H", 4)
```

- [ ] **Step 2: Run the focused test remotely and verify RED**

Run:

```bash
python -m unittest discover -s SIAB/tests -p 'test_high_l_response_spectrum.py' -v
```

Expected: missing `radial_residual_spectrum_many`.

- [ ] **Step 3: Extract reusable covariance terms and aggregate targets**

Refactor the singleton implementation around:

```python
@dataclass(frozen=True)
class RadialResidualTerms:
    projected_overlap: torch.Tensor
    covariance: torch.Tensor
    magnetic_channels: tuple

def radial_residual_spectrum_many(
    data_items, c0, fixed_specs, element, l,
    relative_rank_tolerance=1.0e-4,
    magnetic_overlap_tolerance=1.0e-4,
):
    if not data_items:
        raise ValueError("data_items must be nonempty")
    terms = [
        _radial_residual_terms(
            data, c0, fixed_specs, element, l,
            magnetic_overlap_tolerance,
        )
        for data in data_items
    ]
    _validate_compatible_radial_terms(terms)
    overlap = sum(value.projected_overlap for value in terms) / len(terms)
    covariance = sum(value.covariance for value in terms)
    return _diagonalize_radial_terms(
        overlap, covariance, element, l, terms[0].magnetic_channels,
        relative_rank_tolerance,
    )
```

Keep `radial_residual_spectrum` as a singleton wrapper so previous results and
tests remain unchanged.

- [ ] **Step 4: Run focused and full tests remotely**

Expected: singleton d spectra remain byte-for-byte equivalent within declared
float tolerance; new f/g tests pass.

- [ ] **Step 5: Commit the generalized spectrum**

Commit with message:

```text
feat(siab): aggregate high-l response spectra
```

---

### Task 3: Add family-normalized spillage and ghost borrowing

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/sternheimer_spillage.py`
- Create: `SIAB/opt_orb_pytorch_dpsi/response_selection.py`
- Create: `SIAB/tests/test_response_selection.py`

- [ ] **Step 1: Write failing tests for family normalization and borrowing**

Use a two-center synthetic target where the second center lowers residual only
when included:

```python
def test_borrowing_gap_is_own_minus_all_center_residual(self):
    family = make_two_center_fragment_family(own_loss=0.60, all_loss=0.20)
    value = borrowing_gap(family, coefficients())
    self.assertAlmostEqual(value, 0.40, places=13)

def test_family_loss_is_normalized_to_fixed_dzp(self):
    family = make_family(current_residual=2.0, dzp_residual=8.0)
    self.assertAlmostEqual(normalized_family_loss(family), 0.25, places=13)
```

- [ ] **Step 2: Verify RED remotely**

Expected: import failure for `response_selection`.

- [ ] **Step 3: Add explicit-column projector evaluation**

Add a low-level projector API that shares the validated Cholesky and roundoff
checks with `SternheimerSpillage`:

```python
def evaluate_spillage_for_columns(data, c, include):
    assembled, labels = assemble_orbital_coefficients(data, c)
    indices = tuple(i for i, label in enumerate(labels) if include(label))
    if not indices:
        raise ValueError("selected projector contains no orbital columns")
    return _evaluate_projector(data, assembled[:, indices], labels, indices)
```

Implement `own` as `label.atom_index == real_atom_index` and `all` as every
real or ghost block. Clamp only roundoff-negative borrowing values; reject
materially negative values as inconsistent target construction.

- [ ] **Step 4: Implement normalized target families**

Create immutable `ResponseTargetFamily` and functions:

```python
@dataclass(frozen=True)
class ResponseTargetFamily:
    name: str
    data: tuple
    role: str
    real_atom_index: int | None = None

def normalized_family_loss(family, current, fixed_dzp):
    def full_residual(data, coefficients):
        return evaluate_spillage_for_columns(
            data, coefficients, include=lambda label: True
        ).weighted_residual
    numerator = sum(full_residual(data, current) for data in family.data)
    denominator = sum(full_residual(data, fixed_dzp) for data in family.data)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("fixed-DZP family residual must be positive")
    return numerator / denominator
```

- [ ] **Step 5: Run focused and full tests remotely, then commit**

Commit with message:

```text
feat(siab): measure fragment response borrowing
```

---

### Task 4: Implement deterministic one-shell scoring

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/response_selection.py`
- Modify: `SIAB/tests/test_response_selection.py`

- [ ] **Step 1: Write failing score, cost, and tie-break tests**

```python
def test_score_is_gain_plus_balance_reduction_per_ao_function(self):
    candidate = CandidateGain(l=3, mode=0, atom=0.9, multicenter=0.4,
                              balance=-0.1)
    self.assertAlmostEqual(score_candidate(candidate), 1.2 / 7.0, places=14)

def test_selector_prefers_more_gain_per_actual_ao_function(self):
    d = CandidateGain(l=2, mode=0, atom=0.50, multicenter=0.0, balance=0.0)
    f = CandidateGain(l=3, mode=0, atom=0.84, multicenter=0.0, balance=0.0)
    self.assertEqual(select_best_candidate([d, f]).l, 3)

def test_tie_break_is_lower_cost_then_l_then_mode(self):
    candidates = equal_score_candidates()
    self.assertEqual(select_best_candidate(candidates).key, (1, 0))
```

- [ ] **Step 2: Verify RED remotely**

Expected: missing `CandidateGain` and selector functions.

- [ ] **Step 3: Implement immutable scores and deterministic selection**

```python
@dataclass(frozen=True)
class CandidateGain:
    l: int
    mode: int
    atom: float
    multicenter: float
    balance: float

    @property
    def cost(self):
        return 2 * self.l + 1

    @property
    def key(self):
        return (self.l, self.mode)

def score_candidate(value):
    physical = value.atom + value.multicenter
    if physical <= 1.0e-14:
        return float("-inf")
    return (physical + value.balance) / value.cost

def select_best_candidate(values):
    admissible = [value for value in values if score_candidate(value) > 0.0]
    if not admissible:
        raise RuntimeError("no admissible positive-score response shell remains")
    return min(
        admissible,
        key=lambda value: (-score_candidate(value), value.cost,
                           value.l, value.mode),
    )
```

- [ ] **Step 4: Add the real candidate evaluator**

For each first residual eigenmode in `l=0,...,4`, append the mode to a cloned
coefficient set, evaluate all three normalized gains, and return the full list
including deterministic rejection reasons.

- [ ] **Step 5: Run tests remotely and commit**

Commit with message:

```text
feat(siab): rank response shells by balanced AO gain
```

---

### Task 5: Build the nested-sequence driver and immutable manifests

**Files:**
- Create: `SIAB/example_H_sternheimer/greedy_response_selection/select_response_shells.py`
- Create: `SIAB/example_H_sternheimer/greedy_response_selection/selection_config.json`
- Create: `SIAB/example_H_sternheimer/greedy_response_selection/README.md`
- Create: `SIAB/tests/test_select_response_shells.py`

- [ ] **Step 1: Write failing CLI and manifest tests**

Require a three-step synthetic run to write byte-identical manifests twice,
preserve fixed columns, and omit H2 energy keys:

```python
def test_frozen_manifest_is_deterministic_and_contains_no_energy(self):
    first = run_fixture(seed=20260720)
    second = run_fixture(seed=20260720)
    self.assertEqual(first.read_bytes(), second.read_bytes())
    text = first.read_text()
    self.assertNotIn("h2_energy", text)
    self.assertNotIn("rpa_binding", text)

def test_each_step_keeps_fixed_dzp_columns_bitwise_equal(self):
    manifest = run_fixture(seed=20260720)
    for step in load_steps(manifest):
        assert_fixed_columns_equal(BASELINE, step.coefficients)
```

- [ ] **Step 2: Verify RED remotely**

Expected: missing selector CLI.

- [ ] **Step 3: Implement the one-step-at-a-time driver**

The checked-in config fixes:

```json
{
  "format_version": 1,
  "element": "H",
  "max_l": 4,
  "relative_rank_tolerance": 0.0001,
  "magnetic_overlap_tolerance": 0.0001,
  "global_capture": 0.999,
  "per_l_residual_limit": 0.01,
  "seed": 20260720,
  "fixed_orbitals": [
    {"element": "H", "l": 0, "zeta": 1},
    {"element": "H", "l": 0, "zeta": 2},
    {"element": "H", "l": 1, "zeta": 1}
  ]
}
```

Write each candidate score before selecting, canonicalize the chosen column,
append it without changing earlier columns, and stop only at the specified
spillage condition or explicit no-candidate failure.

- [ ] **Step 4: Connect each accepted step to joint ST+dpsi optimization**

Generate a step-local SIAB `INPUT` containing the selected `Nu.H`, all named
Sternheimer target files, the existing H3 origin/dpsi files, fixed DZP list,
seed, and loss gates. Invoke the existing optimizer as a checked subprocess and
reject a step without `ORBITAL_RESULTS.txt`, `Spillage.dat`, and `ORBITAL_1U.dat`.

- [ ] **Step 5: Run focused and full tests remotely and commit**

Commit with message:

```text
feat(siab): generate frozen response-basis sequences
```

---

### Task 6: Produce and validate `l<=4` atom/H3/ghost targets

**Files:**
- Create: `SIAB/example_H_sternheimer/greedy_response_selection/producer_atom/INPUT`
- Create: `SIAB/example_H_sternheimer/greedy_response_selection/producer_h3/INPUT`
- Create: `SIAB/example_H_sternheimer/greedy_response_selection/producer_fragment_ghost/INPUT`
- Create: `SIAB/example_H_sternheimer/greedy_response_selection/run_targets.slurm`
- Modify: `SIAB/tests/test_h_sternheimer_smoke.py`

- [ ] **Step 1: Write failing static producer-contract tests**

Require every producer input to contain:

```text
sternheimer_siab_output 1
sternheimer_siab_lmax 4
sternheimer_nfreq 16
exx_pca_threshold 10
rpa_ccp_rmesh_times 5
```

Require `normal`, one MPI rank per frequency node, 30 OpenMP threads, 110610 MB
per node, immutable binary/source hashes, output non-reuse, and complete target
validation. Every STRU supplies the explicit fixed
`H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs` with SHA256
`d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f`;
the product-PCA auxiliary space is disabled by the threshold above.

- [ ] **Step 2: Verify RED on df_dcu `normal`**

Expected: missing three producer directories.

- [ ] **Step 3: Add immutable producer inputs and validators**

Use the existing validated ABACUS Sternheimer-SIAB build first. The generic
producer is expected to emit `25 * sum(2*l+1, l=0..4) = 625` primitive columns
per H atom. The explicit fixed ABS must retain nonzero f/g perturbation rows
after full-Coulomb orthonormalization; reject a target whose reference metadata
contains only s/p/d auxiliary channels. If the existing binary rejects `l=4`,
stop and open a separate ABACUS producer bugfix stage; do not silently reduce
`lmax`.

- [ ] **Step 4: Submit target jobs and verify artifacts**

Run all jobs on `df_dcu` `normal`. Verify scheduler `COMPLETED 0:0`, complete
`m` multiplets, 16 frequencies, Coulomb-orthonormal perturbation provenance,
matrix sizes, hashes, wall time, and peak memory before selector use.

- [ ] **Step 5: Commit target contracts and evidence**

Commit with message:

```text
test(siab): add high-l response target campaign
```

---

### Task 7: Generate and inspect the frozen basis sequence

**Files:**
- Modify: `SIAB/example_H_sternheimer/greedy_response_selection/README.md`
- Create: `SIAB/example_H_sternheimer/greedy_response_selection/plot_sequence.py`
- Create: `SIAB/tests/test_plot_response_sequence.py`

- [ ] **Step 1: Run the selector on df_dcu `normal`**

Use one full node for each SIAB optimization step. Record the selected order,
per-step `Nu.H`, total AO count, atom/H3 residuals, borrowing gap, condition
numbers, DFT/dpsi gates, wall time, and hashes.

- [ ] **Step 2: Add a tested radial-orbital and spectrum plotter**

The test requires every selected shell and residual spectrum to appear once,
stable axis labels, and no path-dependent title. Generate PNG and CSV data for
the project TeX.

- [ ] **Step 3: Review physical and numerical gates**

Reject oscillatory radial functions, material overlap-condition failures,
changed fixed DZP columns, incomplete angular multiplets, or increased final
borrowing relative to fixed DZP. Do not run H2 for a rejected sequence.

- [ ] **Step 4: Commit the frozen sequence metadata**

Commit only compact coefficients, manifests, plots, and hashes; keep large
producer matrices on the immutable remote path. Use message:

```text
data(siab): freeze greedy H response-basis sequence
```

---

### Task 8: Run fixed-ABS SOS, counterpoise, and Delta-ST gates

**Files:**
- Create: `SIAB/example_H_sternheimer/greedy_response_selection/rpa_gate/run_rpa_gate.slurm`
- Create: `SIAB/example_H_sternheimer/greedy_response_selection/rpa_gate/summarize_gate.py`
- Modify: `SIAB/example_H_sternheimer/greedy_response_selection/README.md`
- Modify: `SIAB/example_H_sternheimer/README.md`
- Modify: `../sternheimer_siab_project/main.tex`

- [ ] **Step 1: Write failing static and arithmetic gate tests**

Require H2/H/H+ghost for every frozen candidate, all LCAO bands, the same fixed
214-function-per-H ABS, 20-A cell, 0.74085-A bond, 100 Ry, 16 frequencies,
full Coulomb, and immutable ABACUS/LibRPA binaries. Test exact formulas:

```python
bsse = d_raw - d_cp
passes = abs(d_cp - d_delta_st) < 0.1 and abs(bsse) < 0.1
```

- [ ] **Step 2: Verify RED remotely**

Expected: missing gate scripts and case matrix.

- [ ] **Step 3: Add and run the production array on `normal`**

Use one full node per independent ABACUS-to-LibRPA case. Validate SCF,
band/spin counts, integer occupations, fixed ABS hash, full-Coulomb marker, and
LibRPA completion before accepting any energy.

- [ ] **Step 4: Select the first passing frozen candidate**

The summarizer reads the frozen sequence order and production outputs. It must
not reorder candidates by energy. If none passes, report failure and preserve
all results without changing selector weights.

- [ ] **Step 5: Update documentation and relabel the historical SOS line**

Change the figure label from `independent SOS reference` to
`historical raw SOS: different orbital, no CP`. Add the nested-sequence,
spillage, BSSE, CP, and matched Delta-ST tables before the final Ecut figure.
Compile XeLaTeX twice and inspect the rendered pages.

- [ ] **Step 6: Run full regression, verify, and commit**

Run the full SIAB suite on `normal`, verify all production job states and
hashes, run `git diff --check`, and commit with message:

```text
docs(siab): record greedy response-basis RPA gate
```
