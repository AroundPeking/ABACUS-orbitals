# H2 Greedy-Basis Truncated Counterpoise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the anomalous high-band SOS-RPA contribution of the `13s11p10d5f4g` H basis cancels under counterpoise by comparing H2 and spin-polarized H+ghost at identical 120- and 160-band response cutoffs.

**Architecture:** Reuse the completed 334-band H2 and 167-band isolated-H reader-v1 producers. Run LibRPA for H2 at 120/160 bands and for H at the matched per-atom sizes 60/80. Produce H+ghost once with physical `nspin=2`, `nbands=160`, the same 20-Angstrom cell, 100-Ry grid, fixed 214-function-per-H ABFS, and historical binaries; run LibRPA at 120 bands and at its full 160-band space. A small tested parser computes zero-order, RPA-correlation, total CP binding, and the raw-minus-CP BSSE without changing the physical selector or using ghost data in optimization.

**Tech Stack:** Python 3 standard library, unittest, Bash, Slurm `normal`, ABACUS reader-v1, LibRPA, XeLaTeX.

---

### Task 1: Add a checked truncated-CP result parser

**Files:**
- Create: `SIAB/example_H_sternheimer/held_out_h2_sos_greedy_full/analyze_truncated_cp.py`
- Create: `SIAB/tests/test_analyze_truncated_cp.py`

- [ ] **Step 1: Write failing parser and energy-combination tests**

Create fixtures containing one ABACUS `rpa_lcao_exx(Ha):` block, one SCF
`!FINAL_ETOT_IS` line, and two LibRPA `Total EcRPA:` outputs. Require the
parser to reject missing or duplicate energy markers and verify

```python
def binding_kcal(monomer_ha, dimer_ha):
    return (2.0 * monomer_ha - dimer_ha) * 627.5094740631
```

for zero-order, correlation, and total energies independently.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest SIAB.tests.test_analyze_truncated_cp -v
```

Expected: import failure because `analyze_truncated_cp.py` does not exist.

- [ ] **Step 3: Implement the strict parser and JSON/table output**

Implement these public functions:

```python
def parse_unique_float(path, pattern, label): ...
def parse_abacus_energy(abacus_stdout, running_scf): ...
def parse_librpa_ec(path): ...
def combine_counterpoise(h2, isolated_h, ghost, hartree_to_kcal=627.5094740631): ...
```

The command-line interface accepts H2, isolated-H, and ghost ABACUS pairs plus
repeated
`--response H2_BAND:H2_LIBRPA:H_BAND:H_LIBRPA:GHOST_LIBRPA` entries. It writes
exact JSON plus a compact Markdown table. It must not infer, round, or silently
reuse missing energies.

- [ ] **Step 4: Run focused and full SIAB tests**

Run:

```bash
python -m unittest SIAB.tests.test_analyze_truncated_cp -v
python -m unittest discover -s SIAB/tests -v
```

Expected: focused tests and the complete SIAB suite pass.

- [ ] **Step 5: Commit the parser**

Commit with message:

```text
test(siab): add truncated counterpoise analyzer
```

---

### Task 2: Stage the physical H+ghost 160-band producer

**Files:**
- Create: `SIAB/example_H_sternheimer/held_out_h2_sos_greedy_full/run_ghost_truncated_cp.slurm`
- Modify: `SIAB/example_H_sternheimer/held_out_h2_sos_greedy_full/README.md`
- Modify: `SIAB/tests/test_analyze_truncated_cp.py`

- [ ] **Step 1: Add a failing static-contract test**

Require the Slurm script to contain the exact contracts:

```text
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
nbands 160
nspin 2
nupdown 1
n_bands_chi0 = 120
```

and reject `debug`, `nspin 1`, a changed Coulomb kernel, or a changed frequency
count.

- [ ] **Step 2: Run the static-contract test and verify RED**

Run:

```bash
python -m unittest SIAB.tests.test_analyze_truncated_cp.TruncatedGhostContractTest -v
```

Expected: failure because `run_ghost_truncated_cp.slurm` is absent.

- [ ] **Step 3: Implement the immutable production script**

The script copies `cases/H_ghost` into a fresh work directory, changes only
`nbands` from 334 to 160 and the output suffix, then validates all other INPUT
keys against the source template. It runs the exact historical ABACUS binary,
checks 320 `band_out` rows and total occupation 1, and runs the exact historical
LibRPA binary twice:

```text
lane 120: n_bands_chi0 = 120
lane 160: n_bands_chi0 omitted, therefore all 160 producer bands
```

Each lane has a separate `output_dir`, log, and SHA256 manifest. The script
must stop before LibRPA if ABACUS lacks the final SCF, EXX, reader-v1, Coulomb,
or wavefunction markers.

- [ ] **Step 4: Run focused and full tests**

Run the focused static-contract test and the complete SIAB suite. Expected:
all tests pass and the script contains no `debug` partition or fractional-spin
shortcut.

- [ ] **Step 5: Commit the production contract**

Commit with message:

```text
chore(siab): stage truncated full-basis CP diagnostic
```

---

### Task 3: Run and interpret the matched CP diagnostic

**Files:**
- Modify: `SIAB/example_H_sternheimer/held_out_h2_sos_greedy_full/README.md`
- Modify: `/Users/ghj/\u540c\u6b65\u7a7a\u95f4/AITP_project/sternheimer_abacus/sternheimer_siab_project/main.tex`

- [ ] **Step 1: Stage an immutable source closure on df_dcu**

Record `SOURCE_COMMIT`, `SOURCE_SHA256SUMS`, orbital/pseudopotential/ABFS hashes,
the ABACUS and LibRPA binary hashes, and the completed H2 producer root. Use
`normal`, one full 30-thread node, 110610 MB, and 24 hours.

- [ ] **Step 2: Submit and verify scheduler acceptance**

Submit the job, then check `squeue` and `scontrol show job`. Expected: accepted
by `normal`, no `BadConstraints`, exactly one node, 30 CPUs, and 110610 MB.

- [ ] **Step 3: Verify producer and both LibRPA lanes**

Require ABACUS `COMPLETED 0:0`, final SCF and EXX markers, 160 bands per spin,
occupation sum 1, reader-v1 files, and successful LibRPA completion for both
120 and 160 bands. Record wall time and peak RSS separately for ABACUS and each
LibRPA lane.

- [ ] **Step 4: Re-run H2 and H LibRPA at matching cutoffs and compute CP**

Use the completed H2 334-band producer with `n_bands_chi0=120` and `160`, and
the completed H 167-band producer with `n_bands_chi0=60` and `80`, then run
`analyze_truncated_cp.py`. Report:

```text
H2/ghost cutoff, H cutoff, D0_CP, Dc_CP, Dtotal_CP, Dtotal_raw, BSSE
```

Interpretation gate:

```text
If Dtotal_CP approaches 108.7 kcal/mol from 120 to 160 bands while raw diverges,
the high-band anomaly is predominantly BSSE.
If Dtotal_CP moves away or remains unstable, stop basis compression and audit
the SOS/LRI response and optimization metric before any full-band memory fix.
```

- [ ] **Step 5: Update and compile the TeX research note**

Add the raw band scan, overlap condition, absolute/per-frequency spillage,
failed full ghost jobs, matched CP table, confirmed interpretation, and the
next decision. Build with:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Require no unresolved references or new overfull boxes, render the affected
pages, and inspect them visually.

- [ ] **Step 6: Commit the measured result**

Commit repository documentation and durable parser output with message:

```text
docs(siab): record truncated full-basis CP diagnosis
```

## Execution outcome (2026-07-29)

- The strict analyzer and counterpoise combination tests were committed as
  `2ba2b3e4`; the production contract was committed as `41d4bb69`.
- Job `21421833` failed before ABACUS because the input parser rejected the
  standard `INPUT_PARAMETERS` header. Commit `9bdff344` moved the rewrite into
  the tested Python analyzer.
- Job `21421982` then exposed that a read-only source template remained
  read-only after copying. Commit `3ad69809` installs only the destination
  input as mode 0644; the immutable-source staging preflight and all 199 SIAB
  tests passed.
- Matching LibRPA postprocessing completed for isolated H: 60 and 80 bands
  give -0.018141011 and -0.019071348 Ha. Together with the completed H2 120/160
  lanes, the raw binding energies are 109.503973 and 113.591135 kcal/mol.
- Job `21422123` passed staging and converged SCF, but the 160-band two-spin
  H+ghost RPA-LRI producer was killed by the cgroup after 38:49 at 110,844,908
  KB. It wrote neither the final ABACUS marker nor reader-v1 data, so no CP
  result exists.
- The 334-to-160 band reduction changed the single-rank peak by only 1.47%.
  A 120-band producer retry is therefore not a meaningful continuation. Task
  3 steps 3--4 are blocked by the ABACUS producer memory layout, not by LibRPA
  or the counterpoise analyzer. Basis compression must remain paused until the
  empty-spin/two-spin RPA-LRI object is streamed, skipped when unoccupied, or
  genuinely distributed and validated on a smaller control.
