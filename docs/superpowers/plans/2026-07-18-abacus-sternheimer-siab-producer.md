# ABACUS Sternheimer-SIAB Producer And H Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the validated ABACUS Delta-Sternheimer path emit compact, versioned H-atom first-order-wavefunction overlaps for SIAB, then compare ST-only and constrained H-TZDP optimizations by a held-out H2 SOS-RPA calculation.

**Architecture:** The ABACUS producer is a separate opt-in output path at the point where each reconstructed Delta-ST wavefunction is available. It evaluates `n_rho`, `<Y_rho|B_e>`, and `<B_e|B_e'>` on the existing uniform real-space grid with the same `DeltaOmega`, writes a strict v1 file, and includes complete provenance. Production data and SIAB optimization run on `df_dcu`; LibRPA performs the held-out full-Coulomb SOS-RPA validation.

**Tech Stack:** ABACUS C++14, MPI/OpenMP, GoogleTest, uniform real-space grid, SIAB/PyTorch, LibRPA, SLURM `normal` partition.

---

## Repository And Resource Guards

ABACUS work starts from the active Sternheimer branch whose validated standard-ST routing commit is:

```text
19ab21e01d02cc805604ed77a6e269af698fdd1d
```

and whose strict Delta-ST base is:

```text
564d45ec20aeb5bd7ce87ed16dd8107c844a8604
```

Before editing or submitting, record:

```bash
git status --short --branch
git rev-parse HEAD
git merge-base --is-ancestor 564d45ec20aeb5bd7ce87ed16dd8107c844a8604 HEAD
git show -s --format='%H %s' HEAD
sha256sum "$ABACUS_EXE"
```

Every compute job uses `#SBATCH -p normal`, the maximum allowed wall time and `110G` per node. Never submit these jobs to `debug`. Frequency parallelism uses one MPI rank per frequency group and fills each node with OpenMP threads; `OMP_NUM_THREADS` is set from the node's physical-core allocation. Every script prints SLURM variables, `lscpu`, executable hash, source commit, and INPUT parameters before launch.

## File Map

Create in the active ABACUS Sternheimer Git worktree:

- `source/source_lcao/module_ri/sternheimer_siab_data.h`: v1 row/block/provenance structures.
- `source/source_lcao/module_ri/sternheimer_siab_overlap.h`: pure uniform-grid overlap API.
- `source/source_lcao/module_ri/sternheimer_siab_overlap.cpp`: `DeltaOmega`-weighted reductions.
- `source/source_lcao/module_ri/sternheimer_siab_writer.h`: strict v1 writer API.
- `source/source_lcao/module_ri/sternheimer_siab_writer.cpp`: deterministic tagged serialization.
- `source/source_lcao/module_ri/test/test_sternheimer_siab_overlap.cpp`: analytic grid tests.
- `source/source_lcao/module_ri/test/test_sternheimer_siab_writer.cpp`: parser-contract golden test.

Modify:

- `source/source_io/module_parameter/input_parameter.h`: add `out_sternheimer_siab`.
- `source/source_io/module_parameter/read_input_item_output.cpp`: register and validate output flag.
- `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp`: collect rows at reconstructed-wavefunction scope and write once after all frequency groups are gathered.
- `source/source_lcao/module_ri/CMakeLists.txt`: compile the producer sources.
- `source/source_lcao/module_ri/test/CMakeLists.txt`: register the three unit-test targets.

No LibRPA source is changed for the producer. SIAB reader work follows `2026-07-18-siab-sternheimer-loss.md`.

### Task 1: Pin the Uniform-Grid Overlap Convention

**Files:**
- Create: `source/source_lcao/module_ri/sternheimer_siab_data.h`
- Create: `source/source_lcao/module_ri/sternheimer_siab_overlap.h`
- Create: `source/source_lcao/module_ri/test/test_sternheimer_siab_overlap.cpp`

- [x] **Step 1: Write analytic overlap tests first**

For a local four-point real grid with `DeltaOmega=0.25`, use:

```cpp
const std::vector<std::complex<double>> y = {{1.0, 1.0}, {2.0, 0.0}, {0.0, -1.0}, {1.0, 0.0}};
const std::vector<std::vector<std::complex<double>>> b = {
    {{1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}},
    {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}},
};
```

Assert:

```cpp
EXPECT_NEAR(norm(y, 0.25), 2.0, 1.0e-14);
EXPECT_EQ(overlap_q(y, b, 0.25).size(), 2);
EXPECT_NEAR(overlap_q(y, b, 0.25)[0].real(), 0.25, 1.0e-14);
EXPECT_NEAR(overlap_q(y, b, 0.25)[0].imag(), 0.00, 1.0e-14);
EXPECT_NEAR(overlap_q(y, b, 0.25)[1].real(), 0.75, 1.0e-14);
EXPECT_NEAR(overlap_q(y, b, 0.25)[1].imag(), 0.00, 1.0e-14);
```

The test for `overlap_s(b,0.25)` asserts a Hermitian diagonal matrix with both diagonal values `0.5`.

- [x] **Step 2: Run the new test target and confirm link failure**

```bash
cmake --build build-sternheimer-siab -j 8 --target test_sternheimer_siab_overlap
ctest --test-dir build-sternheimer-siab -R sternheimer_siab_overlap --output-on-failure
```

Expected before implementation: undefined overlap API or missing target.

- [x] **Step 3: Define explicit data types and pure APIs**

```cpp
namespace module_ri::sternheimer_siab
{
struct PrimitiveBlock
{
    std::string element;
    int atom_index;
    int l;
    int m;
    int n_primitive;
    int offset;
};

struct ReferenceRow
{
    int occupied_state;
    int auxiliary_channel;
    double frequency_ha;
    double occupation;
    double frequency_weight;
    double norm;
    std::vector<std::complex<double>> q;
};

double norm(const std::vector<std::complex<double>>& y, double delta_omega);
std::vector<std::complex<double>> overlap_q(
    const std::vector<std::complex<double>>& y,
    const std::vector<std::vector<std::complex<double>>>& primitives,
    double delta_omega);
std::vector<std::complex<double>> overlap_s(
    const std::vector<std::vector<std::complex<double>>>& primitives,
    double delta_omega);
}
```

All three APIs compute `conj(left)*right*DeltaOmega`. In the final frequency-MPI path each owner already holds a complete reconstructed response, whereas the primitive FFT output is distributed over PW z slabs. The production wrapper therefore allgathers the primitive slabs once, evaluates `norm` and `Q` locally on the owning rank without an overlap `Allreduce`, and gathers variable row payloads to rank 0. Reducing the complete owner response would double count it. It must not transform the reconstructed response between PW and real-space representations.
An MPI rank with a zero-length local FFT slab remains valid during primitive assembly and contributes an empty slab; it must not exit before the collective. A globally empty or incompletely covered grid remains invalid at the production-wrapper level.

- [x] **Step 4: Implement and pass the analytic tests**

```bash
cmake --build build-sternheimer-siab -j 8 --target test_sternheimer_siab_overlap
ctest --test-dir build-sternheimer-siab -R sternheimer_siab_overlap --output-on-failure
```

Expected: one test passes.

- [x] **Step 5: Commit only overlap/data-model work**

```bash
git add source/source_lcao/module_ri/sternheimer_siab_data.h \
        source/source_lcao/module_ri/sternheimer_siab_overlap.h \
        source/source_lcao/module_ri/sternheimer_siab_overlap.cpp \
        source/source_lcao/module_ri/test/test_sternheimer_siab_overlap.cpp \
        source/source_lcao/module_ri/CMakeLists.txt \
        source/source_lcao/module_ri/test/CMakeLists.txt
GIT_COMMITTER_NAME='AroundPeking' GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit --author='Codex <codex@openai.com>' -m 'test(sternheimer): pin SIAB grid overlaps'
```

### Task 2: Generate the Same Primitive Basis Used by H-TZDP

**Files:**
- Modify: `source/source_io/module_bessel/numerical_basis.h`
- Modify: `source/source_io/module_bessel/numerical_basis.cpp`
- Create: `source/source_lcao/module_ri/test/test_sternheimer_siab_primitives.cpp`

- [x] **Step 1: Write a PW-versus-grid primitive regression**

Extract the existing numerical-Bessel construction behind `Numerical_Basis::cal_overlap_Q` into a public read-only helper. For every explicit `(atom,l,m,ie)`, first construct the exact reciprocal-space primitive coefficients already used by `cal_overlap_Q`, then call `PW_Basis_K::recip2real` once and divide the local real-space values by `sqrt(ucell.omega)`. The latter factor converts ABACUS's FFT convention to a physical grid function satisfying

```text
DeltaOmega * sum_r conj(B_e(r)) B_e'(r)
= sum_G conj(B_e(G)) B_e'(G)
```

on the represented PW subspace. For a Gamma-only H atom fixture, assert that direct grid integration of a test PW wavefunction with each primitive agrees with the existing reciprocal-space overlap to `1e-10` absolute and relative tolerance at `ecut=25 Ry`.

- [x] **Step 2: Confirm the regression fails before extraction**

```bash
cmake --build build-sternheimer-siab -j 8 --target MODULE_RI_sternheimer_siab_primitives_test
ctest --test-dir build-sternheimer-siab -R sternheimer_siab_primitives --output-on-failure
```

- [x] **Step 3: Add the public helper without duplicating radial definitions**

The new interface returns local-grid blocks ordered by:

```text
element order -> atom index -> l -> m=-l..l -> primitive index ie=0..Ne-1
```

Each block records `n_primitive=Ne` and a cumulative offset. Reuse the current spherical-Bessel cutoff, `Ecut`, reciprocal normalization, real spherical-harmonic convention, structure-factor atom phase, PW cutoff, FFT distribution, and `DeltaOmega`. Do not reimplement Bessel roots, radial transforms, spherical-harmonic phases, or FFT normalization in `module_ri`. The helper returns only the current rank's local real-space slab; Task 4 reassembles those slabs into the full primitive grid once because the frequency owner holds a full response grid.

- [x] **Step 4: Run primitive and overlap tests**

```bash
cmake --build build-sternheimer-siab -j 8 --target \
  MODULE_RI_sternheimer_siab_overlap_test \
  MODULE_RI_sternheimer_siab_primitives_test
ctest --test-dir build-sternheimer-siab -R 'sternheimer_siab_(overlap|primitives)' --output-on-failure
```

Expected: PW and grid overlaps match within `1e-10`.

- [x] **Step 5: Commit the primitive-grid bridge**

```bash
git add source/source_io/module_bessel/numerical_basis.h \
        source/source_io/module_bessel/numerical_basis.cpp \
        source/source_lcao/module_ri/test/test_sternheimer_siab_primitives.cpp
GIT_COMMITTER_NAME='AroundPeking' GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit --author='Codex <codex@openai.com>' -m 'feat(sternheimer): expose SIAB primitive grid values'
```

### Task 3: Add the Strict v1 Writer and Input Flag

**Files:**
- Create: `source/source_lcao/module_ri/sternheimer_siab_writer.h`
- Create: `source/source_lcao/module_ri/sternheimer_siab_writer.cpp`
- Create: `source/source_lcao/module_ri/test/test_sternheimer_siab_writer.cpp`
- Modify: `source/source_io/module_parameter/input_parameter.h`
- Modify: `source/source_io/module_parameter/read_input_item_output.cpp`

- [x] **Step 1: Write the writer golden test**

Construct two `ReferenceRow` values and the four-dimensional fixture from the SIAB plan. Write to a temporary file and assert all six tagged sections occur once, row counts are exact, floating output uses deterministic shortest exact round-trip formatting, provenance JSON is valid, and the file text matches `SIAB/tests/fixtures/sternheimer_matrix_v1.dat`. The cell is exactly nine finite row-major Bohr components with nonzero determinant; valid UTF-8 is preserved and malformed UTF-8 is rejected before touching the temporary file.

- [x] **Step 2: Run and confirm writer failure**

```bash
cmake --build build-sternheimer-siab -j 8 --target test_sternheimer_siab_writer
ctest --test-dir build-sternheimer-siab -R sternheimer_siab_writer --output-on-failure
```

- [x] **Step 3: Implement deterministic writing and atomic replacement**

Expose:

```cpp
void write_v1(
    const std::string& path,
    double grid_volume_bohr3,
    const std::vector<PrimitiveBlock>& blocks,
    const std::vector<ReferenceRow>& rows,
    const std::vector<std::complex<double>>& overlap_s,
    const Provenance& provenance);
```

Validate dimensions and Hermiticity before writing. Write `path + ".tmp"`, flush/close successfully, then rename to `path`; rank 0 alone writes after gathered arrays are complete.

- [x] **Step 4: Register an output-only ABACUS parameter**

Add:

```text
out_sternheimer_siab 0
```

Allowed values are `0` and `1`. Value `1` requires `basis_type=lcao`, the Sternheimer calculation path, and loaded LCAO basis metadata; invalid combinations stop during input validation. It is independent of the existing `rpa` switch and writes `OUT.ABACUS/sternheimer_matrix.dat`.

- [x] **Step 5: Run writer/input tests and commit**

```bash
cmake --build build-sternheimer-siab -j 8 --target test_sternheimer_siab_writer
ctest --test-dir build-sternheimer-siab -R 'sternheimer_siab_(overlap|primitives|writer)' --output-on-failure
git add source/source_lcao/module_ri/sternheimer_siab_writer.h \
        source/source_lcao/module_ri/sternheimer_siab_writer.cpp \
        source/source_lcao/module_ri/test/test_sternheimer_siab_writer.cpp \
        source/source_io/module_parameter/input_parameter.h \
        source/source_io/module_parameter/read_input_item_output.cpp
GIT_COMMITTER_NAME='AroundPeking' GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit --author='Codex <codex@openai.com>' -m 'feat(sternheimer): write versioned SIAB targets'
```

### Task 4: Connect the Writer to Delta-ST and Verify MPI Identity

**Files:**
- Modify: `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp`
- Create: `source/source_lcao/module_ri/test/test_sternheimer_siab_mpi.cpp`

- [x] **Step 1: Add a serial-versus-frequency-MPI regression**

At the MPI assembly layer, use fixed complete response vectors and two frequency owners. Parse the serial and gathered v1 files and assert:

```text
same primitive blocks and metadata exactly
max_abs(norm_serial - norm_mpi) < 1e-12
max_abs(Q_serial - Q_mpi) < 1e-12
max_abs(S_serial - S_mpi) < 1e-12
```

The test must fail if rank-local rows are missing, duplicated, double-counted, or ordered by arrival instead of `(occupied_state,auxiliary_channel,frequency_index)`.

For a real H calculation, a raw row of `Q` may acquire the arbitrary global phase of the occupied KS orbital and reconstructed first-order wavefunction when the total MPI layout changes. Therefore the end-to-end comparison additionally reports row-wise phase-aligned `Q` and the phase-invariant row projector `Q^dagger Q`; it must not require raw `Q` equality before gauge alignment. Metadata and `S` remain directly comparable.

- [x] **Step 2: Insert collection at the full reconstructed response**

At the scope containing `response.response.reconstructed_wavefunction`, calculate one `ReferenceRow` per `(occupied_state, auxiliary_channel, ifreq)`. Use the complete reconstructed Delta-ST wavefunction, not only the complementary correction. Set:

```text
norm = <Y|Y>_grid
q[e] = <Y|B_e>_grid
occupation = the same spin-resolved occupation used by chi0 assembly
frequency_weight = the same GreenX minimax quadrature weight used by RPA
frequency_ha = the actual imaginary frequency in Hartree
```

Build `S_B` once. Sort gathered rows by the tuple above before writing. The output contains no Coulomb matrix and no raw real-space wavefunction.

- [x] **Step 3: Record complete provenance**

Write source commit, executable SHA256, full-Coulomb kernel label, cell vectors in bohr, `ecut_ry`, PP SHA256, initial orbital SHA256, ABFS SHA256, `exx_pca_thr`, `sternheimer_nfreq`, frequency list, spin convention, MPI ranks, and OMP threads. Store `DeltaOmega` as the v1 header's `grid_volume_bohr3`. Reject an empty Git hash or missing file hash.

- [x] **Step 4: Build with a private immutable executable**

```bash
cmake -S . -B build-sternheimer-siab -DCMAKE_CXX_FLAGS='-O3 -g' -DDEBUG_INFO=ON
cmake --build build-sternheimer-siab -j 32 --target abacus_3p
cp build-sternheimer-siab/abacus_3p build-sternheimer-siab/abacus-$(git rev-parse --short=12 HEAD)
sha256sum build-sternheimer-siab/abacus-$(git rev-parse --short=12 HEAD)
```

Do not overwrite an executable used by a running job.

- [x] **Step 5: Run MPI identity checks on `normal` and commit**

Submit the tiny two-frequency cases to `normal`, inspect `sacct`, `OUT.ABACUS/running_scf.log`, and both producer files. Require zero ABACUS exit codes, complete row counts, exact metadata and `S`, converged response equations, and report both row-wise phase-aligned `Q` and the phase-invariant row projector. The synthetic fixed-response MPI regression retains the strict `1e-12` criterion.

```bash
git add source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp \
        source/source_lcao/module_ri/test/test_sternheimer_siab_mpi.cpp
GIT_COMMITTER_NAME='AroundPeking' GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit --author='Codex <codex@openai.com>' -m 'feat(sternheimer): export MPI-complete SIAB targets'
```

**Completed evidence (2026-07-18):** ABACUS commit `80a606f57a2610bc2532468661b687b01f58074c` was built on `df_dcu` from an isolated source snapshot with the established `-O3 -g`, `DEBUG_INFO=ON` profile. Job `21310975` passed all seven SIAB overlap/writer/provenance/MPI/primitive tests and produced the immutable executable
`/public/home/ghj/app/src/abacus-sternheimer-siab-task4-80a606f57a26/build-task4-o3-debug-info/abacus-80a606f57a26`, SHA256 `2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d`.

The fully converged real-H comparison used the same two frequencies, PP, H-TZDP 8-au orbital, cell, cutoff, auxiliary basis, and immutable executable. The one-rank run was job `21311121`; the two-frequency-owner run was job `21311122`. Both exited `0:0`, wrote 66 rows and 32 primitives, and reported all response equations converged. Metadata and `S` were exactly equal. The maximum relative differences were `4.43e-12` for `norm`, `2.24e-11` for row-wise phase-aligned `Q`, and `8.11e-12` for the phase-invariant row projector `Q^dagger Q`. The stricter synthetic MPI test remains bitwise identical; the real-run residual reflects upstream SCF/linear-solver floating reductions and KS gauge, not missing or duplicated producer rows.

### Task 5: Regenerate the Historical DFT/dpsi Constraint Data

**Files:**
- Use checked-in: `Dojo-NC-SR/Orbitals_v2.0/H_TZDP/info/8/INPUT`
- Create in the campaign result directory: `baseline_manifest.json`

- [ ] **Step 1: Recreate the three historical structures**

Use H dimers at the checked-in bond lengths `0.7`, `0.9`, and `1.3` Angstrom (the original `SIAB.py` emits `Cartesian_angstrom`), H Dojo-NC-SR PP, `Rcut=8 bohr`, spherical-Bessel `Ecut=100 Ry`, `bessel_nao_smooth=true`, `bessel_nao_sigma=0.1 bohr`, and the same ABACUS origin/dpsi producer settings. Submit only to `normal`.

- [ ] **Step 2: Verify producer integrity before SIAB**

For all six files, require nonzero size, closing tags, identical primitive dimensions, and successful ABACUS logs:

```bash
for d in OUT.H-STRU2-8-0.7 OUT.H-STRU2-8-0.9 OUT.H-STRU2-8-1.3; do
  test -s "$d/orb_matrix.0.dat"
  test -s "$d/orb_matrix.1.dat"
  grep -q '</OVERLAP_Q>' "$d/orb_matrix.0.dat"
  grep -q '</OVERLAP_Q>' "$d/orb_matrix.1.dat"
  grep -q '!FINAL_ETOT_IS' "$d/running_scf.log"
done
```

- [ ] **Step 3: Reproduce the checked-in baseline loss**

Run SIAB loss evaluation at the checked-in `ORBITAL_RESULTS.txt` without an optimizer step. Compare the total and each available legacy component against the historical output to relative tolerance `1e-8`. If this fails, stop; do not tune constraints against a different baseline.

- [ ] **Step 4: Write a complete baseline manifest**

Record ABACUS commit/executable hash, PP hash, all six producer hashes, input hashes, Python/PyTorch versions, SIAB commit, and reproduced loss values in `baseline_manifest.json`.

### Task 6: Generate H-Atom ST Data and Run Both Optimizations

**Files:**
- Use: `SIAB/example_H_sternheimer/INPUT.st_only`
- Use: `SIAB/example_H_sternheimer/INPUT.st_constrained`
- Create in campaign result directory: `optimization_summary.csv`

- [ ] **Step 1: Run the converged H-atom Delta-ST producer**

Use the same converged H/H2 campaign settings already established for the H2 figure: same cell, PP, initial H-TZDP 8-au basis, ABFS, full-Coulomb eigenspace, `exx_pca_thr=1e-6`, `sternheimer_nfreq=16`, spin convention, and converged grid cutoff. The producer perturbations are Coulomb-orthonormalized with `V_full^-1/2`; the ST output itself contains no extra Coulomb factor.

- [ ] **Step 2: Validate the producer with the SIAB reader before optimization**

Print and archive:

```text
n_reference, n_primitive, n_blocks
min/max norm
min/max effective weight
Hermiticity error and min/max eigenvalue of S_B
all provenance hashes
```

Require matching PP/orbital/ABFS hashes with the intended campaign and `kernel=full_coulomb`.

- [ ] **Step 3: Run ST-only and constrained from byte-identical C0**

Archive the initial coefficient hash before each run and require equality. Run deterministic seeds. If constrained loss ratios exceed `1.05` or `1.10`, multiply only the violated penalty by `10` and restart from C0; test penalties in the sequence `10`, `100`, `1000`. Never restart from an intermediate basis.

- [ ] **Step 4: Produce the optimization comparison**

`optimization_summary.csv` columns are:

```text
mode,initial_c_sha256,final_c_sha256,initial_dft,final_dft,dft_ratio,initial_dpsi,final_dpsi,dpsi_ratio,initial_st,final_st,st_ratio,max_condition,penalty_dft,penalty_dpsi,steps,wall_seconds
```

Require both modes lower ST, constrained mode satisfies both ratios, and the initial/final fixed `1s` coefficient blocks have identical SHA256 values.

### Task 7: Run the Held-Out H/H2 SOS-RPA Validation

**Files:**
- Create in campaign result directory: `h2_validation.csv`
- Modify in ABACUS-orbitals: `docs/superpowers/specs/2026-07-18-sternheimer-siab-h-basis-design.md`
- Modify the project TeX progress document containing the existing H2 convergence figure/table.

- [ ] **Step 1: Generate `.orb` files for three candidates**

Candidates are exactly:

```text
initial_H_TZDP_8au
st_only_H_TZDP_8au
st_constrained_H_TZDP_8au
```

All retain `3s2p`, `Rcut=8 bohr`, and the byte-identical level1 `1s`.

- [ ] **Step 2: Run identical H and 0.74-A H2 DFT/EXX/SOS inputs**

For every candidate use the same 20-A cell, 0.74-A H-H distance, PP, ABFS construction, `exx_pca_thr=1e-6`, full Ewald Coulomb for SOS/LibRPA, 16 minimax frequencies, all LCAO bands, spin convention, and LibRPA executable/input. Only the orbital file changes.

- [ ] **Step 3: Compute and independently cross-check binding energies**

Store Hartree energies and calculate:

```text
D_PBE_EXX = -(E_H2_PBE_EXX - 2 E_H_PBE_EXX) * 627.5094740631
D_RPAcorr = -(Ec_RPA_H2 - 2 Ec_RPA_H) * 627.5094740631
D_RPA_at_PBE = D_PBE_EXX + D_RPAcorr
```

Use LibRPA's reported total where available and a separate parser calculation; require agreement below `1e-6 kcal/mol`.

- [ ] **Step 4: Populate the final table only from completed logs**

`h2_validation.csv` columns are:

```text
basis,dft_origin_loss,dft_dpsi_loss,st_loss,E_H_PBE_EXX,E_H2_PBE_EXX,Ec_H_RPA,Ec_H2_RPA,D_PBE_EXX_kcalmol,D_RPAcorr_kcalmol,D_RPA_at_PBE_kcalmol,delta_from_DeltaST_kcalmol,abacus_commit,librpa_commit,orbital_sha256
```

The preferred basis is constrained only if `abs(delta_from_DeltaST_kcalmol) <= 0.1` and its PBE+EXX binding contribution differs from initial TZDP by at most `0.1 kcal/mol`. If only ST-only reaches the Delta-ST tolerance, report it as diagnostic and retain constrained as unresolved.

- [ ] **Step 5: Update documentation and commit only completed results**

Document both positive and negative results, exact parameters, job IDs, wall times, hashes, and the fact that H2 RPA was held out. Do not write expected values into result cells.

```bash
git add docs/superpowers/specs/2026-07-18-sternheimer-siab-h-basis-design.md \
        SIAB/example_H_sternheimer/README.md
GIT_COMMITTER_NAME='AroundPeking' GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit --author='Codex <codex@openai.com>' -m 'docs(siab): report H Sternheimer basis validation'
```

## Production Completion Gate

The work is complete only when all items below have direct artifacts:

1. ABACUS writer unit tests and serial/MPI identity tests pass.
2. Producer rows contain complete reconstructed Delta-ST wavefunctions and the same uniform-grid `DeltaOmega` in `n`, `Q`, and `S_B`.
3. ST perturbations use the validated full-Coulomb orthonormal auxiliary space; no extra Coulomb factor enters the SIAB loss.
4. Historical DFT/dpsi baseline is reproduced before constraints are applied.
5. ST-only and constrained runs start from byte-identical H-TZDP coefficients and both lower ST loss.
6. Fixed `1s` is byte-identical in all three orbital files.
7. H/H2 validation changes only the orbital file and uses completed ABACUS-to-LibRPA outputs.
8. Result tables contain hashes, commits, job IDs, resources, and wall times.
