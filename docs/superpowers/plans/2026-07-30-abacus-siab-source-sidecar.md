# ABACUS SIAB Source Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the molecular ABACUS Sternheimer-SIAB producer write the source overlaps `D_ia,e = <psi_i vbar_a|B_e>` and provide a source-only mode that skips every first-order Sternheimer solve.

**Architecture:** Reuse the existing full-Coulomb whitening, complete uniform-grid occupied wavefunctions, reciprocal SIAB primitives, deterministic MPI gather, and v1 provenance. A source row is built from `0.5 * perturbation_ry * psi`, projected with the same `<left|right>` primitive-overlap routine as the response target, and written to a separate strict sidecar. The normal SIAB path writes both response and source files; `sternheimer_siab_source_only` writes only the sidecar and bypasses Delta-subspace construction and frequency loops.

**Tech Stack:** ABACUS C++14, complex128 uniform-grid/PW transforms, MPI/OpenMP, GoogleTest, CMake, SLURM `normal` on `df_dcu`.

---

## Repository And Worktree Guard

The current molecular source worktree

```text
/Users/ghj/同步空间/AITP_project/sternheimer_abacus/.staging/abacus-sternheimer-siab-l2
```

contains unrelated uncommitted Ewald/auxiliary-charge work. Do not edit,
stage, reset, or commit those files. Create a detached implementation worktree
from the exact current molecular commit without creating a persistent branch:

```bash
git -C /Users/ghj/同步空间/AITP_project/sternheimer_abacus/.staging/abacus-sternheimer-siab-l2 \
  worktree add --detach \
  /Users/ghj/同步空间/AITP_project/sternheimer_abacus/.worktrees/abacus-siab-projected-pi-source \
  3a805e56a79abbeb64c049257b23de945100194b
```

Before every commit, verify that only files named in that task are staged and
use the task-specific command and message printed below.

Keep the detached worktree until its commit chain has been integrated into the
existing molecular branch after the unrelated dirty work is resolved. Never
discard that dirty work to make integration easier.

## File Map

Create:

- `source/source_lcao/module_ri/sternheimer_siab_source.h`: pure source-grid API.
- `source/source_lcao/module_ri/sternheimer_siab_source.cpp`: Hartree-from-Ry source construction.
- `source/source_lcao/module_ri/test/test_sternheimer_siab_source.cpp`: unit and factor-of-two tests.

Modify:

- `source/source_lcao/module_ri/sternheimer_siab_data.h`: add `SourceRow`.
- `source/source_lcao/module_ri/sternheimer_siab_writer.h`: expose `write_source_v1`.
- `source/source_lcao/module_ri/sternheimer_siab_writer.cpp`: strict sidecar validation and atomic writer.
- `source/source_lcao/module_ri/test/test_sternheimer_siab_writer.cpp`: canonical source fixture and rejection tests.
- `source/source_lcao/module_ri/sternheimer_siab_mpi.h`: source-row gather declaration.
- `source/source_lcao/module_ri/sternheimer_siab_mpi.cpp`: deterministic source-row pack/gather/unpack.
- `source/source_lcao/module_ri/test/test_sternheimer_siab_mpi.cpp`: serial/MPI equality tests.
- `source/source_io/module_parameter/input_parameter.h`: add the source-only Boolean.
- `source/source_io/module_parameter/read_input_item_output.cpp`: register and validate it.
- `source/source_io/test_serial/read_input_item_test.cpp`: default/read/invalid-combination tests.
- `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp`: produce source rows, skip Delta-ST in source-only mode, and report status.
- `source/source_lcao/module_ri/test/sternheimer_abacus_st_smoke_test.cpp`: source ownership and expected-row helpers.
- `source/source_lcao/module_ri/CMakeLists.txt`: compile the new source module.
- `source/source_lcao/module_ri/test/CMakeLists.txt`: build its unit test.

No LibRPA, Coulomb-kernel, GreenX, Sternheimer solver, or existing response-v1
format is changed.

The design's focused source-writer boundary is implemented as the separate
`write_source_v1` API inside the existing writer translation unit. This keeps
source/response validation paths distinct while reusing one provenance JSON
serializer and one atomic-file replacement implementation.

### Task 1: Pin The Hartree Source Convention

**Files:**
- Create: `source/source_lcao/module_ri/sternheimer_siab_source.h`
- Create: `source/source_lcao/module_ri/sternheimer_siab_source.cpp`
- Create: `source/source_lcao/module_ri/test/test_sternheimer_siab_source.cpp`
- Modify: `source/source_lcao/module_ri/CMakeLists.txt`
- Modify: `source/source_lcao/module_ri/test/CMakeLists.txt`

- [ ] **Step 1: Write the failing complex-grid test**

Use a two-point occupied wavefunction and a perturbation already scaled to Ry:

```cpp
using Complex = std::complex<double>;
const std::vector<Complex> psi = {{1.0, 2.0}, {-0.5, 1.0}};
const std::vector<double> potential_ry = {4.0, -2.0};

const auto source = build_source_grid_from_rydberg_potential(
    psi, potential_ry);

EXPECT_EQ(source[0], Complex(2.0, 4.0));
EXPECT_EQ(source[1], Complex(0.5, -1.0));
```

Add one primitive `B={(1,-1),(2,0.5)}` and `DeltaOmega=0.25`. Assert that
`overlap_q(source,{B},DeltaOmega)[0]` equals the direct expression

```cpp
0.25 * (std::conj(source[0]) * B[0]
      + std::conj(source[1]) * B[1])
```

to `1e-14`. Also assert that passing the Hartree values `{2,-1}` as though
they were Ry gives exactly half the intended source. This makes the required
`0.5` conversion observable rather than implicit.

- [ ] **Step 2: Run the focused target on `df_dcu` and verify RED**

After syncing the detached source tree to the remote build root defined in
Task 6, run:

```bash
cmake --build "$BUILD_ROOT" -j 30 \
  --target MODULE_RI_sternheimer_siab_source_test
```

Expected before implementation: missing target or undefined
`build_source_grid_from_rydberg_potential`.

- [ ] **Step 3: Implement the pure checked API**

Expose exactly:

```cpp
namespace module_ri
{
namespace sternheimer_siab
{
using ComplexGrid = std::vector<std::complex<double>>;

ComplexGrid build_source_grid_from_rydberg_potential(
    const ComplexGrid& occupied_wavefunction,
    const std::vector<double>& perturbation_ry);
}
}
```

The implementation rejects unequal or empty vectors and non-finite real or
complex values. It computes:

```cpp
source[ir] = occupied_wavefunction[ir]
           * (0.5 * perturbation_ry[ir]);
```

Do not conjugate `psi` here. The existing overlap routine conjugates its left
argument, producing `<psi_i vbar_a|B_e>`.

- [ ] **Step 4: Run the focused source and overlap tests**

```bash
cmake --build "$BUILD_ROOT" -j 30 --target \
  MODULE_RI_sternheimer_siab_source_test \
  MODULE_RI_sternheimer_siab_overlap_test
ctest --test-dir "$BUILD_ROOT" -R \
  'sternheimer_siab_(source|overlap)' --output-on-failure
```

Expected: both tests pass and the factor-of-two assertion is exact to
`1e-14`.

- [ ] **Step 5: Commit the source convention**

```bash
git add source/source_lcao/module_ri/sternheimer_siab_source.h \
        source/source_lcao/module_ri/sternheimer_siab_source.cpp \
        source/source_lcao/module_ri/test/test_sternheimer_siab_source.cpp \
        source/source_lcao/module_ri/CMakeLists.txt \
        source/source_lcao/module_ri/test/CMakeLists.txt
GIT_AUTHOR_NAME='Codex' GIT_AUTHOR_EMAIL='codex@openai.com' \
GIT_COMMITTER_NAME='AroundPeking' \
GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit -m 'test(sternheimer): pin SIAB source overlap units'
```

### Task 2: Add The Strict Source Sidecar Writer

**Files:**
- Modify: `source/source_lcao/module_ri/sternheimer_siab_data.h`
- Modify: `source/source_lcao/module_ri/sternheimer_siab_writer.h`
- Modify: `source/source_lcao/module_ri/sternheimer_siab_writer.cpp`
- Modify: `source/source_lcao/module_ri/test/test_sternheimer_siab_writer.cpp`

- [ ] **Step 1: Add a failing canonical sidecar test**

Define the new row type in the test contract:

```cpp
SourceRow row;
row.occupied_state = 0;
row.auxiliary_channel = 1;
row.occupation = 2.0;
row.norm = 1.25;
row.d = {{0.5, -0.25}, {1.0, 0.75}};
```

Call `write_source_v1` with two primitive columns and assert that the output
contains each section exactly once and in this order:

```text
STERNHEIMER_SIAB_SOURCE_HEADER
PRIMITIVE_BLOCKS
SOURCE_METADATA
OVERLAP_D
OVERLAP_S
PROVENANCE_JSON
```

Assert the metadata row is `0 1 2 1.25`, `D` is primitive-fast row-major, the
same provenance JSON serializer is used as response v1, and the writer leaves
no `.tmp` file. Add failures for duplicate `(occupied_state,channel)`, wrong
`D` width, non-positive norm, non-finite complex data, and non-Hermitian `S`.

- [ ] **Step 2: Verify RED**

```bash
cmake --build "$BUILD_ROOT" -j 30 \
  --target MODULE_RI_sternheimer_siab_writer_test
"$BUILD_ROOT/source/source_lcao/module_ri/test/MODULE_RI_sternheimer_siab_writer_test" \
  --gtest_filter='*Source*'
```

Expected: `SourceRow` or `write_source_v1` is undefined.

- [ ] **Step 3: Add the data type and writer API**

Add:

```cpp
struct SourceRow
{
    int occupied_state;
    int auxiliary_channel;
    double occupation;
    double norm;
    std::vector<std::complex<double>> d;
};
```

and:

```cpp
void write_source_v1(
    const std::string& path,
    double grid_volume_bohr3,
    const std::vector<PrimitiveBlock>& blocks,
    const std::vector<SourceRow>& rows,
    const std::vector<std::complex<double>>& overlap_s,
    const Provenance& provenance);
```

Reuse the existing block, overlap, provenance, round-trip float, UTF-8, temp
file, flush, close, and atomic-rename helpers. Do not duplicate provenance JSON
logic. Sort source rows lexicographically by `(occupied_state,
auxiliary_channel)` before writing, but reject duplicate keys.

- [ ] **Step 4: Pass all writer regressions**

```bash
cmake --build "$BUILD_ROOT" -j 30 \
  --target MODULE_RI_sternheimer_siab_writer_test
"$BUILD_ROOT/source/source_lcao/module_ri/test/MODULE_RI_sternheimer_siab_writer_test"
```

Expected: response-v1 golden tests and all source-v1 tests pass.

- [ ] **Step 5: Commit the sidecar format**

```bash
git add source/source_lcao/module_ri/sternheimer_siab_data.h \
        source/source_lcao/module_ri/sternheimer_siab_writer.h \
        source/source_lcao/module_ri/sternheimer_siab_writer.cpp \
        source/source_lcao/module_ri/test/test_sternheimer_siab_writer.cpp
GIT_AUTHOR_NAME='Codex' GIT_AUTHOR_EMAIL='codex@openai.com' \
GIT_COMMITTER_NAME='AroundPeking' \
GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit -m 'feat(sternheimer): define SIAB source sidecar'
```

### Task 3: Gather Source Rows Deterministically Across MPI

**Files:**
- Modify: `source/source_lcao/module_ri/sternheimer_siab_mpi.h`
- Modify: `source/source_lcao/module_ri/sternheimer_siab_mpi.cpp`
- Modify: `source/source_lcao/module_ri/test/test_sternheimer_siab_mpi.cpp`

- [ ] **Step 1: Write failing serial and two-rank tests**

Expose the intended API in the test:

```cpp
std::vector<SourceRow> gather_source_rows_to_root(
    const std::vector<SourceRow>& local_rows,
    std::size_t nprimitive,
    int root,
    MPI_Comm communicator);
```

Rank 0 contributes key `(0,1)` and rank 1 contributes `(0,0)`. On root, assert
the result is sorted `(0,0),(0,1)`, all `D` values are exact, and writing it is
byte-identical to the serial gather. Add collective failures for a wrong row
width and duplicate global keys.

- [ ] **Step 2: Verify RED with one and two ranks**

```bash
cmake --build "$BUILD_ROOT" -j 30 \
  --target MODULE_RI_sternheimer_siab_mpi_test
"$BUILD_ROOT/source/source_lcao/module_ri/test/MODULE_RI_sternheimer_siab_mpi_test" \
  --gtest_filter='*Source*'
mpirun -np 2 \
  "$BUILD_ROOT/source/source_lcao/module_ri/test/MODULE_RI_sternheimer_siab_mpi_test" \
  --gtest_filter='*Source*'
```

Expected: missing source gather API.

- [ ] **Step 3: Implement checked pack/gather/unpack**

Pack each row as:

```text
occupied_state, auxiliary_channel, occupation, norm,
then real and imaginary parts for every D entry in primitive order
```

Use the same `MPI_Gather`/`MPI_Gatherv` pattern as reference rows. Validate
integer-valued keys before conversion, all local inputs collectively before
`Gatherv`, count/displacement overflow, final row width, and duplicate keys.
The non-MPI overload accepts only `root=0` and returns the same sorted result.

- [ ] **Step 4: Run the complete SIAB MPI target**

```bash
ctest --test-dir "$BUILD_ROOT" -R \
  'MODULE_RI_sternheimer_siab_mpi_test(_mpi2)?$' --output-on-failure
```

Expected: serial and MPI2 tests pass.

- [ ] **Step 5: Commit MPI source assembly**

```bash
git add source/source_lcao/module_ri/sternheimer_siab_mpi.h \
        source/source_lcao/module_ri/sternheimer_siab_mpi.cpp \
        source/source_lcao/module_ri/test/test_sternheimer_siab_mpi.cpp
GIT_AUTHOR_NAME='Codex' GIT_AUTHOR_EMAIL='codex@openai.com' \
GIT_COMMITTER_NAME='AroundPeking' \
GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit -m 'feat(sternheimer): gather SIAB source rows with MPI'
```

### Task 4: Add The Source-Only Input Contract

**Files:**
- Modify: `source/source_io/module_parameter/input_parameter.h`
- Modify: `source/source_io/module_parameter/read_input_item_output.cpp`
- Modify: `source/source_io/test_serial/read_input_item_test.cpp`

- [ ] **Step 1: Write failing parser tests**

Assert:

```cpp
EXPECT_FALSE(param.input.sternheimer_siab_source_only);
```

then read `true` and assert it becomes true. Its `check_value` must exit with
an explicit message for either invalid case:

```text
sternheimer_siab_source_only=true, out_sternheimer_siab=false
sternheimer_siab_source_only=true, out_sternheimer_librpa=true
```

The valid case has `out_sternheimer_siab=true`,
`out_sternheimer_librpa=false`, `basis_type=lcao`, and
`sternheimer_delta=true`.

- [ ] **Step 2: Verify RED**

```bash
cmake --build "$BUILD_ROOT" -j 30 --target MODULE_IO_read_item_serial
"$BUILD_ROOT/source/source_io/test_serial/MODULE_IO_read_item_serial" \
  --gtest_filter='ReadInputItemTest.output'
```

Expected: the parameter or item is absent.

- [ ] **Step 3: Register the default-false Boolean**

Add to `Input_para`:

```cpp
bool sternheimer_siab_source_only = false;
```

Register `Input_Item("sternheimer_siab_source_only")` immediately after
`out_sternheimer_siab`, use `read_sync_bool`, and describe that it writes
`OUT.ABACUS/STERNHEIMER_SIAB_SOURCE_V1.dat` without solving first-order
equations. Keep all existing `out_sternheimer_siab` validation unchanged.

- [ ] **Step 4: Pass the complete serial input test**

```bash
"$BUILD_ROOT/source/source_io/test_serial/MODULE_IO_read_item_serial"
```

Expected: all input-item tests pass.

- [ ] **Step 5: Commit the input contract**

```bash
git add source/source_io/module_parameter/input_parameter.h \
        source/source_io/module_parameter/read_input_item_output.cpp \
        source/source_io/test_serial/read_input_item_test.cpp
GIT_AUTHOR_NAME='Codex' GIT_AUTHOR_EMAIL='codex@openai.com' \
GIT_COMMITTER_NAME='AroundPeking' \
GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit -m 'feat(sternheimer): add SIAB source-only output mode'
```

### Task 5: Integrate Source Production Before Delta-ST

**Files:**
- Modify: `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp`
- Modify: `source/source_lcao/module_ri/test/sternheimer_abacus_st_smoke_test.cpp`

- [ ] **Step 1: Write failing ownership and row-count tests**

Extract pure helpers and test them before integration:

```cpp
int siab_source_owner(
    int occupied_state, int channel, int num_channels, int mpi_ranks);

std::size_t expected_siab_source_rows(
    const std::vector<int>& occupied_band_counts, int num_channels);
```

For two occupied states, 5 channels, and 3 ranks, assert every one of the 10
keys has exactly one owner and ownership is
`(occupied_state * num_channels + channel) % mpi_ranks`. Assert expected rows
is `10`, rejects non-positive dimensions, and does not include frequency.

- [ ] **Step 2: Verify RED**

```bash
cmake --build "$BUILD_ROOT" -j 30 \
  --target MODULE_RI_sternheimer_abacus_st_smoke_test
"$BUILD_ROOT/source/source_lcao/module_ri/test/MODULE_RI_sternheimer_abacus_st_smoke_test" \
  --gtest_filter='*SIABSource*'
```

Expected: helper symbols are absent.

- [ ] **Step 3: Build each local source row at zero-order scope**

Add `source_only = PARAM.inp.sternheimer_siab_source_only` and a
`local_siab_source_rows` vector beside `local_siab_rows`. Immediately after
`states.wavefunctions` and occupations are ready, before
`delta_subspace_start`, execute this logical loop:

```cpp
for (int ib = 0; ib != occupied_count; ++ib)
{
    const double occupation = elec_state.wg(spin_index, ib);
    if (occupation <= 1.0e-8) continue;
    const int occupied_state = occupied_state_offset + ib;
    for (int channel = 0; channel != num_channels; ++channel)
    {
        if (siab_source_owner(occupied_state, channel,
                              num_channels, GlobalV::NPROC)
            != GlobalV::MY_RANK) continue;
        const auto source = siab::build_source_grid_from_rydberg_potential(
            states.wavefunctions[ib], perturbations_ry[channel]);
        siab::SourceRow row;
        row.occupied_state = occupied_state;
        row.auxiliary_channel = channel;
        row.occupation = occupation;
        row.norm = siab::norm(source, grid_data.volume_element);
        row.d = project_siab_response_to_primitives(
            source, ucell, siab_primitives);
        local_siab_source_rows.push_back(std::move(row));
    }
}
```

This loop runs only for `write_siab`. The `0.5` conversion remains confined
to the tested source helper. Do not retain a duplicate Hartree-potential
array.

- [ ] **Step 4: Bypass all first-order work in source-only mode**

After source rows for one spin are complete, source-only mode records
zero-valued Delta diagnostics, advances `occupied_state_offset`, and continues
to the next spin. It must not enter:

```text
delta_subspace_start
frequency_start
build_rhs_from_hartree_perturbation
solve_delta_sternheimer_linear_response
```

After all spins, gather source rows, require exactly
`sum(occupied_band_counts) * num_channels`, and write
`OUT.ABACUS/STERNHEIMER_SIAB_SOURCE_V1.dat`. In a normal SIAB run, continue to
gather/write `sternheimer_matrix.dat` exactly as before. In source-only mode,
skip reference-row gather and all frequency output/reduction loops.

- [ ] **Step 5: Make status output explicit and backwards compatible**

For a normal run retain `format siab_v1` and `data_files 1`, then append:

```text
source_files 1
source_file STERNHEIMER_SIAB_SOURCE_V1.dat
sternheimer_siab_source_only no
source_rows 214
```

For source-only output write:

```text
format siab_source_v1
data_files 0
source_files 1
source_file STERNHEIMER_SIAB_SOURCE_V1.dat
sternheimer_siab_source_only yes
source_rows 214
target_file none
solved_equations 0
```

The shown value is the H-atom example. H2 writes `source_rows 428`; every
system writes the integer returned by `expected_siab_source_rows`.

Retain grid, zero-order source, spin, primitive, auxiliary-basis, whitening,
memory, commit, and executable provenance. Add a final progress event
`siab_source_written`. The source-only success path must close/free all MPI
communicators it created before returning.

- [ ] **Step 6: Pass focused and existing producer tests**

```bash
cmake --build "$BUILD_ROOT" -j 30 --target \
  MODULE_RI_sternheimer_siab_source_test \
  MODULE_RI_sternheimer_siab_writer_test \
  MODULE_RI_sternheimer_siab_mpi_test \
  MODULE_RI_sternheimer_abacus_st_smoke_test \
  MODULE_IO_read_item_serial
ctest --test-dir "$BUILD_ROOT" -R \
  'sternheimer_siab|sternheimer_abacus_st_smoke|MODULE_IO_read_item_serial' \
  --output-on-failure
```

Expected: every selected test passes; existing response-v1 fixtures are
unchanged.

- [ ] **Step 7: Commit producer integration**

```bash
git add source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp \
        source/source_lcao/module_ri/test/sternheimer_abacus_st_smoke_test.cpp
GIT_AUTHOR_NAME='Codex' GIT_AUTHOR_EMAIL='codex@openai.com' \
GIT_COMMITTER_NAME='AroundPeking' \
GIT_COMMITTER_EMAIL='gonghuanjing@iphy.ac.cn' \
git commit -m 'feat(sternheimer): emit SIAB source overlaps'
```

### Task 6: Build And Test An Immutable Executable On `df_dcu`

**Files:**
- Create remotely: `/work1/ghj/sternheimer_abacus_tests/siab_projected_pi_source_20260730/build.slurm`
- Create remotely: `/work1/ghj/sternheimer_abacus_tests/siab_projected_pi_source_20260730/fake_git_exact_commit.sh`

- [ ] **Step 1: Sync the detached committed source**

Use the live ControlMaster socket and exclude build/Git state:

```bash
LOCAL=/Users/ghj/同步空间/AITP_project/sternheimer_abacus/.worktrees/abacus-siab-projected-pi-source/
REMOTE=/work1/ghj/sternheimer_abacus_tests/siab_projected_pi_source_20260730/source/
rsync -a --delete --exclude=.git \
  -e 'ssh -S /tmp/codex-ssh-cm-ghj-60.245.128.10-65010 -p 65010' \
  "$LOCAL" "ghj@60.245.128.10:$REMOTE"
```

Record the detached `git rev-parse HEAD` and generate the exact fake-Git
script using that full 40-digit hash. The build must fail if `commit.h` does
not contain it.

- [ ] **Step 2: Submit one full-resource normal-node build**

Adapt the verified build script at
`/work1/ghj/sternheimer_abacus_tests/siab_channel_mpi_exact_c273b4ee7_20260726/build.slurm`
with:

```text
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00
```

Use `/public/home/ghj/app/src/env_60_245_intel2021.sh`, the same LibRI,
LibComm, ELPA, cereal, and GoogleTest paths, `BUILD_TESTING=ON`,
`ENABLE_MPI=ON`, `ENABLE_LIBRI=ON`, `ENABLE_GREENX_MINIMAX=ON`,
`DEBUG_INFO=ON`, and `USE_OPENMP=ON`.

- [ ] **Step 3: Run all relevant tests inside the build job**

Build and execute:

```text
MODULE_RI_sternheimer_siab_source_test
MODULE_RI_sternheimer_siab_overlap_test
MODULE_RI_sternheimer_siab_writer_test
MODULE_RI_sternheimer_siab_mpi_test
MODULE_RI_sternheimer_siab_mpi_test with mpirun -np 2
MODULE_RI_sternheimer_abacus_st_smoke_test
MODULE_IO_read_item_serial
abacus_3p
```

Expected: all return zero. Copy `abacus_3p` and `commit.h` to the immutable
`artifacts/job${SLURM_JOB_ID}/` directory, set mode `0555`, and record SHA256,
inode, size, source commit, source-content SHA256, build job, node, and date.

- [ ] **Step 4: Verify provenance before any physics run**

```bash
sha256sum "$ARTIFACT/abacus_3p"
grep '^#define COMMIT ' "$ARTIFACT/commit.h"
git -C "$LOCAL" log -1 --format='%H %an <%ae> %cn <%ce> %s'
```

Expected author is Codex and committer is AroundPeking. The executable hash is
copied into every subsequent submission manifest.

### Task 7: Produce Exact H And H2 Source Sidecars

**Files:**
- Create remotely: `/work1/ghj/sternheimer_abacus_tests/siab_projected_pi_source_only_h_h2_20260730/`
- Modify locally after completion: `sternheimer_siab_project/main.tex`

- [ ] **Step 1: Clone the exact archived physical target inputs**

Copy, without modifying the source directories:

```text
/work1/ghj/sternheimer_abacus_tests/siab_greedy_targets_h2_channel_mpi_prod_v1_20260726/producer_atom
/work1/ghj/sternheimer_abacus_tests/siab_greedy_targets_h2_channel_mpi_prod_v1_20260726/producer_h2
```

into fresh `H/` and `H2/` directories. Remove copied `OUT.*`, progress,
status, and old stdout files. Preserve and hash `INPUT`, `STRU`, `KPT`, fixed
16-frequency grid, H pseudopotential, TZDP-8au orbital, and explicit ABFS.
Append exactly:

```text
sternheimer_siab_source_only 1
```

All other physical inputs remain byte-identical.

- [ ] **Step 2: Submit a two-case normal array**

Use one MPI rank per node and fill each node with OpenMP:

```text
#SBATCH --partition=normal
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
```

Launch with `mpirun -np 32 -ppn 1`, `OMP_NUM_THREADS=30`, the immutable
artifact, and `/usr/bin/time -v`. Do not submit to `debug`.

- [ ] **Step 3: Enforce producer completion gates**

For both H and H2 require:

```text
Slurm state COMPLETED and ExitCode 0:0
ABACUS finish marker present
status success
format siab_source_v1
sternheimer_siab_source_only yes
solved_equations 0
all first-order progress events absent
STERNHEIMER_SIAB_SOURCE_V1.dat present and nonempty
sternheimer_matrix.dat absent from the fresh OUT directory
```

Expected source rows are `214` for H and `428` for H2. Expected primitive
columns are `625` and `1250`. Record wall time, maximum resident set, node
list, executable SHA256, source commit, and every input SHA256.

- [ ] **Step 4: Verify transitional zero-order identity**

Compare the new and archived `running_scf.log` files for SCF completion,
occupied-state count, occupied eigenvalues, occupations, grid dimensions, and
final total energy. Require occupied eigenvalues and occupations to agree
within `1e-12 Ha`/`1e-14`, grid dimensions to match exactly, and final total
energy within `1e-12 Ha`. A failed comparison blocks pairing with the old v1
response even if all source-file structure checks pass.

- [ ] **Step 5: Record the completed producer stage in TeX**

Add the source definition, Hartree/Ry factor, row dimensions, executable and
input provenance, source-only runtime, and all producer gates to
`/Users/ghj/同步空间/AITP_project/sternheimer_abacus/sternheimer_siab_project/main.tex`.
State explicitly that no projected-Pi physics conclusion exists yet. Compile
with the document's existing engine, run `pdftotext` for the new subsection,
and render the affected pages to verify no table or equation overflows.

- [ ] **Step 6: Stop at the producer boundary**

Do not add `pi_dpsi_joint` or start an optimizer in this plan. Hand the two
source sidecars and their manifests to
`2026-07-30-siab-projected-pi-feasibility.md`.
