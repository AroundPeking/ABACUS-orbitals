#!/bin/bash
#SBATCH --job-name=c_fd8_build_711af_dcu
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110000
#SBATCH --time=02:00:00

set -euo pipefail
trap 'status=$?; echo "C_SOLID_FD8_DFDCU_BUILD_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${C_SOLID_DFDCU_STAGE_ROOT:?missing df_dcu stage root}"
: "${BUILD_RUNNER_SHA256:?missing build runner SHA256}"

DFDCU_ENV=${DFDCU_ENV:-/public/home/ghj/app/src/env_60_245_intel2021.sh}
DFDCU_ELPA_ROOT=${DFDCU_ELPA_ROOT:-/public/home/ghj/app/deps/elpa-2021.11.002-intelmpi2021}
source "$DFDCU_ENV"

source_commit=711af860c125b9757c344a1961b63524c550cfe4
source_root=$C_SOLID_DFDCU_STAGE_ROOT/abacus-source-711af860c
deps=$C_SOLID_DFDCU_STAGE_ROOT/deps
build_root=$C_SOLID_DFDCU_STAGE_ROOT/build-711af860c-intel2021
artifact_root=$C_SOLID_DFDCU_STAGE_ROOT/artifacts/abacus-711af860c-intel2021
runner=$C_SOLID_DFDCU_STAGE_ROOT/source/build_c_solid_fd8_q13_standard_abacus_dfdcu.slurm
metadata=$C_SOLID_DFDCU_STAGE_ROOT/artifacts/job3117942/commit_metadata.txt

test "$SLURM_JOB_PARTITION" = debug
test "$SLURM_CPUS_PER_TASK" = 30
test -d "$source_root/source"
test -d "$deps/LibRI_latest"
test -d "$deps/LibComm_fix_status"
test -d "$deps/cereal-1.3.0/include"
test -d "$DFDCU_ELPA_ROOT"
test -d "$deps/lib/pkgconfig"
test -s "$metadata"
grep -Fq "$source_commit" "$metadata"
test "$(sha256sum "$runner" | awk '{print $1}')" = "$BUILD_RUNNER_SHA256"
test ! -e "$build_root"
test ! -e "$artifact_root"
mkdir -p "$build_root" "$artifact_root"

export CC=mpiicc
export CXX=mpiicpc
export FC=mpiifort
export PKG_CONFIG_PATH=$deps/lib/pkgconfig:${PKG_CONFIG_PATH:-}
export LD_LIBRARY_PATH=$DFDCU_ELPA_ROOT/lib:$deps/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cmake -S "$source_root" -B "$build_root" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_MPI=ON \
  -DUSE_OPENMP=ON \
  -DENABLE_LCAO=ON \
  -DENABLE_LIBXC=ON \
  -DENABLE_LIBRI=ON \
  -DENABLE_LIBCOMM=ON \
  -DENABLE_GREENX_MINIMAX=ON \
  -DDEBUG_INFO=ON \
  -DENABLE_MLALGO=OFF \
  -DENABLE_FFT_TWO_CENTER=ON \
  -DBUILD_TESTING=OFF \
  -DLIBRI_DIR="$deps/LibRI_latest" \
  -DLIBCOMM_DIR="$deps/LibComm_fix_status" \
  -DELPA_DIR="$DFDCU_ELPA_ROOT" \
  -DCEREAL_INCLUDE_DIR="$deps/cereal-1.3.0/include"

/usr/bin/time -v -o "$artifact_root/build.time" \
  cmake --build "$build_root" --target abacus_3p -j "$SLURM_CPUS_PER_TASK"

binary=$build_root/abacus_3p
test -x "$binary"
grep -Eq '^ENABLE_MPI:BOOL=ON$' "$build_root/CMakeCache.txt"
grep -Eq '^ENABLE_LCAO:BOOL=ON$' "$build_root/CMakeCache.txt"
grep -Eq '^ENABLE_LIBRI:BOOL=ON$' "$build_root/CMakeCache.txt"
grep -Eq '^ENABLE_LIBCOMM:BOOL=ON$' "$build_root/CMakeCache.txt"
grep -Eq '^ENABLE_GREENX_MINIMAX:BOOL=ON$' "$build_root/CMakeCache.txt"
grep -Eq '^DEBUG_INFO:BOOL=ON$' "$build_root/CMakeCache.txt"
ldd "$binary" > "$artifact_root/ldd.txt"
! grep -q 'not found' "$artifact_root/ldd.txt"

cp "$binary" "$artifact_root/abacus_3p"
cp "$build_root/CMakeCache.txt" "$artifact_root/CMakeCache.txt"
find "$source_root" -path "$source_root/.git" -prune -o -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$artifact_root/source_sha256.txt"
find "$deps/LibRI_latest" "$deps/LibComm_fix_status" "$deps/cereal-1.3.0" \
  -type f -print0 | sort -z | xargs -0 sha256sum > "$artifact_root/header_dependency_sha256.txt"

{
  echo status=success
  echo source_commit=$source_commit
  echo cluster=df_dcu
  echo environment=$DFDCU_ENV
  echo elpa_root=$DFDCU_ELPA_ROOT
  echo compiler="$($CXX --version | head -1)"
  echo mpi="$(mpirun --version | head -1)"
  echo slurm_job_id=$SLURM_JOB_ID
  sha256sum "$runner" "$artifact_root/abacus_3p" "$artifact_root/CMakeCache.txt" \
    "$artifact_root/source_sha256.txt" "$artifact_root/header_dependency_sha256.txt"
} > "$artifact_root/BUILD_PROVENANCE.txt"

echo "C_SOLID_FD8_DFDCU_BUILD_OK artifact=$artifact_root/abacus_3p"
