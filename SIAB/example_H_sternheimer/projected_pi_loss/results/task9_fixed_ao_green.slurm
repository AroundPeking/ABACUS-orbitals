#!/bin/bash
#SBATCH --job-name=siab-fixedao-green
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=24:00:00
#SBATCH --output=/work1/ghj/sternheimer_abacus_tests/siab_fixed_ao_green_20260807/logs/%x-%j.out

set -euo pipefail

campaign=/work1/ghj/sternheimer_abacus_tests/siab_fixed_ao_green_20260807
archive=${campaign}/input/abacus-orbitals-fixed-ao.tar
work=/tmp/siab-fixedao-green-${SLURM_JOB_ID}
result=${campaign}/jobs/${SLURM_JOB_ID}
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python

mkdir -p "${work}" "${result}"
trap 'status=$?; cp -a "${work}/." "${result}/" 2>/dev/null || true; exit ${status}' EXIT

sha256sum "${archive}" > "${work}/archive.sha256"
tar -xf "${archive}" -C "${work}"
sha256sum \
    "${work}/SIAB/opt_orb_pytorch_dpsi/galerkin_sternheimer.py" \
    "${work}/SIAB/opt_orb_pytorch_dpsi/sternheimer_fixed_ao_data.py" \
    "${work}/SIAB/opt_orb_pytorch_dpsi/IO/read_sternheimer_fixed_ao.py" \
    "${work}/SIAB/opt_orb_pytorch_dpsi/fixed_ao_galerkin.py" \
    "${work}/SIAB/tests/test_galerkin_sternheimer.py" \
    "${work}/SIAB/tests/test_read_sternheimer_fixed_ao.py" \
    "${work}/SIAB/tests/test_fixed_ao_galerkin.py" \
    > "${work}/source.sha256"

{
    date --iso-8601=seconds
    printf 'source_commit=%s\n' 2e548d23a87e8e16f1880703e726e344a563afe7
    printf 'job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'node_list=%s\n' "${SLURM_JOB_NODELIST}"
    printf 'cpus_per_task=%s\n' "${SLURM_CPUS_PER_TASK}"
    "${python}" --version
    "${python}" -c 'import torch; print("torch=" + torch.__version__)'
} > "${work}/environment.txt" 2>&1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TORCH_NUM_INTEROP_THREADS=1

run_suite() {
    local index=$1
    cd "${work}/SIAB/tests"
    set +e
    /usr/bin/time -v -o "${work}/time-run${index}.txt" \
        "${python}" -m unittest -v \
        test_galerkin_sternheimer \
        test_read_sternheimer_fixed_ao \
        test_fixed_ao_galerkin \
        > "${work}/run${index}.stdout.txt" \
        2> "${work}/run${index}.stderr.txt"
    local rc=$?
    set -e
    printf '%s\n' "${rc}" > "${work}/run${index}.rc"
    if [[ "${rc}" -ne 0 ]]; then
        return "${rc}"
    fi
    grep -Fq "Ran 15 tests" "${work}/run${index}.stderr.txt"
    grep -Fxq "OK" "${work}/run${index}.stderr.txt"
    sed -E '/^Ran 15 tests in [0-9.]+s$/d' \
        "${work}/run${index}.stderr.txt" \
        > "${work}/run${index}.normalized.txt"
}

run_suite 1
run_suite 2
cmp "${work}/run1.normalized.txt" "${work}/run2.normalized.txt"
sha256sum \
    "${work}/run1.normalized.txt" \
    "${work}/run2.normalized.txt" \
    > "${work}/normalized.sha256"
touch "${work}/completed.ok"
