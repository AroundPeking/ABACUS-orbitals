#!/bin/bash
#SBATCH --job-name=siab-galerkin-green
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=24:00:00

set -euo pipefail

campaign=/work1/ghj/sternheimer_abacus_tests/siab_galerkin_green_20260807
archive=${campaign}/input/abacus-orbitals-green.tar
work=/tmp/siab-galerkin-green-${SLURM_JOB_ID}
result=${campaign}/jobs/${SLURM_JOB_ID}
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python

mkdir -p "${work}" "${result}"
trap 'status=$?; cp -a "${work}/." "${result}/" 2>/dev/null || true; exit ${status}' EXIT

sha256sum "${archive}" > "${work}/archive.sha256"
tar -xf "${archive}" -C "${work}"
sha256sum \
    "${work}/SIAB/opt_orb_pytorch_dpsi/galerkin_sternheimer.py" \
    "${work}/SIAB/tests/test_galerkin_sternheimer.py" \
    > "${work}/source.sha256"

{
    date --iso-8601=seconds
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
        test_projected_pi \
        test_projected_pi_optimization \
        > "${work}/run${index}.stdout.txt" \
        2> "${work}/run${index}.stderr.txt"
    local rc=$?
    set -e
    printf '%s\n' "${rc}" > "${work}/run${index}.rc"
    if [[ "${rc}" -ne 0 ]]; then
        return "${rc}"
    fi
    grep -Fq "Ran 49 tests" "${work}/run${index}.stderr.txt"
    grep -Fxq "OK" "${work}/run${index}.stderr.txt"
    sed -E '/^Ran 49 tests in [0-9.]+s$/d' \
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
