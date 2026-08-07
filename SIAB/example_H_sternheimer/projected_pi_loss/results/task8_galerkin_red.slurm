#!/bin/bash
#SBATCH --job-name=siab-galerkin-red
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=24:00:00

set -euo pipefail

campaign=/work1/ghj/sternheimer_abacus_tests/siab_galerkin_red_20260807
archive=${campaign}/input/abacus-orbitals-red.tar
work=/tmp/siab-galerkin-red-${SLURM_JOB_ID}
result=${campaign}/jobs/${SLURM_JOB_ID}
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python

mkdir -p "${work}" "${result}"
trap 'status=$?; cp -a "${work}/." "${result}/" 2>/dev/null || true; exit ${status}' EXIT

sha256sum "${archive}" > "${work}/archive.sha256"
tar -xf "${archive}" -C "${work}"

{
    date --iso-8601=seconds
    printf 'job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'node_list=%s\n' "${SLURM_JOB_NODELIST}"
    printf 'cpus_per_task=%s\n' "${SLURM_CPUS_PER_TASK}"
    "${python}" --version
    "${python}" -c 'import torch; print("torch=" + torch.__version__)'
} > "${work}/environment.txt" 2>&1

cd "${work}/SIAB/tests"
set +e
"${python}" -m unittest -v \
    test_galerkin_sternheimer.GalerkinSternheimerTest.test_two_level_analytic_response \
    > "${work}/red.stdout.txt" 2> "${work}/red.stderr.txt"
rc=$?
set -e
printf '%s\n' "${rc}" > "${work}/red.rc"

if [[ "${rc}" -eq 0 ]]; then
    echo "RED test unexpectedly passed" >&2
    exit 1
fi
if ! grep -Fq "No module named 'galerkin_sternheimer'" "${work}/red.stderr.txt"; then
    echo "RED test failed for an unexpected reason" >&2
    cat "${work}/red.stderr.txt" >&2
    exit 1
fi

touch "${work}/red_expected.ok"
