#!/bin/bash
#SBATCH --job-name=siab-rpa-hist
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=24:00:00
#SBATCH --output=/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_historical_ranking_20260805_task7_1a229d77/evidence/slurm-%j.out
#SBATCH --error=/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_historical_ranking_20260805_task7_1a229d77/evidence/slurm-%j.err

set -euo pipefail
umask 027

root=/work1/ghj/sternheimer_abacus_tests/siab_rpa_sensitive_historical_ranking_20260805_task7_1a229d77
controller="$root/controller"
wrapper_path=$(readlink -f "$0")
archive="$controller/abacus-orbitals-siab-dojo-1a229d77.tar"
code="$root/code"
inputs="$root/inputs"
runs="$root/runs"
evidence="$root/evidence"
runtime="$root/runtime"
node_scratch=/tmp/siab-task7-${SLURM_JOB_ID}
node_local_runs="$node_scratch/runs"
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python
expected_commit=1a229d77d735ed18689e77db0bba1f3b94e2308b
expected_archive_sha256=bc945bccaa1e27a1ebb0b8380e69c62d7ef992ce506953fba233c23f372fa533
analyzer_rel=SIAB/example_H_sternheimer/projected_pi_loss/analyze_rpa_sensitive_ranking.py
launcher="$controller/task7_torch_launcher.py"
validator="$controller/task7_validate_reproduction.py"
expected_inputs="$controller/task7_expected_inputs.tsv"

test "${SLURM_JOB_PARTITION:-}" = normal
test "${SLURM_JOB_NUM_NODES:-}" = 1
test "${SLURM_NTASKS:-}" = 1
test "${SLURM_CPUS_PER_TASK:-}" = 30
test "${SLURM_MEM_PER_NODE:-}" = 110610

export OMP_NUM_THREADS=30
export MKL_NUM_THREADS=30
export OPENBLAS_NUM_THREADS=30
export NUMEXPR_NUM_THREADS=30
export VECLIB_MAXIMUM_THREADS=30
export PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$node_scratch/mplconfig"
export TMPDIR="$node_scratch/tmp"
test ! -e "$node_scratch"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR" "$node_local_runs" "$runs" "$runtime"
trap 'rm -rf "$node_scratch"' EXIT

actual_archive_sha256=$(sha256sum "$archive" | awk '{print $1}')
test "$actual_archive_sha256" = "$expected_archive_sha256"
test ! -e "$code"
mkdir "$code"
tar -xf "$archive" -C "$code"
chmod -R a-w "$code"

{
    printf 'commit=%s\n' "$expected_commit"
    printf 'archive=%s\n' "$archive"
    printf 'archive_sha256=%s\n' "$actual_archive_sha256"
    printf 'archive_bytes=%s\n' "$(stat -c %s "$archive")"
    printf 'archive_member_count=%s\n' "$(tar -tf "$archive" | wc -l)"
    printf 'archive_member_list_sha256=%s\n' "$(tar -tf "$archive" | LC_ALL=C sort | sha256sum | awk '{print $1}')"
    printf 'analyzer_sha256=%s\n' "$(sha256sum "$code/$analyzer_rel" | awk '{print $1}')"
    printf 'launcher_sha256=%s\n' "$(sha256sum "$launcher" | awk '{print $1}')"
    printf 'validator_sha256=%s\n' "$(sha256sum "$validator" | awk '{print $1}')"
    printf 'wrapper_sha256=%s\n' "$(sha256sum "$wrapper_path" | awk '{print $1}')"
} > "$evidence/source_manifest.txt"
find "$code/SIAB/opt_orb_pytorch_dpsi" -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > "$evidence/optimizer_source_sha256_manifest.txt"
sha256sum "$code/$analyzer_rel" >> "$evidence/optimizer_source_sha256_manifest.txt"

verify_inputs() {
    local output=$1
    : > "$output"
    while IFS=$'\t' read -r label expected_sha256 source_path; do
        local staged="$inputs/$label"
        test -L "$staged"
        test "$(readlink "$staged")" = "$source_path"
        local actual_sha256
        actual_sha256=$(sha256sum "$staged" | awk '{print $1}')
        test "$actual_sha256" = "$expected_sha256"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$label" "$expected_sha256" "$actual_sha256" \
            "$(readlink "$staged")" "$(readlink -f "$staged")" \
            "$(stat -Lc %s "$staged")" >> "$output"
    done < "$expected_inputs"
    test "$(wc -l < "$output")" = 11
}

verify_inputs "$evidence/input_sha256_manifest.before.txt"

{
    date --iso-8601=seconds
    hostname
    printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
    printf 'SLURM_JOB_PARTITION=%s\n' "$SLURM_JOB_PARTITION"
    printf 'SLURM_JOB_NUM_NODES=%s\n' "$SLURM_JOB_NUM_NODES"
    printf 'SLURM_NTASKS=%s\n' "$SLURM_NTASKS"
    printf 'SLURM_CPUS_PER_TASK=%s\n' "$SLURM_CPUS_PER_TASK"
    printf 'SLURM_MEM_PER_NODE=%s\n' "$SLURM_MEM_PER_NODE"
    printf 'SLURM_JOB_NODELIST=%s\n' "$SLURM_JOB_NODELIST"
    taskset -pc $$
    scontrol show job "$SLURM_JOB_ID"
} > "$evidence/slurm_resources.txt"
{
    env | LC_ALL=C sort | grep -E '^(OMP|MKL|OPENBLAS|NUMEXPR|VECLIB|PYTHONPATH|PYTHONDONTWRITEBYTECODE|MPLCONFIGDIR|TMPDIR)='
} > "$evidence/thread_settings.txt"
"$python" "$launcher" --versions > "$evidence/runtime_versions.json"

run_analyzer() {
    local run_name=$1
    local output_dir="$node_local_runs/$run_name"
    test ! -e "$output_dir"
    set +e
    /usr/bin/time -v -o "$evidence/time-$run_name.txt" \
        "$python" "$launcher" "$code/$analyzer_rel" \
        --h-response "$inputs/H_response" \
        --h2-response "$inputs/H2_response" \
        --h-source "$inputs/H_source" \
        --h2-source "$inputs/H2_source" \
        --h-audit "$inputs/H_audit" \
        --h2-audit "$inputs/H2_audit" \
        --two-d "$inputs/two_d" \
        --first-f "$inputs/first_f" \
        --first-g "$inputs/first_g" \
        --second-f "$inputs/second_f" \
        --second-g "$inputs/second_g" \
        --output-dir "$output_dir" \
        > "$evidence/analyzer-$run_name.stdout.txt" \
        2> "$evidence/analyzer-$run_name.stderr.txt"
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$evidence/analyzer-$run_name.rc"
    if test "$rc" != 0 && test "$rc" != 2; then
        return 1
    fi
}

run_analyzer run1
verify_inputs "$evidence/input_sha256_manifest.between.txt"
cmp "$evidence/input_sha256_manifest.before.txt" "$evidence/input_sha256_manifest.between.txt"
run_analyzer run2
verify_inputs "$evidence/input_sha256_manifest.after.txt"
cmp "$evidence/input_sha256_manifest.before.txt" "$evidence/input_sha256_manifest.after.txt"

rc1=$(cat "$evidence/analyzer-run1.rc")
rc2=$(cat "$evidence/analyzer-run2.rc")

publish_run() {
    local run_name=$1
    local source_dir="$node_local_runs/$run_name"
    local destination_dir="$runs/$run_name"
    local staging_dir="$runs/.$run_name.staging-$SLURM_JOB_ID"
    test -d "$source_dir"
    test ! -e "$destination_dir"
    test ! -e "$staging_dir"
    mkdir "$staging_dir"
    cp -a "$source_dir/." "$staging_dir/"
    mv -T "$staging_dir" "$destination_dir"
}

publish_run run1
publish_run run2

"$python" "$validator" \
    --run1 "$runs/run1" \
    --run2 "$runs/run2" \
    --rc1 "$rc1" \
    --rc2 "$rc2" \
    --output "$evidence/reproduction_summary.json"

find "$runs/run1" "$runs/run2" -maxdepth 1 -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > "$evidence/output_sha256_manifest.txt"
date --iso-8601=seconds > "$evidence/completed.ok"
