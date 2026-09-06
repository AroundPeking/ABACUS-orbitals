#!/bin/bash
#SBATCH --job-name=c_radial_grad_diag
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=105000M
#SBATCH --time=01:00:00
#SBATCH --no-requeue

source ~/.bashrc >/dev/null 2>&1 || true
set -euo pipefail
export LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
: "${C_RADIAL_STAGE:?}"
: "${C_RADIAL_SOURCE_COMMIT:?}"
: "${C_RADIAL_ARCHIVE_SHA256:?}"
root=/work1/ghj/c-solid-fd8-q13-standard-20260903
stage=$C_RADIAL_STAGE
py=$root/optimizer-env-torch1121cpu/bin/python
status=failed
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'rc=$?; printf "%s\n" "$status" > "$stage/STATUS"; exit "$rc"' EXIT
mkdir "$stage/EXECUTION_LOCK"
cd "$stage"
"$py" - "$stage" <<'PY'
import hashlib, json, os, pathlib, sys, tarfile, torch
p = pathlib.Path(sys.argv[1])
archive = p/'source.tar'
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest == os.environ['C_RADIAL_ARCHIVE_SHA256']
assert torch.__version__ == '1.12.1+cpu' and torch.version.hip is None and torch.version.cuda is None
hashes = {}
with tarfile.open(str(archive)) as src:
    for item in src.getmembers():
        if not item.isfile():
            continue
        relative = pathlib.Path(item.name)
        assert not relative.is_absolute() and '..' not in relative.parts
        h = hashlib.sha256(src.extractfile(item).read()).hexdigest()
        assert hashlib.sha256((p/'source'/relative).read_bytes()).hexdigest() == h
        hashes[item.name] = h
with (p/'PREFLIGHT.json').open('x') as f:
    json.dump(dict(status='success', source_commit=os.environ['C_RADIAL_SOURCE_COMMIT'],
        archive_sha256=digest, source_hashes=hashes, python=sys.version,
        torch=torch.__version__, scheduler_job=os.environ['SLURM_JOB_ID'],
        scope='fixed_candidate_gradients_and_frozen_band_stencils_no_new_physics',
        candidate_alpha=.25, physical_release_gate='hold'), f, indent=2)
PY
"$py" -m unittest discover -s "$stage/source/SIAB/tests" \
 -p 'test_periodic_galerkin_radial_diagnostics.py' -v > "$stage/tests.log" 2>&1
/usr/bin/time -v -o "$stage/diagnostic.time" "$py" -u \
 "$stage/source/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/diagnose_c_radial_gradients.py" \
 --candidate-result "$root/stage-c-all-radial-backoff-quarter-e3beab29/result/RESULT.json" \
 --candidate-result-sha256 720fdfeb26617020bd834a241b2393c30b452aa0e2ac7d64c4975beff92337c1 \
 --pbe-collection "$root/stage-c-all-radial-quarter-pbe-e3beab29/pbe/PBE_ENDPOINT_COLLECTION.json" \
 --pbe-collection-sha256 72b34192895333b058d000cec18989fd343f9935e72126ac4f178b29ca365d17 \
 --source-commit "$C_RADIAL_SOURCE_COMMIT" --output "$stage/result"
status=success
