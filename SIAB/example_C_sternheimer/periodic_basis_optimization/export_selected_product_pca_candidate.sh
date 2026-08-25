#!/bin/bash

set -euo pipefail
trap 'status=$?; echo "C24_SELECTED_PRODUCT_PCA_EXPORT_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load python/3.9.22

runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
code=/data/home/df_iopcas_ghj/app/siab/periodic-c-e4f1914a
campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
siab_commit=e4f1914ade0308ebb3f207a3867461db9d35c30f
selector=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/select_periodic_candidate.py
exporter=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/export_periodic_orbitals.py
heldout=$campaign/runs/product-pca-20260825/heldout-q3-fixed-prefix-layout-checkpoint-500
comparison=$heldout/COMPARISON_RESULT.json
output=$campaign/runs/product-pca-20260825/selected-fixed-prefix-checkpoint-500
selection=$output/SELECTED_CANDIDATE.json
orbital=$output/C_gga_10au_100Ry_selected_product_pca.orb

test "$(cat "$code/.git/HEAD")" = "$siab_commit"
test "$(cat "$code/SIAB_COMMIT")" = "$siab_commit"
test -s "$selector"
test -s "$exporter"
test -s "$comparison"
grep -qx 'status=success' "$heldout/provenance.txt"
test ! -e "$output"
mkdir -p "$output"

export PYTHONPATH=$runtime/lib-dynload:$runtime/site-packages:$code
export LD_LIBRARY_PATH=$runtime/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python3 "$selector" \
  --comparison "$comparison" \
  --output "$selection" \
  --occupied-capture-floor 0.9998982409775239 \
  > "$output/selection.log"

mapfile -t selected < <(python3 - "$selection" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], encoding="ascii"))
print(selection["label"])
print(",".join(str(value) for value in selection["nu"]))
print(selection["ao_count_cell"])
print(selection["coefficients"])
PY
)
label=${selected[0]}
nu=${selected[1]}
ao_count=${selected[2]}
coefficients=${selected[3]}
case "$label" in
  joint-two-g)
    test "$nu" = 3,3,2,1,2
    test "$ao_count" = 94
    ;;
  joint-three-g)
    test "$nu" = 3,3,2,1,3
    test "$ao_count" = 112
    ;;
  *)
    echo "unsupported selected label: $label" >&2
    exit 2
    ;;
esac

python3 "$exporter" \
  --input "$coefficients" \
  --output "$orbital" \
  --element C \
  --nu "$nu" \
  --radial-rows 31 \
  --ecut-ry 100 \
  --rcut-bohr 10 \
  --dr-bohr 0.01 \
  --smoothing-sigma-bohr 0.1 \
  > "$output/EXPORT_METADATA.json"

test -s "$orbital"
grep -qx 'Energy Cutoff(Ry)           100.0' "$orbital"
grep -qx 'Radius Cutoff(a.u.)         10.0' "$orbital"
grep -qx 'Lmax                        4' "$orbital"
python3 - "$selection" "$output/EXPORT_METADATA.json" "$orbital" <<'PY'
import hashlib
import json
import pathlib
import sys

selection_path = pathlib.Path(sys.argv[1])
metadata_path = pathlib.Path(sys.argv[2])
orbital_path = pathlib.Path(sys.argv[3])
selection = json.loads(selection_path.read_text(encoding="ascii"))
metadata = json.loads(metadata_path.read_text(encoding="ascii"))
assert metadata["nu"] == selection["nu"]
selection.update(
    {
        "exported_orbital": str(orbital_path.resolve()),
        "exported_orbital_sha256": hashlib.sha256(orbital_path.read_bytes()).hexdigest(),
        "export_metadata": str(metadata_path.resolve()),
        "export_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }
)
temporary = selection_path.with_name(".SELECTED_CANDIDATE.json.tmp")
temporary.write_text(
    json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="ascii",
)
temporary.replace(selection_path)
PY

{
  echo status=success
  echo purpose=periodic_c_select_and_export_product_pca_candidate
  echo siab_commit=$siab_commit
  echo optimizer_array_job=${OPTIMIZER_ARRAY_JOB_ID:?missing optimizer array job id}
  echo heldout_job=${HELDOUT_JOB_ID:?missing heldout job id}
  echo selected_label=$label
  echo selected_nu=$nu
  echo selected_ao_count_cell=$ao_count
  echo ecut_ry=100
  echo rcut_bohr=10
  echo dr_bohr=0.01
  echo smoothing_sigma_bohr=0.1
  sha256sum "$0" "$comparison" "$selection" "$coefficients" "$selector" \
    "$exporter" "$output/EXPORT_METADATA.json" "$orbital"
} > "$output/provenance.txt"

echo "C24_SELECTED_PRODUCT_PCA_EXPORT_OK label=$label orbital=$orbital"
