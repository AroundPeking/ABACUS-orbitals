# Frozen-Space RPA Optimizer

`optimize_periodic_basis.py --objective rpa` uses the existing differentiable
fitter with Pi, local trace-log and integrated body correlation-energy errors.
The default remains `--objective pi`. RPA mode rejects atomic inputs and
explicit family balancing; q weights retain their physical values.

For a two-step engineering trial on an accepted single-q dataset:

```sh
python optimize_periodic_basis.py \
  --dataset "$ACCEPTED_DATASET" --initial "$ORIGINAL_COEFFICIENTS" \
  --output-directory "$NEW_RESULT_DIRECTORY" --siab-commit "$FULL_COMMIT" \
  --objective rpa --allow-partial-q \
  --rpa-pi-weight 1 --rpa-trace-log-weight 1 --rpa-energy-weight 1 \
  --nu 3,3,2,0,0 --fixed-nu 2,2,1,0,0 --radial-rows 31 --max-l 4 \
  --max-steps 2 --minimum-steps 0 --learning-rate 0.001 \
  --block-cache-workers 1 \
  --omitted-reference-projection-validation sha256
```

Equal objective weights are uncalibrated engineering defaults. A partial-q
run requires the explicit flag, reports its actual coverage, and never
renormalizes that coverage to one. The energy excludes head/wing. Even a full-q
result keeps `physical_release_gate=hold`: it is not candidate SCF, ordinary
ABACUS/LRI SOS, or final qavg validation.

The reader omits the large reference-projection tensors from resident memory
but verifies their hashes by default. H/S, occupied projections, sources,
Coulomb transforms and reference Pi remain available. Use a compute node for
real datasets; a synthetic unit-test runtime is not a C timing estimate.

Acceptance of an engineering trial requires scheduler completion, STATUS
success, finite loss/gradients, preserved fixed prefix and occupied-capture
floor, and identical final/best coefficient hashes. The result records the
actual Pi error separately from joint RPA loss, initial/best q-frequency
contributions, elapsed time and peak process RSS. Complete reference coverage
and independent physical validation remain separate requirements.
