# Carbon PBE Reference Gate Result

## Acceptance

The formal server66 calculation passed all scheduler, numerical, physical, and
provenance gates:

```text
status=PBE_GATE_PASSED
zero_field_comparison_status=ZERO_FIELD_COMPARISON_PASSED
restart_chain_evidence=RESTART_CHAIN_VERIFIED
blocked_on=None
```

This establishes one reproducible neutral C triplet PBE reference with exactly
3 up / 1 down integer occupations.  It authorizes a separate C Delta-ST
reference calculation and later C SIAB basis optimization.  It does not itself
run Delta-ST or establish an RPA result.

## Provenance

| Item | Formal value |
| --- | --- |
| Date | 2026-08-22 |
| Slurm array | `410668` |
| Stable job name | `c_pbe_gate_507c214ec6a7` |
| Immutable root | `/home/ghj/abacus/260822/c-atom-pbe-equivalence-server66-7527d03bb187` |
| Source commit | `7527d03bb1875cea04e9a3ec415060276d8a5ea7` |
| Source archive SHA256 | `843a181ff4872c0f272c979682f388ba8ba38baac89f63b14ec315e2a2ef8c15` |
| ABACUS version | `v3.9.0.25` |
| ABACUS SHA256 | `27722d5e3e5cf2c94d00ac9489152b7ea00adcf51a8b8bb3a8eed3d8d094c279` |
| SG15 C pseudopotential SHA256 | `e95d682a8b918557fb57e2e0ec11b2f48cf693cb72a11d078cf07ec489a8fa99` |
| TZDP-10au C orbital SHA256 | `7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d` |
| Local/server66 preflight | `Ran 176 tests`, `OK` |

Every array task used partition `640`, one node, one MPI rank, 48 OpenMP
threads, 180000 MB, and a 24-hour limit.  Scheduler accounting was:

| Branch | Slurm task record | Node | Elapsed | State / exit |
| --- | ---: | --- | ---: | --- |
| `fixed` | `410669` | `cpu08` | 00:01:37 | `COMPLETED / 0:0` |
| `dir0` | `410670` | `cpu09` | 00:02:14 | `COMPLETED / 0:0` |
| `dir1` | `410671` | `cpu10` | 00:02:11 | `COMPLETED / 0:0` |
| `dir2` | `410668` | `cpu08` | 00:02:25 | `COMPLETED / 0:0` |

## Physical Results

The field amplitude was `1e-4`.  Field-bearing phases select an orientation of
the degenerate C 2p manifold.  Their energies are not zero-field reference
energies.

| Phase | SCF iterations | Final `drho` | Wall time (s) | Energy (eV) |
| --- | ---: | ---: | ---: | ---: |
| `fixed/fixed_field_seed` | 44 | `4.3215e-11` | 49 | `-147.4773377745117` |
| `fixed/fixed_zero_restart` | 28 | `8.8814e-11` | 34 | `-147.4773363622957` |
| `dir0/field_seed` | 49 | `8.3108e-11` | 53 | `-147.4773377745111` |
| `dir0/free_restart1` | 28 | `7.7107e-11` | 35 | `-147.4773363622966` |
| `dir0/free_restart2` | 18 | `6.8116e-11` | 24 | `-147.4773363622953` |
| `dir1/field_seed` | 45 | `9.6789e-11` | 49 | `-147.4773377744943` |
| `dir1/free_restart1` | 29 | `9.0958e-11` | 35 | `-147.4773363622971` |
| `dir1/free_restart2` | 18 | `7.4357e-11` | 24 | `-147.4773363622991` |
| `dir2/field_seed` | 54 | `3.8870e-11` | 58 | `-147.4773377744951` |
| `dir2/free_restart1` | 29 | `8.3387e-11` | 35 | `-147.4773363622958` |
| `dir2/free_restart2` | 26 | `6.0455e-11` | 31 | `-147.4773363622955` |

All 11 phases contain exactly 3 up / 1 down integer occupations.  The final
comparison uses only `fixed_zero_restart` and the three `free_restart2`
energies:

| Quantity | Observed | Required |
| --- | ---: | ---: |
| Fixed zero-field energy | `-5.419692147585443 Ha` | finite |
| Fixed seed-to-zero-restart drift | `3.256647533451255e-5 kcal/mol` | `< 0.001 kcal/mol` |
| Maximum free restart drift | `4.570191058765812e-11 kcal/mol` | `< 0.001 kcal/mol` |
| Maximum fixed/free zero-field difference | `1.252331571777177e-13 Ha` | `< 1e-5 Ha` |
| Maximum free/free zero-field difference | `1.394440118929197e-13 Ha` | `< 1e-5 Ha` |

The field-seed energy is excluded from the final physical comparison.  Its
only acceptance role is the fixed seed-to-zero-restart drift check.

## Next Gate

The next physical task is to compute the C atomic Delta-ST first-order response
from this accepted PBE state, then optimize a compact C SIAB basis and compare
its SOS response against the Delta-ST reference.  Delta-ST must use this same
cell, grid, pseudopotential, zero-field triplet state, frequency definition,
and auxiliary-basis definition unless a one-variable convergence test changes
one item explicitly.
