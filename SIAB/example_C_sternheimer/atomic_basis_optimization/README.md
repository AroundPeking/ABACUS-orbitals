# C atomic response-basis optimization

This campaign continues the accepted 20-step C `3s3p2d` atomic Sternheimer
checkpoint to a machine-audited plateau. It freezes the SG15 DZP prefix and
varies only `3s`, `3p`, and `2d` with the same full-frequency FD8 target.

The continuation is not a released basis. Only a `TZDP_CONVERGED` result may
enter the angular-residual analysis. That analysis ranks `l=0,...,4` by the
leading remaining response eigenvalue divided by the number of added AOs and
constructs exactly one deterministic leading-mode shell seed. Projected-Pi,
C-C, held-out-volume, and raw SOS-RPA checks remain separate later gates.

The magnetic-channel overlap tolerance is `3e-4`. The accepted FD8 C target
shows relative deviations from zero through `2.05375e-4` as `l` increases
from 0 to 4; this threshold admits that measured Cartesian-grid anisotropy
while retaining an explicit diagnostic in every channel record.

The df runner uses one `p1` task with 8 CPU threads and 16 GiB. The submitter
preflights the resource request and refuses existing receipts or outputs.

## First f-shell gate

`SIAB_INPUT.f_shell_optimization.json` starts from the deterministic leading
`l=3` residual mode, keeps the same DZP prefix frozen, and jointly optimizes
`3s`, `3p`, `2d`, and `1f`. The numerical gate is unchanged: the final-100
best-loss reduction must be below `1e-4`, or the optimizer must report 51
nonimproving accepted steps, with maximum overlap condition below `1e12`.

Numerical convergence alone does not promote the basis. The converged loss must
also improve on the accepted `3s3p2d` loss `0.7114382940310687` by at least 1%.
Only `F_SHELL_MATERIAL_GAIN` may enter C-C and projected-Pi validation;
`F_SHELL_MARGINAL_GAIN` stops this shell-growth route before SOS-RPA.
