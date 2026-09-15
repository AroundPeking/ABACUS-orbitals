"""Embed compact C coefficients exactly in a larger nested jY radial mother."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO/'SIAB/opt_orb_pytorch_dpsi'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
    write_periodic_optimizer_coefficients,
)
from prepare_c_jy_operator_restart import zero_pad_radial_coefficients  # noqa: E402


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ao_per_element(profile):
    return sum((2*l+1)*count for l, count in enumerate(profile))


def export_expanded_coefficients(source, output, manifest, *, profile,
                                 source_rows, target_rows):
    source = Path(source).resolve()
    output = Path(output).resolve()
    manifest = Path(manifest).resolve()
    if output.exists() or manifest.exists():
        raise FileExistsError(output if output.exists() else manifest)
    profile = tuple(profile)
    original = read_periodic_optimizer_coefficients(
        source, element='C', radial_rows=source_rows, expected_nu=profile)
    expanded = zero_pad_radial_coefficients(
        original, source_rows=source_rows, target_rows=target_rows)
    write_periodic_optimizer_coefficients(output, expanded)
    recovered = read_periodic_optimizer_coefficients(
        output, element='C', radial_rows=target_rows, expected_nu=profile)
    prefix_exact = all(torch.equal(before, after[:source_rows])
                       for before, after in zip(original['C'], recovered['C']))
    appended_zero = all(int(torch.count_nonzero(after[source_rows:])) == 0
                        for after in recovered['C'])
    if not prefix_exact or not appended_zero:
        raise ValueError('expanded coefficient roundtrip changed the old basis')
    result = dict(status='success', element='C', profile=list(profile),
        ao_per_C=ao_per_element(profile), source_radial_rows=source_rows,
        expanded_radial_rows=target_rows,
        old_parameter_count=source_rows*sum(profile),
        new_parameter_count=(target_rows-source_rows)*sum(profile),
        total_parameter_count=target_rows*sum(profile),
        prefix_exact=True, appended_rows_exactly_zero=True,
        source_coefficients=str(source), source_coefficients_sha256=sha(source),
        expanded_coefficients=str(output),
        expanded_coefficients_sha256=sha(output),
        physical_release_gate='hold', ordinary_sos_validated=False)
    manifest.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n',
                        encoding='ascii')
    return result


def parse_profile(value):
    try:
        profile = tuple(int(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('profile must contain integers') from error
    if len(profile) != 5 or not any(profile) or any(item < 0 for item in profile):
        raise argparse.ArgumentTypeError('profile must have five nonnegative entries')
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--profile', type=parse_profile, required=True)
    parser.add_argument('--source-rows', type=int, required=True)
    parser.add_argument('--target-rows', type=int, required=True)
    args = parser.parse_args(argv)
    result = export_expanded_coefficients(
        args.source, args.output, args.manifest, profile=args.profile,
        source_rows=args.source_rows, target_rows=args.target_rows)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
