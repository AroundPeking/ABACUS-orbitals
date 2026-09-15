"""Reduce eight direct Ec gradients and prepare one fixed-rank jY step."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys


REPO = Path(__file__).resolve().parents[4]
OPTIMIZER = REPO / "SIAB/opt_orb_pytorch_dpsi"
WORKFLOW = Path(__file__).resolve().parent
for path in (OPTIMIZER, WORKFLOW):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from c_jy_full_q_gradient import (  # noqa: E402
    PROFILE,
    propose_full_q_ec_step,
    reduce_full_q_energy_gradients,
)
from collect_c_jy_compressed_rank_ladder import (  # noqa: E402
    validate_q_source_commits,
)
from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
    write_periodic_optimizer_coefficients,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("exact " + name + " SHA256 required")
    return value


def _finite(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(name + " must be finite")
    return float(value)


def write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                    encoding="ascii")


def validate_baseline_reproduction(
        reduced, *, expected_candidate_energy_ha,
        expected_reference_energy_ha, tolerance_ha):
    """Require zero padding to reproduce both accepted full-q energies."""
    expected_candidate = _finite(expected_candidate_energy_ha,
                                 "expected candidate energy")
    expected_reference = _finite(expected_reference_energy_ha,
                                 "expected reference energy")
    tolerance = _finite(tolerance_ha, "baseline tolerance")
    if tolerance <= 0:
        raise ValueError("baseline tolerance must be positive")
    candidate = _finite(reduced.get("candidate_energy_ha"),
                        "candidate energy")
    reference = _finite(reduced.get("reference_energy_ha"),
                        "reference energy")
    candidate_difference = candidate - expected_candidate
    reference_difference = reference - expected_reference
    if (abs(candidate_difference) > tolerance
            or abs(reference_difference) > tolerance):
        raise ValueError("zero-padding baseline does not reproduce accepted energy")
    return dict(
        zero_padding_baseline_gate="pass",
        expected_candidate_energy_ha=expected_candidate,
        expected_reference_energy_ha=expected_reference,
        candidate_energy_difference_ha=candidate_difference,
        reference_energy_difference_ha=reference_difference,
        tolerance_ha=tolerance)


def serialize_proposal(
        proposal, *, gradient_sha256, input_coefficient_sha256,
        output_coefficient_sha256):
    """Remove live tensors and bind a proposal to its immutable artifacts."""
    result = {key: value for key, value in proposal.items()
              if key not in ("coefficients", "direction")}
    result.update(
        gradient_sha256=_sha256(gradient_sha256, "gradient"),
        input_coefficient_sha256=_sha256(
            input_coefficient_sha256, "input coefficient"),
        output_coefficient_sha256=_sha256(
            output_coefficient_sha256, "output coefficient"))
    return result


def _validate_source(contract):
    manifest = json.loads(
        (REPO / "SOURCE_MANIFEST.json").read_text(encoding="ascii"))
    if manifest.get("commit") != contract.get("source_commit"):
        raise ValueError("immutable source mismatch")
    for name, expected in manifest.get("files", {}).items():
        if sha(REPO / name) != expected:
            raise ValueError("source hash mismatch: " + name)


def run(contract_path, output):
    contract_path = Path(contract_path)
    output = Path(output)
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    if (os.environ.get("C_EXECUTION_HOST") != "df_iopcas_ghj"
            or os.environ.get("SLURM_JOB_ID") != contract.get("job_id")):
        raise ValueError("registered DF collector job required")
    _validate_source(contract)
    for name, expected in contract.get("inputs", {}).items():
        if sha(name) != expected:
            raise ValueError("input hash mismatch: " + name)
    paths = [Path(path) for path in contract.get("q_result_paths", ())]
    if len(paths) != 8:
        raise ValueError("eight q gradient paths required")
    artifacts = [json.loads(path.read_text(encoding="ascii"))
                 for path in paths]
    if [row.get("q_slot") for row in artifacts] != list(range(8)):
        raise ValueError("ordered complete q gradient slots required")
    q_source_commits = validate_q_source_commits(
        artifacts, contract.get("q_source_commits"))
    coefficient_path = Path(contract["coefficient_path"])
    coefficient_sha256 = _sha256(
        contract["coefficient_sha256"], "input coefficient")
    if sha(coefficient_path) != coefficient_sha256:
        raise ValueError("input coefficient hash mismatch")
    coefficients = read_periodic_optimizer_coefficients(
        coefficient_path, element="C", radial_rows=48,
        expected_nu=PROFILE)

    output.mkdir()
    try:
        reduced = reduce_full_q_energy_gradients(
            artifacts, coefficients, coefficient_sha256=coefficient_sha256,
            source_radial_rows=31, radial_rows=48)
        baseline = validate_baseline_reproduction(
            reduced,
            expected_candidate_energy_ha=contract[
                "expected_candidate_energy_ha"],
            expected_reference_energy_ha=contract[
                "expected_reference_energy_ha"],
            tolerance_ha=contract.get("baseline_tolerance_ha", 1e-12))
        reduced.update(baseline)
        reduced.update(
            source_commit=contract["source_commit"],
            q_source_commits=q_source_commits,
            input_result_sha256={str(path): sha(path) for path in paths})
        write_new(output / "FULL_Q_GRADIENT.json", reduced)
        gradient_sha256 = sha(output / "FULL_Q_GRADIENT.json")

        proposal = propose_full_q_ec_step(
            coefficients, reduced["energy_gradient"],
            radius=contract.get("trust_radius", .02))
        coefficient_output = output / "COEFFICIENTS.txt"
        write_periodic_optimizer_coefficients(
            coefficient_output, proposal["coefficients"])
        output_coefficient_sha256 = sha(coefficient_output)
        serialized = serialize_proposal(
            proposal, gradient_sha256=gradient_sha256,
            input_coefficient_sha256=coefficient_sha256,
            output_coefficient_sha256=output_coefficient_sha256)
        serialized.update(
            zero_padding_baseline_gate="pass",
            actual_full_q_energy="pending",
            physical_release_gate="hold")
        write_new(output / "PROPOSAL.json", serialized)
        write_new(output / "RESULT.json", dict(
            status="success",
            scope="expanded_45ao_full_q_ec_gradient_proposal",
            zero_padding_baseline_gate="pass",
            gradient_sha256=gradient_sha256,
            input_coefficient_sha256=coefficient_sha256,
            output_coefficient_sha256=output_coefficient_sha256,
            proposal_sha256=sha(output / "PROPOSAL.json"),
            new_row_horizontal_gradient_squared_fraction=reduced[
                "new_row_horizontal_gradient_squared_fraction"],
            predicted_ec_delta_ha_per_cell=serialized[
                "predicted_ec_delta_ha_per_cell"],
            actual_full_q_energy="pending",
            physical_release_gate="hold"))
        write_new(output / "PROVENANCE.json", dict(
            status="success", job_id=contract["job_id"],
            source_commit=contract["source_commit"],
            q_source_commits=q_source_commits,
            contract_sha256=sha(contract_path),
            gradient_sha256=gradient_sha256,
            proposal_sha256=sha(output / "PROPOSAL.json"),
            result_sha256=sha(output / "RESULT.json"),
            max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        (output / "STATUS").write_text("success\n", encoding="ascii")
    except Exception as error:
        write_new(output / "FAILURE.json", dict(
            status="failed", error=str(error)))
        (output / "STATUS").write_text("failed\n", encoding="ascii")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.contract, arguments.output)
