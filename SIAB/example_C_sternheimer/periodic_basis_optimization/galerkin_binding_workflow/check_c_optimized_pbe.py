"""Prepare/collect a C2 solid PBE endpoint without launching or judging a job."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


PREPARATION = "PBE_ENDPOINT_PREPARED.json"
COLLECTION = "PBE_ENDPOINT_COLLECTION.json"
SCOPE = "c_solid_only_pbe_endpoint"
TOLERANCE_EV_PER_C = 0.010
FROZEN_NUMERICS = {
    "nspin": 1, "ecutwfc": 45, "lcao_ecut": 100, "nx": 24, "ny": 24,
    "nz": 24, "kpar": 4, "nbands": 44, "symmetry": 1,
}


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _read_hashed(path, expected):
    content = Path(path).read_bytes()
    if not isinstance(expected, str) or _sha256(content) != expected:
        raise ValueError("SHA256 mismatch: " + str(path))
    return content


def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be finite") from error
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(name + " must be finite")
    return number


def _lines(content):
    return [line for line in (
        re.split(r"#|//", raw, maxsplit=1)[0].strip()
        for raw in content.decode("utf-8").splitlines()
    ) if line]


def _input_values(content):
    lines = _lines(content)
    if not lines or lines[0] != "INPUT_PARAMETERS":
        raise ValueError("missing INPUT_PARAMETERS header")
    values = {}
    for line in lines[1:]:
        words = line.split()
        key = words[0].lower()
        if len(words) < 2 or key in values:
            raise ValueError("invalid or duplicate INPUT key: " + key)
        values[key] = " ".join(words[1:])
    return values


def read_input(path):
    return _input_values(Path(path).read_bytes())


def _basename(value):
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value)
            or value in (".", "..")):
        raise ValueError("expected a local filename or suffix")
    return value


def _c2_files(content):
    names = ("ATOMIC_SPECIES", "NUMERICAL_ORBITAL", "LATTICE_CONSTANT",
             "LATTICE_VECTORS", "ATOMIC_POSITIONS")
    sections, current = {}, None
    for line in _lines(content):
        if line == "ABFS_ORBITAL":
            raise ValueError("pure PBE endpoint does not accept ABFS_ORBITAL")
        if line in names:
            if line in sections:
                raise ValueError("duplicate STRU section")
            current = line
            sections[current] = []
        elif current is None:
            raise ValueError("invalid STRU section")
        else:
            sections[current].append(line)
    if set(sections) != set(names) or len(sections["ATOMIC_SPECIES"]) != 1:
        raise ValueError("only a single-species C2 solid STRU is supported")
    species = sections["ATOMIC_SPECIES"][0].split()
    positions = sections["ATOMIC_POSITIONS"]
    if (len(species) not in (3, 4) or species[0] != "C" or len(positions) != 6
            or positions[0] not in ("Direct", "Cartesian", "Cartesian_angstrom", "Cartesian_au")
            or positions[1] != "C" or positions[3] != "2"
            or _finite(positions[2], "C magnetization") != 0):
        raise ValueError("only a nonmagnetic C2 solid is supported")
    if _finite(species[1], "C mass") <= 0:
        raise ValueError("invalid C mass")
    for row in positions[4:]:
        if len(row.split()) < 3:
            raise ValueError("invalid C coordinates")
        for value in row.split()[:3]:
            _finite(value, "C coordinate")
    constant, lattice = sections["LATTICE_CONSTANT"], sections["LATTICE_VECTORS"]
    if len(constant) != 1 or _finite(constant[0], "lattice constant") <= 0 or len(lattice) != 3:
        raise ValueError("invalid solid lattice")
    matrix = [[_finite(v, "lattice vector") for v in row.split()] for row in lattice]
    if any(len(row) != 3 for row in matrix):
        raise ValueError("invalid lattice vectors")
    a, b, c = matrix
    determinant = (a[0] * (b[1]*c[2] - b[2]*c[1]) - a[1] * (b[0]*c[2] - b[2]*c[0])
                   + a[2] * (b[0]*c[1] - b[1]*c[0]))
    if determinant == 0:
        raise ValueError("singular solid lattice")
    orbitals = sections["NUMERICAL_ORBITAL"]
    if len(orbitals) != 1:
        raise ValueError("one C orbital file is required")
    pseudo, orbital = _basename(species[2]), _basename(orbitals[0])
    if pseudo == orbital or {pseudo, orbital} & {"INPUT", "STRU", "KPT", PREPARATION}:
        raise ValueError("input filenames collide")
    return pseudo, orbital


def _pure_pbe(values):
    for key, expected in FROZEN_NUMERICS.items():
        if _finite(values.get(key), key) != expected:
            raise ValueError("frozen INPUT mismatch: " + key)
    for key, expected in (("basis_type", "lcao"), ("calculation", "scf"),
                          ("smearing_method", "fixed")):
        if values.get(key, "").lower() != expected:
            raise ValueError("frozen INPUT mismatch: " + key)
    if values.get("dft_functional", "pbe").lower() != "pbe":
        raise ValueError("only PBE is supported")
    for key in ("dft_plus_u", "deepks_scf", "lspinorb", "noncolin"):
        if values.get(key, "0").lower() not in ("0", "false"):
            raise ValueError("pure nspin=1 PBE excludes active " + key)
    if values.get("ntype", "1") != "1":
        raise ValueError("only one C species is supported")
    for key, default in (("stru_file", "STRU"), ("kpoint_file", "KPT")):
        if values.get(key, default) != default:
            raise ValueError("original must use " + default)
    dropped = ("sternheimer", "out_sternheimer", "bessel", "restart", "exx_", "rpa_", "out_librpa")
    result = {key: value for key, value in values.items()
              if not key.startswith(dropped) and key not in ("read_file_dir", "read_wfc", "charge_extrap")}
    result.update(calculation="scf", dft_functional="pbe", rpa="0", nspin="1",
                  out_chg="1", out_wfc_lcao="1", init_chg="atomic", init_wfc="atomic",
                  pseudo_dir="./", orbital_dir="./")
    return result


def _scf_log(content):
    text = content.decode("utf-8")
    converged = list(re.finditer(r"#SCF IS CONVERGED#", text, re.I))
    failed = re.search(r"SCF\s+(?:IS\s+)?NOT\s+CONVERGED|convergence\s+has\s+not|"
                       r"(?:SCF|charge\s+density)\s+(?:did\s+not|failed\s+to)\s+converge", text, re.I)
    if len(converged) != 1 or failed:
        raise ValueError("SCF log is not unambiguously converged")
    records = list(re.finditer(r"^\s*!FINAL_ETOT_IS\b[^\r\n]*", text, re.M))
    if len(records) != 1 or converged[0].start() > records[0].start():
        raise ValueError("expected exactly one final energy after SCF convergence")
    match = re.fullmatch(r"\s*!FINAL_ETOT_IS\s+(\S+)\s+eV\s*", records[0].group())
    if match is None:
        raise ValueError("final PBE total energy must be in eV")
    counts = {}
    for name, label in (
        ("occupied", r"(?:Occupied electronic states|Number of occupied bands)"),
        ("total", r"(?:Number of electronic states \(NBANDS\)|NBANDS)"),
    ):
        matches = re.findall(r"^\s*" + label + r"\s*(?:=|:)?\s+(\S+)\s*$", text, re.M | re.I)
        count = _finite(matches[-1], name + " band count") if matches else None
        if count is not None and (count <= 0 or not count.is_integer()):
            raise ValueError("invalid " + name + " band count")
        counts[name] = int(count) if count is not None else None
    if counts["total"] is not None and counts["total"] != 44:
        raise ValueError("SCF band count differs from frozen nbands=44")
    if counts["occupied"] is not None and counts["occupied"] >= 44:
        raise ValueError("invalid occupied band count")
    return {"energy_ev": _finite(match.group(1), "final PBE energy"), "band_counts": counts,
            "energy_record": "!FINAL_ETOT_IS", "log_sha256": _sha256(content)}


def _optimizer_artifacts(path, expected):
    content = _read_hashed(path, expected)
    result = json.loads(content)
    if (result.get("status") != "success"
            or result.get("scope") != "optimized_full_q_frozen_body_rpa_calibration"):
        raise ValueError("completed solid optimizer result required")
    for key, expected in (("nu", [3, 3, 2, 0, 0]), ("fixed_nu", [0, 0, 0, 0, 0])):
        value = result.get(key)
        if (not isinstance(value, list) or value != expected
                or any(type(item) is not int for item in value)):
            raise ValueError("all-radial C endpoint requires {}={}".format(key, expected))
    if type(result.get("ao_per_C")) is not int or result["ao_per_C"] != 22:
        raise ValueError("all-radial C endpoint requires ao_per_C=22")
    steps, best_step = result.get("steps_completed"), result.get("best_step")
    if type(steps) is not int or type(best_step) is not int or not 0 < best_step <= steps:
        raise ValueError("optimizer must complete an improving nonzero step")
    initial, best = result.get("initial", {}), result.get("best", {})
    initial_loss = _finite(initial.get("loss"), "initial loss")
    best_loss = _finite(best.get("loss"), "best loss")
    if (not 0 <= best_loss < initial_loss or initial.get("step") != 0
            or best.get("step") != best_step):
        raise ValueError("optimizer best loss must improve over initial loss")
    root = Path(path).parent
    artifacts = {"optimizer_RESULT.json": content}
    for name in ("ORBITAL_RESULTS.txt", "BEST_ORBITAL_CHECKPOINT.txt"):
        artifacts[name] = _read_hashed(root / name, result.get("coefficient_sha256"))
    if artifacts["ORBITAL_RESULTS.txt"] != artifacts["BEST_ORBITAL_CHECKPOINT.txt"]:
        raise ValueError("final coefficients differ from best checkpoint")
    artifacts["BEST_CHECKPOINT.json"] = (root / "BEST_CHECKPOINT.json").read_bytes()
    metadata = json.loads(artifacts["BEST_CHECKPOINT.json"])
    if (metadata.get("step") != best_step or metadata.get("objective") != "rpa"
            or _finite(metadata.get("loss"), "checkpoint loss") != best_loss
            or metadata.get("orbital_file") != "BEST_ORBITAL_CHECKPOINT.txt"
            or metadata.get("orbital_sha256") != result.get("coefficient_sha256")):
        raise ValueError("best checkpoint metadata mismatch")
    orbital = _read_hashed(root / "C_3s3p2d_optimized.orb", result.get("orbital_sha256"))
    if not orbital or not artifacts["ORBITAL_RESULTS.txt"]:
        raise ValueError("empty candidate artifact")
    return result, orbital, artifacts


def _write_json(path, result):
    with Path(path).open("x", encoding="ascii") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def prepare_pbe_endpoint(*, optimizer_result, optimizer_result_sha256, original_q1,
                         baseline_log_sha256, output):
    """Stage new C2 inputs only after a hashed, improved optimizer result exists."""
    root, output = Path(original_q1).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result, orbital, artifacts = _optimizer_artifacts(Path(optimizer_result), optimizer_result_sha256)
    originals = {name: (root / name).read_bytes() for name in ("INPUT", "STRU", "KPT")}
    values = _input_values(originals["INPUT"])
    pure = _pure_pbe(values)
    pseudo_name, orbital_name = _c2_files(originals["STRU"])
    source_paths = {name: str(root / name) for name in originals}
    for name, key in ((pseudo_name, "pseudo_dir"), (orbital_name, "orbital_dir")):
        path = (root / values.get(key, ".") / name).resolve()
        originals[name] = path.read_bytes()
        source_paths[name] = str(path)
        if not originals[name]:
            raise ValueError("empty original input: " + name)
    suffix = _basename(values.get("suffix", "ABACUS"))
    baseline_path = root / ("OUT." + suffix) / "running_scf.log"
    baseline_content = _read_hashed(baseline_path, baseline_log_sha256)
    baseline = _scf_log(baseline_content)
    baseline["original_log_path"] = str(baseline_path)
    prepared = dict(originals)
    prepared["INPUT"] = ("INPUT_PARAMETERS\n" + "".join(
        "{} {}\n".format(key, value) for key, value in pure.items()
    )).encode("utf-8")
    prepared[orbital_name] = orbital
    evidence = {"original_" + name: content for name, content in originals.items()}
    evidence.update(artifacts)
    evidence["original_running_scf.log"] = baseline_content
    manifest = {
        "format_version": 1, "status": "prepared", "scope": SCOPE, "atoms_per_cell": 2,
        "original_q1": str(root), "original_input_paths": source_paths,
        "original_input_sha256": {name: _sha256(data) for name, data in originals.items()},
        "prepared_input_sha256": {name: _sha256(data) for name, data in prepared.items()},
        "evidence_sha256": {".provenance/" + name: _sha256(data) for name, data in evidence.items()},
        "candidate_orbital_filename": orbital_name, "candidate_orbital_sha256": result["orbital_sha256"],
        "candidate_coefficient_sha256": result["coefficient_sha256"],
        "optimizer_result_path": str(Path(optimizer_result).resolve()),
        "optimizer_result_sha256": optimizer_result_sha256,
        "optimizer_steps_completed": result["steps_completed"], "optimizer_best_step": result["best_step"],
        "baseline": baseline, "candidate_log_relative_path": "OUT." + suffix + "/running_scf.log",
        "energy_quantity": "PBE_total_energy_not_RPA_E0", "tolerance_ev_per_c": TOLERANCE_EV_PER_C,
        "scheduler_gate": "pending_external_validation", "physical_release_gate": "hold",
    }
    output.mkdir(parents=True)
    (output / ".provenance").mkdir()
    for name, content in prepared.items():
        (output / name).write_bytes(content)
    for name, content in evidence.items():
        (output / ".provenance" / name).write_bytes(content)
    _write_json(output / PREPARATION, manifest)
    return manifest


def _within(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe preparation path")
    path = (root / path).resolve()
    if root not in path.parents:
        raise ValueError("preparation path escapes endpoint")
    return path


def collect_pbe_endpoint(*, prepared_dir, preparation_sha256):
    """Collect SCF log evidence only; scheduler/source/binary checks are external."""
    root = Path(prepared_dir).resolve()
    manifest = json.loads(_read_hashed(root / PREPARATION, preparation_sha256))
    if (manifest.get("status") != "prepared" or manifest.get("scope") != SCOPE
            or manifest.get("atoms_per_cell") != 2):
        raise ValueError("invalid solid-only preparation manifest")
    for group in ("prepared_input_sha256", "evidence_sha256"):
        for name, expected in manifest[group].items():
            _read_hashed(_within(root, name), expected)
    baseline = _scf_log(_read_hashed(
        root / ".provenance/original_running_scf.log", manifest["baseline"]["log_sha256"]
    ))
    candidate_path = _within(root, manifest["candidate_log_relative_path"])
    candidate = _scf_log(candidate_path.read_bytes())
    counts, reference_counts = candidate["band_counts"], baseline["band_counts"]
    for key, value in counts.items():
        if value is not None and reference_counts[key] is not None and value != reference_counts[key]:
            raise ValueError("candidate and baseline " + key + " band counts differ")
    delta = (candidate["energy_ev"] - baseline["energy_ev"]) / 2
    result = {
        "status": "collected", "scope": SCOPE, "energy_quantity": "PBE_total_energy_not_RPA_E0",
        "baseline_energy_ev": baseline["energy_ev"], "candidate_energy_ev": candidate["energy_ev"],
        "energy_delta_ev_per_c": delta, "tolerance_ev_per_c": TOLERANCE_EV_PER_C,
        "comparison": "absolute_candidate_minus_original_per_C_lte_tolerance",
        "pbe_total_energy_gate": "pass" if abs(delta) <= TOLERANCE_EV_PER_C + 1e-12 else "fail",
        "scf_log_gate": "pass", "band_counts": counts,
        "band_count_check": "pass" if all(value is not None for value in counts.values()) else "unavailable",
        "candidate_log_path": str(candidate_path), "candidate_log_sha256": candidate["log_sha256"],
        "baseline_log_sha256": baseline["log_sha256"], "preparation_sha256": preparation_sha256,
        "optimizer_result_sha256": manifest["optimizer_result_sha256"],
        "candidate_orbital_sha256": manifest["candidate_orbital_sha256"],
        "scheduler_gate": "pending_external_validation", "physical_release_gate": "hold",
    }
    _write_json(root / COLLECTION, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("optimizer-result", "optimizer-result-sha256", "original-q1",
                 "baseline-log-sha256", "output"):
        prepare.add_argument("--" + name, required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--prepared-dir", required=True)
    collect.add_argument("--preparation-sha256", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "prepare":
        manifest = prepare_pbe_endpoint(**args)
        result = {"preparation": manifest,
                  "preparation_sha256": _sha256((Path(args["output"]) / PREPARATION).read_bytes())}
    else:
        result = collect_pbe_endpoint(**args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if command == "collect" and result["pbe_total_energy_gate"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
