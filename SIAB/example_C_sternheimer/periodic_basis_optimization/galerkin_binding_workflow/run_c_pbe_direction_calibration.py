"""One sequential eight-probe PBE batch, without response or optimizer calls."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_c_pbe_direction_calibration import prepare_calibration
from check_c_optimized_pbe import _read_hashed, _sha256, _within, _write_json
from check_c_direction_probe_pbe import prepare_probe_pbe, collect_probe_pbe, PREPARATION, COLLECTION
from periodic_galerkin_direction_calibration import analyze_pbe_axes, SIGNED_RADII

ROOT = Path("/work1/ghj/c-solid-fd8-q13-standard-20260903")
CENTER = ROOT/"stage-c-all-radial-backoff-quarter-e3beab29/result/RESULT.json"
CENTER_SHA = "720fdfeb26617020bd834a241b2393c30b452aa0e2ac7d64c4975beff92337c1"
PBE = ROOT/"stage-c-all-radial-quarter-pbe-e3beab29/pbe"
PBE_PREP_SHA = "ec2b35949cdb079d726eb1e6a10cc70b8d1bd345f4350d436c3b4a180cc40e6b"
PBE_COLLECTION_SHA = "72b34192895333b058d000cec18989fd343f9935e72126ac4f178b29ca365d17"
DIAGNOSTIC = ROOT/"stage-c-all-radial-gradient-diagnostic-7864f4ff/result/RADIAL_GRADIENT_DIAGNOSTIC.json"
DIAGNOSTIC_SHA = "42ba257d9f3bfc7293954b1f4aad644a19c55cbeed8eaa3b6404cb47368d7b53"
ABACUS = ROOT/"artifacts/abacus-711af860c-intel2021/abacus_3p"
ABACUS_SHA = "ab8f1192ad7f7246218422c80218c869d8053fa0d44f6ad728b9323eb4ee5ab3"
PMI = Path("/opt/gridview/slurm/lib/libpmi2.so")
PMI_SHA = "094c1179452217a4d4a5985cb4642440c824d227784167d0017e7a953d87e5cf"


def run_prepared_probes(*, prepared_root, preparation_sha256, center_dir,
                       center_preparation_sha256, center_collection_sha256, launcher):
    root = Path(prepared_root).resolve()
    prepared = json.loads(_read_hashed(root/"CALIBRATION_PREPARED.json", preparation_sha256))
    rows = prepared["probes"]
    expected = [(name, r) for name in ("T", "N") for r in SIGNED_RADII]
    if (prepared.get("status") != "success" or len(rows) != 8
            or [(r["direction_name"], r["signed_radius"]) for r in rows] != expected
            or any(r["status"] != "prepared" for r in rows)):
        raise ValueError("all eight unique probes must pass cheap screens before any SCF")
    bindings = ("center_result_sha256", "center_coefficient_sha256", "center_orbital_sha256")
    directions = json.loads(_read_hashed(root/"probes/DIRECTIONS.json", prepared["directions_sha256"]))
    if (any(not prepared.get(key) or directions.get(key) != prepared[key] for key in bindings)
            or prepared.get("pbe_collection_sha256") != center_collection_sha256):
        raise ValueError("directions must bind the same center and PBE collection")
    probes = []
    for row in rows:
        probe = _within(root/"probes", row["path"])
        content = json.loads(_read_hashed(probe, row["sha256"]))
        if (any(content.get(key) != prepared[key] for key in bindings)
                or content.get("directions_sha256") != prepared["directions_sha256"]
                or content.get("direction_name") != row["direction_name"]
                or content.get("signed_radius") != row["signed_radius"]
                or probe in probes):
            raise ValueError("probe identity must match its direction, center and manifest row")
        probes.append(probe)
    center = json.loads(_read_hashed(Path(center_dir)/"PBE_ENDPOINT_COLLECTION.json", center_collection_sha256))
    (root/"PBE_EXECUTION_LOCK").mkdir()
    staged = []
    # Validate and stage every input before spending any SCF work.
    for row, probe in zip(rows, probes):
        destination = root/"pbe"/probe.parent.name
        prepare_probe_pbe(center_dir=center_dir, center_preparation_sha256=center_preparation_sha256,
            center_collection_sha256=center_collection_sha256, probe_path=probe,
            probe_sha256=row["sha256"], output=destination)
        prep_sha = _sha256((destination/PREPARATION).read_bytes())
        staged.append((row, probe, destination, prep_sha))
    samples, records = [], []
    for row, probe, destination, prep_sha in staged:
        started = time.perf_counter()
        with (destination/"abacus.out").open("xb") as stream:
            subprocess.run(launcher, cwd=destination, stdout=stream, stderr=subprocess.STDOUT, check=True)
        collected = collect_probe_pbe(prepared_dir=destination, preparation_sha256=prep_sha)
        samples.append(collected)
        records.append(dict(direction_name=row["direction_name"], signed_radius=row["signed_radius"],
            probe_sha256=row["sha256"], preparation_sha256=prep_sha,
            collection_path=str(destination/COLLECTION), collection_sha256=_sha256((destination/COLLECTION).read_bytes()),
            seconds=time.perf_counter()-started, exit_code=0))
        _write_json(probe.parent/"EXECUTION.json", records[-1])
        print(json.dumps(dict(completed=len(samples), sample=collected), allow_nan=False), flush=True)
    result = analyze_pbe_axes(center["candidate_energy_ev"], samples)
    result.update(samples=samples, executions=records, preparation_sha256=preparation_sha256,
        center_collection_sha256=center_collection_sha256, actual_scf_count=len(samples),
        source_commit=prepared.get("source_commit"), scheduler_gate="pending_external_validation")
    _write_json(root/"PBE_DIRECTION_CALIBRATION.json", result)
    return result


def verify_source(stage):
    expected = os.environ["C_CALIBRATION_DEPLOYMENT_SHA256"]
    manifest = json.loads(_read_hashed(stage/"DEPLOYMENT.json", expected))
    source = Path(manifest["source_directory"]).resolve()
    if source != Path(__file__).resolve().parents[4]:
        raise ValueError("running source does not match immutable deployment")
    for name, digest in manifest["files"].items():
        _read_hashed(_within(source, name), digest)
    _read_hashed(Path(manifest["archive_path"]), manifest["archive_sha256"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    stage = Path(args.stage).resolve()
    source = verify_source(stage)
    _read_hashed(ABACUS, ABACUS_SHA)
    _read_hashed(PMI, PMI_SHA)
    if (os.environ.get("SLURM_JOB_NUM_NODES") != "1" or os.environ.get("OMP_NUM_THREADS") != "7"
            or os.environ.get("I_MPI_PMI_LIBRARY") != str(PMI)):
        raise ValueError("frozen single-node PMI2/OMP layout required")
    _write_json(stage/"EXECUTION_PROVENANCE.json", dict(status="launching", server="df_dcu",
        scheduler_job=os.environ["SLURM_JOB_ID"], source_commit=source["source_commit"],
        source_archive_sha256=source["archive_sha256"], abacus_sha256=ABACUS_SHA, pmi_sha256=PMI_SHA,
        profile="frozen-C-reference-711af860c", version_verdict="feature-branch-exception",
        exception_reason="same immutable ABACUS and pure PBE inputs as accepted original and quarter",
        mpi_ranks=4, omp_threads=7, delta_st="not_run", librpa="not_run", rpa="not_run",
        physical_release_gate="hold"))
    prepare_calibration(center_result=CENTER, center_result_sha256=CENTER_SHA,
        pbe_collection=PBE/"PBE_ENDPOINT_COLLECTION.json", pbe_collection_sha256=PBE_COLLECTION_SHA,
        diagnostic=DIAGNOSTIC, diagnostic_sha256=DIAGNOSTIC_SHA,
        output=stage/"result", source_commit=source["source_commit"])
    result = run_prepared_probes(prepared_root=stage/"result",
        preparation_sha256=_sha256((stage/"result/CALIBRATION_PREPARED.json").read_bytes()),
        center_dir=PBE, center_preparation_sha256=PBE_PREP_SHA, center_collection_sha256=PBE_COLLECTION_SHA,
        launcher=["srun", "--mpi=pmi2", "--cpu-bind=none", "-n", "4", str(ABACUS)])
    verify_source(stage)
    _read_hashed(ABACUS, ABACUS_SHA)
    _read_hashed(PMI, PMI_SHA)
    _write_json(stage/"PROVENANCE.json", dict(status="success", source_commit=source["source_commit"],
        result_sha256=_sha256((stage/"result/PBE_DIRECTION_CALIBRATION.json").read_bytes()),
        consistency_gate=result["consistency_gate"], actual_scf_count=8, physical_release_gate="hold"))


if __name__ == "__main__":
    main()
