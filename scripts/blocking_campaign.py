"""Freeze the SAT blocking scenario on the exact reference corpus."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import compact_bfs_campaign as common

ROOT = Path(__file__).resolve().parents[1]
VARIANT = "T20"
UNIT = "dfa-paper-blocking-20261014.service"


def make_plan(reference):
    baseline = common.make_plan(reference)["variants"][0]
    configuration = {**baseline["configuration"], "strictness": False}
    options = reference["immutable"]["options"]
    execution = {k: options[k] for k in ("state_counts", "workers", "timeout_seconds", "repetitions",
                                         "memory_gib_per_test", "minimum_available_ram_gib")}
    execution.update(variant_sequence=[VARIANT], case_selection=[
        {k: case[k] for k in ("input_kind", "case_id")} for case in reference["immutable"]["corpus"]])
    return {"status": "SAT blocking supplement",
            "reference_campaign_id": reference["campaign_id"], "reference_variant": "T01",
            "execution": execution, "variants": [{"id": VARIANT, "name": "SAT / external strictness blocking",
                "configuration": configuration, "maxsat_algorithm": None, "rc2": None}],
            "scope": "Finite strictness still required; invalid ordered factor pairs checked externally and blocked; nonemptiness stays encoded"}


def adapt_harness_source(source):
    source = common.adapt_harness_source(source)
    replacements = [
        ('if c["factor_sizes"] is not None or c["strictness"] is not True:',
         'if c["factor_sizes"] is not None or not isinstance(c["strictness"], bool):'),
        ('"this campaign requires automatic sizes and encoded strictness"',
         '"this campaign requires automatic sizes and a Boolean strictness setting"'),
        ('argv = [str(output / job["case"]["input"]), "--strictness", "--optimization", c["optimization"],',
         'argv = [str(output / job["case"]["input"]), "--strictness" if c["strictness"] else "--no-strictness", "--optimization", c["optimization"],'),
        ('"strictness_strategy": "witness",',
         '"strictness_strategy": "witness" if c["strictness"] else "blocking",'),
    ]
    for old, new in replacements:
        if new in source:
            continue
        if source.count(old) != 1:
            raise ValueError(f"unexpected harness; cannot safely update blocking support: {old}")
        source = source.replace(old, new)
    return source


def prepare(reference_dir, output):
    if sys.platform != "linux":
        raise ValueError("prepare the blocking campaign on the Linux server")
    reference_dir, output = reference_dir.resolve(), output.resolve()
    if output == reference_dir or output.is_relative_to(reference_dir):
        raise ValueError("blocking outputs must be separate from the reference")
    reference = common.checked_manifest(reference_dir)
    plan = make_plan(reference)
    cases = common.frozen_cases(reference_dir, reference)
    harness_path = ROOT / "remote/run_paper.py"
    source = harness_path.read_text(encoding="utf-8")
    adapted = adapt_harness_source(source)
    if source != adapted:
        harness_path.write_text(adapted, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("blocking_campaign_harness", harness_path)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    h.selected_cases = lambda options: cases
    output.mkdir(parents=True, exist_ok=True)
    manifest = h.prepare(output, plan)
    common.assert_same_corpus(reference, manifest)
    if reference["immutable"]["runtime"] != manifest["immutable"]["runtime"]:
        raise ValueError("installed Python/solver runtime differs from the reference; do not launch")
    resources = common.load(reference_dir / "runtime-resources.json")
    if resources["campaign_id"] != reference["campaign_id"]:
        raise ValueError("reference resource provenance mismatch")
    resources.update(campaign_id=manifest["campaign_id"], unit=UNIT,
                     reference_recorded_at_utc=resources.get("recorded_at_utc"),
                     recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                     reason="SAT blocking campaign with reference resource controls")
    common.write_json(output / "runtime-resources.json", resources)
    proof = {"reference_campaign_id": reference["campaign_id"], "reference_variant": "T01",
             "campaign_id": manifest["campaign_id"], "variant": VARIANT,
             "configuration": plan["variants"][0]["configuration"], "identical_input_hashes": len(cases),
             "identical_runtime": True, "total_jobs": manifest["immutable"]["total_jobs"],
             "resources": resources, "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "limitations": "One repetition; noncontemporaneous T01 and T20 campaigns"}
    common.write_json(output / "reference-comparison.json", proof)
    print(json.dumps(proof, indent=2))
    return proof


def check_results(reference_dir, output):
    """Validate finished blocking optima against available solved T01 cases."""
    reference, manifest = common.checked_manifest(reference_dir), common.checked_manifest(output)
    common.assert_same_corpus(reference, manifest)
    expected = make_plan(reference)["variants"]
    if manifest["immutable"]["options"]["variants"] != expected:
        raise ValueError("blocking scenario differs from its controlled protocol")
    compared = complete = 0
    statuses = {}
    baseline = next(v for v in reference["immutable"]["options"]["variants"] if v["id"] == "T01")
    reference_cases = {(c["input_kind"], c["case_id"]): c for c in reference["immutable"]["corpus"]}
    for case in manifest["immutable"]["corpus"]:
        for repetition in range(1, manifest["immutable"]["options"]["repetitions"] + 1):
            tail = Path(f"repeat_{repetition:02d}") / case["input_kind"] / case["case_id"] / "result.json"
            path = output / "runs" / f"states_{case['states']:02d}" / VARIANT / tail
            if not path.exists():
                continue
            value = common.load(path)
            if value["paper"]["campaign_id"] != manifest["campaign_id"] or value["paper"]["job"] != {
                    "variant": expected[0], "case": case, "repetition": repetition}:
                raise ValueError("blocking result provenance mismatch")
            if value["status"] not in common.TERMINAL:
                raise ValueError("blocking campaign has an unfinished/error saved result")
            complete += 1
            statuses[value["status"]] = statuses.get(value["status"], 0) + 1
            if value["status"] not in ("sat", "unsat"):
                continue
            for key, desired in (("strictness_strategy", "blocking"), ("optimization_backend", "sat"),
                ("sat_solver", "g3"), ("factor_order", "max"), ("strictness_mode", "finite"),
                ("product_encoding", "factored"), ("bfs_encoding", "classic"), ("bfs_parent_redundancy", True)):
                if value.get(key) != desired:
                    raise ValueError(f"blocking result configuration mismatch: {key}")
            original = reference_dir / "runs" / f"states_{case['states']:02d}" / "T01" / tail
            prior = common.load(original)
            if prior["paper"]["campaign_id"] != reference["campaign_id"] or prior["paper"]["job"] != {
                    "case": reference_cases[(case["input_kind"], case["case_id"])],
                    "variant": baseline, "repetition": repetition}:
                raise ValueError("baseline result provenance mismatch")
            if prior["status"] in ("sat", "unsat"):
                for key in ("status", "edit_distance", "factor_sizes"):
                    if value.get(key) != prior.get(key):
                        raise ValueError(f"blocking/baseline disagreement: {case['case_id']} {key}")
                compared += 1
    result = {"campaign_id": manifest["campaign_id"], "finalized": complete,
              "total": manifest["immutable"]["total_jobs"], "statuses": statuses,
              "matched_solved_T01_checks": compared, "disagreements": 0,
              "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    common.write_json(output / "baseline-check.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.reference, args.output)
    else:
        check_results(args.reference, args.output)


if __name__ == "__main__":
    main()
