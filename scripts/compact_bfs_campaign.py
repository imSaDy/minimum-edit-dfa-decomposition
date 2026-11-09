"""Prepare an exact-corpus supplement and compare the four BFS scenarios.

Server preparation uses the existing Linux campaign harness (remote/run_paper.py). No old manifest, snapshot, or result is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("T01", "T08", "T18", "T19")
LABELS = {"T01": "Classic BFS + converse", "T08": "Classic BFS - converse",
          "T18": "Compact BFS + converse", "T19": "Compact BFS - converse"}
TERMINAL = {"sat", "unsat", "timeout", "memory_limit"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def checked_manifest(directory):
    manifest = load(Path(directory) / "campaign.json")
    if manifest["campaign_id"] != digest(manifest["immutable"]):
        raise ValueError("campaign manifest digest mismatch")
    return manifest


def make_plan(reference, include_classic=False):
    options = reference["immutable"]["options"]
    by_id = {v["id"]: v for v in options["variants"]}
    baseline = {"optimization": "sat", "sat_solver": "g3", "factor_sizes": None,
                "factor_order": "max", "strictness": True, "strictness_mode": "finite",
                "product_encoding": "factored", "bfs_parent_redundancy": True}
    for variant, redundancy in (("T01", True), ("T08", False)):
        actual = dict(by_id[variant]["configuration"])
        actual.setdefault("bfs_encoding", "classic")
        if actual != {**baseline, "bfs_parent_redundancy": redundancy, "bfs_encoding": "classic"}:
            raise ValueError(f"reference {variant} is not the intended matching baseline")
    variants = []
    for variant, kind, redundancy in (("T01", "classic", True), ("T08", "classic", False),
                                       ("T18", "compact", True), ("T19", "compact", False)):
        variants.append({"id": variant, "name": LABELS[variant], "configuration": {
            **baseline, "bfs_encoding": kind, "bfs_parent_redundancy": redundancy},
            "maxsat_algorithm": None, "rc2": None})
    execution = {k: options[k] for k in ("state_counts", "workers", "timeout_seconds", "repetitions",
                                         "memory_gib_per_test", "minimum_available_ram_gib")}
    execution.update(variant_sequence=list(VARIANTS if include_classic else VARIANTS[2:]),
                     case_selection=[{k: c[k] for k in ("input_kind", "case_id")}
                                     for c in reference["immutable"]["corpus"]])
    return {"status": "Supplemental compact BFS comparison; empirical results pending",
            "reference_campaign_id": reference["campaign_id"], "defaults": baseline,
            "variants": variants, "execution": execution,
            "scope": "Section 3.2 compact predicates; complete prefixes; no Section 3.3 constraints"}


def adapt_harness_source(source):
    """Add the selector and result guard to the existing v1 worker harness.

    This function is also the portable, tested server adapter for a checkout
    where the ignored remote infrastructure has not received this update.
    """
    if '"--bfs-encoding", c.get("bfs_encoding", "classic")' not in source:
        old = '"--factor-order", c["factor_order"], "--sat-solver", c["sat_solver"],'
        if source.count(old) != 1:
            raise ValueError("unexpected worker harness: cannot insert BFS selector")
        source = source.replace(old, old + '\n                "--bfs-encoding", c.get("bfs_encoding", "classic"),')
    if '"bfs_encoding": c.get("bfs_encoding", "classic")' not in source:
        old = '"product_encoding": c["product_encoding"],'
        if source.count(old) != 1:
            raise ValueError("unexpected worker harness: cannot insert BFS result guard")
        source = source.replace(old, old + '\n                        "bfs_encoding": c.get("bfs_encoding", "classic"),')
    return source


def frozen_cases(reference_dir, reference):
    cases = []
    seen = set()
    for case in reference["immutable"]["corpus"]:
        key = (case["input_kind"], case["case_id"])
        if key in seen:
            raise ValueError("duplicate corpus identity")
        seen.add(key)
        path = (reference_dir / case["input"]).resolve()
        if not path.is_relative_to(reference_dir.resolve()):
            raise ValueError("input escapes the reference campaign")
        if sha(path) != case["input_dot_sha256"]:
            raise ValueError(f"reference input changed: {key}")
        cases.append(SimpleNamespace(state_count=case["states"], kind=case["input_kind"],
                                    case_id=case["case_id"], benchmark=SimpleNamespace(
            input_path=path, dot_sha256=case["input_dot_sha256"], source_size=case.get("source_size"),
            instance_id=case.get("instance_id"), metadata=case.get("metadata"))))
    return cases


def prepare(reference_dir, output, include_classic=False):
    if sys.platform != "linux":
        raise ValueError("prepare on the Linux server with its installed campaign harness")
    reference_dir, output = reference_dir.resolve(), output.resolve()
    if output == reference_dir or output.is_relative_to(reference_dir):
        raise ValueError("the supplement must be separate from the reference campaign")
    reference = checked_manifest(reference_dir)
    plan = make_plan(reference, include_classic)
    cases = frozen_cases(reference_dir, reference)
    harness_path = ROOT / "remote/run_paper.py"
    source = harness_path.read_text(encoding="utf-8")
    adapted = adapt_harness_source(source)
    if source != adapted:
        harness_path.write_text(adapted, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("compact_campaign_harness", harness_path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.selected_cases = lambda options: cases
    output.mkdir(parents=True, exist_ok=True)
    manifest = harness.prepare(output, plan)
    assert_same_corpus(reference, manifest)
    runtime_matches = reference["immutable"]["runtime"] == manifest["immutable"]["runtime"]
    if not runtime_matches:
        raise ValueError("server Python/solver runtime differs from the reference; do not launch")
    # Retain the latest completed campaign's scheduling policy and kernel
    # limits, without editing either immutable manifest.
    resources_path = reference_dir / "runtime-resources.json"
    if resources_path.exists():
        resources = load(resources_path)
        if resources["campaign_id"] != reference["campaign_id"]:
            raise ValueError("reference resource override has incorrect provenance")
        resources.update(campaign_id=manifest["campaign_id"],
                         reason="Compact BFS supplement using reference runtime resource limits",
                         unit="dfa-paper-compact-bfs-20261012.service")
        write_json(output / "runtime-resources.json", resources)
    proof = {"reference_campaign_id": reference["campaign_id"], "campaign_id": manifest["campaign_id"],
             "identical_input_hashes": len(cases), "identical_runtime": runtime_matches,
             "selected_variants": plan["execution"]["variant_sequence"],
             "total_jobs": manifest["immutable"]["total_jobs"],
             "reference_directory": str(reference_dir), "new_directory": str(output),
             "timing_limitation": "Old classic and new compact measurements are not contemporaneous; one repetition"}
    write_json(output / "reference-comparison.json", proof)
    print(json.dumps(proof, indent=2))
    return proof


def corpus_map(manifest):
    result = {}
    for case in manifest["immutable"]["corpus"]:
        key = (case["input_kind"], case["case_id"])
        if key in result:
            raise ValueError("duplicate input identity")
        result[key] = (case["states"], case["input_dot_sha256"])
    return result


def assert_same_corpus(reference, supplement):
    if corpus_map(reference) != corpus_map(supplement):
        raise ValueError("the two campaigns do not contain identical input identities and hashes")
    for key in ("state_counts", "timeout_seconds", "repetitions", "memory_gib_per_test"):
        if reference["immutable"]["options"][key] != supplement["immutable"]["options"][key]:
            raise ValueError(f"comparison protocol differs: {key}")


def read_results(directory, manifest, selected):
    rows = {}
    for variant in manifest["immutable"]["options"]["variants"]:
        if variant["id"] not in selected:
            continue
        for case in manifest["immutable"]["corpus"]:
            for repetition in range(1, manifest["immutable"]["options"]["repetitions"] + 1):
                path = directory / "runs" / f"states_{case['states']:02d}" / variant["id"] / (
                    f"repeat_{repetition:02d}") / case["input_kind"] / case["case_id"] / "result.json"
                if not path.exists():
                    continue
                value = load(path)
                job = {"case": case, "variant": variant, "repetition": repetition}
                if value.get("paper", {}).get("campaign_id") != manifest["campaign_id"] or value["paper"].get("job") != job:
                    raise ValueError(f"result provenance mismatch: {path}")
                config = variant["configuration"]
                if value["status"] in ("sat", "unsat"):
                    for field, expected in (("optimization_backend", "sat"), ("factor_order", "max"),
                                             ("sat_solver", "g3"), ("strictness_mode", "finite"),
                                             ("product_encoding", "factored"),
                                             ("bfs_encoding", config.get("bfs_encoding", "classic")),
                                             ("bfs_parent_redundancy", config["bfs_parent_redundancy"])):
                        actual = value.get(field, "classic" if field == "bfs_encoding" else None)
                        if actual != expected:
                            raise ValueError(f"result configuration mismatch: {field}: {path}")
                rows[(case["input_kind"], case["case_id"], repetition, variant["id"])] = value
    return rows


def summarize(reference, rows):
    summaries, detailed = [], []
    for state in reference["immutable"]["options"]["state_counts"]:
        keys = [(c["input_kind"], c["case_id"], r) for c in reference["immutable"]["corpus"]
                if c["states"] == state
                for r in range(1, reference["immutable"]["options"]["repetitions"] + 1)]
        joint_sat, joint_terminal = [], []
        for key in keys:
            values = [rows.get((*key, variant)) for variant in VARIANTS]
            solved = [v for v in values if v and v["status"] in ("sat", "unsat")]
            if len({(v["status"], v.get("edit_distance"), tuple(v.get("factor_sizes") or [])) for v in solved}) > 1:
                raise ValueError(f"semantic/optimum disagreement on {key}")
            if all(v and v["status"] == "sat" for v in values):
                joint_sat.append(key)
            if all(v and v["status"] in TERMINAL for v in values):
                joint_terminal.append(key)
            for variant, value in zip(VARIANTS, values):
                paper = value.get("paper", {}) if value else {}
                detailed.append({"states": state, "input_kind": key[0], "case_id": key[1],
                    "repetition": key[2], "variant": variant,
                    "status": value["status"] if value else "pending",
                    "edit_distance": value.get("edit_distance") if value else None,
                    "factor_sizes": value.get("factor_sizes") if value else None,
                    "wall_seconds": paper.get("wall_seconds"),
                    "kernel_peak_rss_bytes": paper.get("kernel_peak_rss_bytes")})
        for variant in VARIANTS:
            values = [rows.get((*k, variant)) for k in keys]
            statuses = Counter(v["status"] if v else "pending" for v in values)
            wall = [rows[(*k, variant)]["paper"]["wall_seconds"] for k in joint_sat]
            rss = [rows[(*k, variant)]["paper"]["kernel_peak_rss_bytes"] for k in joint_terminal]
            summaries.append({"states": state, "variant": variant, "label": LABELS[variant],
                "scheduled": len(keys), "sat": statuses["sat"], "unsat": statuses["unsat"],
                "timeout": statuses["timeout"], "memory_limit": statuses["memory_limit"],
                "pending": statuses["pending"], "other": sum(n for s, n in statuses.items() if s not in TERMINAL | {"pending"}),
                "completion_percent": 100 * (statuses["sat"] + statuses["unsat"]) / len(keys),
                "joint_sat_count": len(joint_sat), "mean_wall_seconds": statistics.mean(wall) if wall else None,
                "joint_terminal_count": len(joint_terminal),
                "mean_kernel_peak_rss_gib": statistics.mean(rss) / 1024**3 if rss else None})
    return summaries, detailed


def compare(reference_dir, supplement_dir, output, allow_incomplete=False):
    reference, supplement = checked_manifest(reference_dir), checked_manifest(supplement_dir)
    assert_same_corpus(reference, supplement)
    expected_plan = make_plan(reference)
    actual_variants = supplement["immutable"]["options"]["variants"]
    fresh_four_way = actual_variants == expected_plan["variants"]
    if not fresh_four_way and actual_variants != expected_plan["variants"][2:]:
        raise ValueError("supplement does not contain exactly the two controlled compact variants")
    rows = {} if fresh_four_way else read_results(reference_dir, reference, VARIANTS[:2])
    rows.update(read_results(supplement_dir, supplement, VARIANTS if fresh_four_way else VARIANTS[2:]))
    summaries, detailed = summarize(reference, rows)
    complete = all(r["status"] in TERMINAL for r in detailed)
    if not complete and not allow_incomplete:
        raise ValueError("comparison is incomplete; use --allow-incomplete for a labeled interim report")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (("state-summary.csv", summaries), ("per-test-results.csv", detailed)):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    report = {"complete": complete, "reference_campaign_id": reference["campaign_id"],
              "supplement_campaign_id": supplement["campaign_id"], "summary": summaries,
              "rules": ["Runtime means use only inputs solved SAT by all four scenarios",
                        "Completion uses the full scheduled denominator; UNSAT is a completed proof",
                        "Memory uses jointly finalized inputs, including timeouts and memory limits",
                        "Peak RSS measures the worker process lifetime, not whole-server memory",
                        "One repetition; classic and compact runs occurred at different times",
                        "Semantic status, edit cost and selected size disagreements stop the report"]}
    write_json(output / "comparison.json", report)
    plot(summaries, output, complete)
    print(json.dumps({"complete": complete, "report": str(output)}, indent=2))
    return report


def plot(summary, output, complete):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, ylabel in (("mean_wall_seconds", "Mean wall time (s), jointly solved inputs"),
                           ("completion_percent", "Solved within limit (%)"),
                           ("mean_kernel_peak_rss_gib", "Mean worker peak RSS (GiB), jointly finalized inputs")):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for variant, marker in zip(VARIANTS, ("o", "s", "^", "D")):
            rows = [r for r in summary if r["variant"] == variant]
            ax.plot([r["states"] for r in rows],
                    [r[metric] if r[metric] is not None else float("nan") for r in rows],
                    marker=marker, label=LABELS[variant])
        ax.set(xlabel="Input DFA states", ylabel=ylabel,
               title="Four BFS configurations" + (" (interim)" if not complete else ""))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(output / f"{metric}.{suffix}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--reference", type=Path, required=True, help="Completed reference campaign directory")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--include-classic", action="store_true", help="Fresh contemporaneous four-way run")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--supplement", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.reference, args.output, args.include_classic)
    else:
        compare(args.reference, args.supplement, args.output, args.allow_incomplete)


if __name__ == "__main__":
    main()
