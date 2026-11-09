"""Run every available 3–9-state DFAMiner DFA and rank five per size.

Both minimum and non-minimum DOT collections are scanned. State counts come
from the parsed DFAs, not their source directory names. Results are isolated
from the project's existing five-per-size benchmarks and are resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dfa_identify.dot_io import dfa_from_dot
from dfa_identify.minimum_edit import ENCODING_REVISION, encoding_revision, normalize_optimization
from run_minimal_five_benchmark import BenchmarkCase, SEARCH_ORDERS, _run_one


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINIMAL_ROOT = PROJECT_ROOT / "benchmarks/dfaminer"
NONMINIMAL_ROOT = PROJECT_ROOT / "benchmarks/dfaminer-nonminimal"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmarks/dfaminer-all-small-3-9"
DATASET_NAME = "dfaminer-all-available-minimal-and-nonminimal"
NONEMPTY_REQUIREMENT = "common-accepting-witness-v1"
TERMINAL_STATUSES = {"sat", "unsat", "timeout"}
MINIMAL_NAME = re.compile(r"dfa-(\d+)-(\d+)$")


def output_root_for_order(base: Path, factor_order: str) -> Path:
    """Keep the original max results and isolate the other search orders."""

    return base if factor_order == "max" else base / f"factor-order-{factor_order}"


@dataclass(frozen=True)
class InputCase:
    kind: str
    benchmark: BenchmarkCase

    @property
    def state_count(self) -> int:
        return self.benchmark.input_state_count

    @property
    def case_id(self) -> str:
        return self.benchmark.case_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonminimal_metadata() -> dict[str, dict[str, Any]]:
    manifest = json.loads(
        (NONMINIMAL_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    attempts = json.loads(
        (NONMINIMAL_ROOT / "five_per_output_size_attempts.json").read_text(
            encoding="utf-8"
        )
    )["attempts"]
    records = {record["case_id"]: record for record in manifest["cases"]}
    records.update(
        {case_id: record for case_id, record in attempts.items() if record["status"] == "success"}
    )
    return records


def scan_cases(min_states: int, max_states: int) -> list[InputCase]:
    metadata = nonminimal_metadata()
    cases: list[InputCase] = []
    seen: set[tuple[str, str]] = set()
    roots = (("minimal", MINIMAL_ROOT), ("nonminimal", NONMINIMAL_ROOT))
    for kind, root in roots:
        for path in sorted((root / "dot").rglob("*.dot")):
            state_count = len(tuple(dfa_from_dot(path).states()))
            if not min_states <= state_count <= max_states:
                continue
            case_id = path.stem
            if (kind, case_id) in seen:
                raise ValueError(f"duplicate {kind} case ID: {case_id}")
            seen.add((kind, case_id))
            digest = sha256(path)
            if kind == "minimal":
                match = MINIMAL_NAME.fullmatch(case_id)
                if match is None:
                    raise ValueError(f"unexpected minimal filename: {path}")
                source_size, instance_id = map(int, match.groups())
                sample = root / "samples" / f"states_{source_size:02d}" / f"res{source_size}-{instance_id}.txt"
                sample_path = sample.as_posix() if sample.is_file() else None
                extra = {
                    "input_kind": kind,
                    "nonempty_requirement": NONEMPTY_REQUIREMENT,
                }
            else:
                record = metadata.get(case_id)
                if record is None:
                    raise ValueError(f"non-minimal DOT has no generation record: {path}")
                if record["status"] != "success":
                    raise ValueError(f"non-minimal DOT is not a successful generation: {path}")
                expected_states = record.get("validation", {}).get("actual_state_count")
                if expected_states != state_count:
                    raise ValueError(f"{case_id}: parsed {state_count} states, manifest says {expected_states}")
                expected_hash = record.get("validation", {}).get("dot_sha256")
                if expected_hash and expected_hash != digest:
                    raise ValueError(f"{case_id}: DOT hash differs from generation record")
                source_size = int(record["source_size"])
                instance_id = int(record["instance_id"])
                sample_path = record.get("copied_sample") or record.get("source_sample")
                extra = {
                    "input_kind": kind,
                    "nonempty_requirement": NONEMPTY_REQUIREMENT,
                    "base_case": record["base_case"],
                    "minimum_consistent_states": record["minimum_consistent_states"],
                }
            cases.append(
                InputCase(
                    kind=kind,
                    benchmark=BenchmarkCase(
                        case_id=case_id,
                        input_path=path.resolve(),
                        input_state_count=state_count,
                        source_size=source_size,
                        instance_id=instance_id,
                        dot_sha256=digest,
                        sample=sample_path,
                        metadata=extra,
                    ),
                )
            )
    cases.sort(key=lambda case: (case.state_count, case.kind, case.case_id))
    return cases


def result_path(output_root: Path, case: InputCase) -> Path:
    return (
        output_root
        / "runs"
        / f"states_{case.state_count:02d}"
        / case.kind
        / case.case_id
        / "result.json"
    )


def matching_result(
    path: Path, case: InputCase, timeout_seconds: int, factor_order: str,
    *, require_current_encoding: bool = True, bfs_parent_redundancy: bool = True,
    optimization: str = "sat",
    product_encoding: str = "factored",
) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    benchmark = result.get("benchmark") or {}
    if (
        benchmark.get("dataset") != DATASET_NAME
        or (
            require_current_encoding
            and benchmark.get("encoding_revision") != encoding_revision(optimization, product_encoding)
        )
        or benchmark.get("case_id") != case.case_id
        or benchmark.get("input_kind") != case.kind
        or benchmark.get("nonempty_requirement") != NONEMPTY_REQUIREMENT
        or benchmark.get("input_dot_sha256") != case.benchmark.dot_sha256
        or benchmark.get("input_state_count") != case.state_count
        or benchmark.get("timeout_seconds") != timeout_seconds
        or benchmark.get("factor_size_search_mode") != "all-proper-unordered"
        or benchmark.get("factor_size_search_order") != SEARCH_ORDERS[optimization]
        or benchmark.get("strategy") != "witness"
        or benchmark.get("strictness_mode") != "finite"
        or ("enumeration" if benchmark.get("optimization_backend") == "hybrid"
            else benchmark.get("optimization_backend")) != optimization
        or benchmark.get("symmetry_breaking") != "bfs"
        or benchmark.get("factor_order") != factor_order
        or benchmark.get("bfs_parent_redundancy", benchmark.get("bfs_parent_strengthening", benchmark.get("bfs_converse", True))) is not bfs_parent_redundancy
        or benchmark.get("product_encoding", "factored") != product_encoding
    ):
        return None
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def archive_saved_file(path: Path, directory: Path) -> Path:
    """Keep an exact copy of a saved attempt before a timeout retry."""

    digest = sha256(path)
    destination = directory / f"{path.stem}-{digest[:16]}{path.suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError(f"Existing attempt archive differs: {destination}")
    else:
        shutil.copy2(path, destination)
        if sha256(destination) != digest:
            raise ValueError(f"Attempt archive verification failed: {destination}")
    return destination


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def ranking_key(row: dict[str, Any], rank_by: str) -> tuple[Any, ...]:
    cost = int(row["edit_distance"])
    seconds = float(row["wall_seconds"])
    if rank_by == "min-cost":
        priority = (cost, seconds)
    elif rank_by == "max-cost":
        priority = (-cost, seconds)
    else:
        priority = (seconds, cost)
    return (*priority, row["input_kind"], row["case_id"])


def write_reports(
    cases: Sequence[InputCase],
    results: dict[tuple[str, str], dict[str, Any]],
    output_root: Path,
    timeout_seconds: int,
    rank_by: str,
    min_states: int,
    max_states: int,
) -> None:
    select_five = rank_by != "none"
    detailed: list[dict[str, Any]] = []
    by_state: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        result = results.get((case.kind, case.case_id))
        benchmark = (result or {}).get("benchmark") or {}
        visualizations = (result or {}).get("visualizations") or {}
        row = {
            "states": case.state_count,
            "input_kind": case.kind,
            "case_id": case.case_id,
            "status": (result or {}).get("status", "not_run"),
            "optimization_backend": benchmark.get("optimization_backend", "") if result else "",
            "bfs_parent_redundancy": benchmark.get("bfs_parent_redundancy", benchmark.get("bfs_parent_strengthening", benchmark.get("bfs_converse", True))) if result else "",
            "product_encoding": benchmark.get("product_encoding", "factored") if result else "",
            "encoding_variables": result.get("encoding_statistics", {}).get("variables", "") if result else "",
            "encoding_clauses": result.get("encoding_statistics", {}).get("clauses", "") if result else "",
            "encoding_literals": result.get("encoding_statistics", {}).get("literals", "") if result else "",
            "encoding_revision": (
                benchmark.get("encoding_revision", "legacy-unversioned")
                if result else ""
            ),
            "edit_distance": (result or {}).get("edit_distance", ""),
            "wall_seconds": benchmark.get("wall_seconds", ""),
            "solve_seconds": ((result or {}).get("timing") or {}).get("solve_seconds", ""),
            "factor_sizes": "x".join(map(str, (result or {}).get("factor_sizes") or [])),
            "common_accepting_witness": (
                json.dumps(result["common_accepting_witness"], ensure_ascii=False)
                if result and result.get("status") == "sat"
                else ""
            ),
            "strictness_witnesses": (
                json.dumps(result.get("strictness_witnesses"), ensure_ascii=False)
                if result and result.get("status") == "sat"
                else ""
            ),
            "A_prime_image": visualizations.get("A_prime", {}).get("image", ""),
            "A1_image": visualizations.get("A1", {}).get("image", ""),
            "A2_image": visualizations.get("A2", {}).get("image", ""),
            "input_path": case.benchmark.input_path.as_posix(),
            "result_path": result_path(output_root, case).as_posix(),
            "solver_log_path": benchmark.get("solver_log", ""),
        }
        detailed.append(row)
        by_state[case.state_count].append(row)

    selected: list[dict[str, Any]] = []
    all_solved = (
        sorted(
            (row for row in detailed if row["status"] == "sat"),
            key=lambda row: ranking_key(row, rank_by),
        )
        if select_five else []
    )
    selected_overall = [
        {"rank": position, **row}
        for position, row in enumerate(all_solved[:5], start=1)
    ]
    summary: list[dict[str, Any]] = []
    cost_distribution: list[dict[str, Any]] = []
    for state in range(min_states, max_states + 1):
        group = by_state.get(state, [])
        solved = (
            sorted(
                (row for row in group if row["status"] == "sat"),
                key=lambda row: ranking_key(row, rank_by),
            )
            if select_five else []
        )
        for position, row in enumerate(solved[:5], start=1):
            selected.append({"rank": position, **row})
        status_counts = Counter(row["status"] for row in group)
        kind_counts = Counter(row["input_kind"] for row in group)
        state_costs = Counter(
            int(row["edit_distance"]) for row in group if row["status"] == "sat"
        )
        for kind in ("minimal", "nonminimal"):
            if not kind_counts[kind]:
                continue
            costs = Counter(
                int(row["edit_distance"])
                for row in group
                if row["input_kind"] == kind and row["status"] == "sat"
            )
            for cost in range(max((2, *costs)) + 1):
                cost_distribution.append(
                    {
                        "states": state,
                        "input_kind": kind,
                        "edit_distance": cost,
                        "count": costs[cost],
                    }
                )
        summary.append(
            {
                "states": state,
                "total": len(group),
                "minimal": kind_counts["minimal"],
                "nonminimal": kind_counts["nonminimal"],
                "sat": status_counts["sat"],
                "unsat": status_counts["unsat"],
                "timeout": status_counts["timeout"],
                "error": status_counts["error"],
                "not_run": status_counts["not_run"],
                "cost_0": state_costs[0],
                "cost_1": state_costs[1],
                "cost_2": state_costs[2],
                "cost_gt_2": sum(
                    count for cost, count in state_costs.items() if cost > 2
                ),
                "selected": min(5, len(solved)),
            }
        )

    report_root = output_root / "reports"
    detail_fields = list(detailed[0]) if detailed else [
        "states", "input_kind", "case_id", "status", "optimization_backend", "bfs_parent_redundancy", "product_encoding", "encoding_variables", "encoding_clauses", "encoding_literals", "encoding_revision", "edit_distance",
        "wall_seconds", "solve_seconds", "factor_sizes", "common_accepting_witness",
        "strictness_witnesses", "A_prime_image", "A1_image", "A2_image",
        "input_path", "result_path", "solver_log_path",
    ]
    write_csv(report_root / "all_results.csv", detail_fields, detailed)
    write_csv(
        report_root / "cost_two_results.csv",
        detail_fields,
        [row for row in detailed if row["status"] == "sat" and row["edit_distance"] == 2],
    )
    write_csv(report_root / "top_five_per_state.csv", ["rank", *detail_fields], selected)
    write_csv(report_root / "top_five_overall.csv", ["rank", *detail_fields], selected_overall)
    write_csv(report_root / "summary.csv", list(summary[0]), summary)
    write_csv(
        report_root / "cost_distribution.csv",
        ["states", "input_kind", "edit_distance", "count"],
        cost_distribution,
    )
    write_json(
        report_root / "selection.json",
        {
            "rank_by": rank_by,
            "ranking_rule": {
                "min-cost": "minimum edit distance, then minimum wall time",
                "max-cost": "maximum edit distance, then minimum wall time",
                "fastest": "minimum wall time, then minimum edit distance",
            }.get(rank_by),
            "only_sat_results_are_ranked": select_five,
            "timeout_seconds": timeout_seconds,
            "all_cases_attempted": all(item["not_run"] == 0 and item["error"] == 0 for item in summary),
            "ranking_fully_resolved": select_five and all(
                item["not_run"] == 0
                and item["error"] == 0
                and item["timeout"] == 0
                and item["selected"] == 5
                for item in summary
            ),
            "summary": summary,
            "selected_per_state": selected,
            "selected_overall": selected_overall,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-states", type=int, default=3)
    parser.add_argument("--max-states", type=int, default=9)
    parser.add_argument("--timeout-seconds", type=int, default=1500)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--factor-order", choices=("max", "sum", "product"), default="max")
    parser.add_argument(
        "--optimization", type=normalize_optimization, choices=("sat", "maxsat", "enumeration"), default="sat",
        help="optimization backend (default: sat); MaxSAT and enumeration results use separate subtrees",
    )
    parser.add_argument(
        "--rank-by", choices=("none", "min-cost", "max-cost", "fastest"),
        default="none", help="optional top-five selection; default: save data only",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-new-runs", type=int, default=0, help="0 means all pending cases")
    parser.add_argument("--retry-timeouts", action="store_true")
    parser.add_argument(
        "--retry-timeouts-only", action="store_true",
        help="retry only saved timeouts with a shorter limit; preserve solved results and skip attempts already made at the requested limit",
    )
    parser.add_argument("--rerun-all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--product-encoding", choices=("factored", "direct"), default="factored",
        help="product-relation closure representation (default: factored)",
    )
    parser.add_argument(
        "--bfs-parent-redundancy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include logically redundant BFS parent clauses (default)",
    )
    parser.add_argument(
        "--bfs-converse", "--bfs-parent-strengthening", dest="bfs_parent_redundancy",
        action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_states < 3 or args.max_states < args.min_states:
        raise SystemExit("state range must start at 3 or higher and be nonempty")
    if args.timeout_seconds <= 0 or args.heartbeat_seconds <= 0 or args.max_new_runs < 0:
        raise SystemExit("timeout and heartbeat must be positive; max-new-runs cannot be negative")
    if args.retry_timeouts_only and (args.retry_timeouts or args.rerun_all):
        raise SystemExit("--retry-timeouts-only cannot be combined with --retry-timeouts or --rerun-all")
    output_root = output_root_for_order(args.output_root.resolve(), args.factor_order)
    if args.optimization != "sat":
        output_root /= args.optimization
    if not args.bfs_parent_redundancy:
        output_root /= "without-bfs-parent-redundancy"
    if args.product_encoding != "factored":
        output_root /= f"product-{args.product_encoding}"
    cases = scan_cases(args.min_states, args.max_states)
    if not cases:
        raise SystemExit("no matching DFA inputs")
    counts = Counter((case.state_count, case.kind) for case in cases)
    print(f"[inventory] {len(cases)} DFAs from {args.min_states} through {args.max_states} states", flush=True)
    for state in range(args.min_states, args.max_states + 1):
        print(
            f"[inventory] {state} states: {counts[state, 'minimal']} minimal + "
            f"{counts[state, 'nonminimal']} nonminimal",
            flush=True,
        )

    results: dict[tuple[str, str], dict[str, Any]] = {}
    pending: list[InputCase] = []
    retry_source_hashes: dict[tuple[str, str], str] = {}
    for case in cases:
        path = result_path(output_root, case)
        matching_timeout = args.timeout_seconds
        if args.retry_timeouts_only:
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                matching_timeout = saved["benchmark"]["timeout_seconds"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise SystemExit(f"Cannot verify a saved attempt; refusing to run new cases: {path}") from error
            if type(matching_timeout) is not int or not 0 < matching_timeout <= args.timeout_seconds:
                raise SystemExit(f"Saved limit is invalid or exceeds the requested limit: {path}")
        existing = matching_result(
            path, case, matching_timeout, args.factor_order,
            require_current_encoding=not args.report_only,
            bfs_parent_redundancy=args.bfs_parent_redundancy,
            optimization=args.optimization,
            product_encoding=args.product_encoding,
        )
        if existing is not None:
            results[(case.kind, case.case_id)] = existing
        if args.retry_timeouts_only:
            if existing is None or existing["status"] not in TERMINAL_STATUSES:
                raise SystemExit(f"Saved attempt is incompatible or incomplete; refusing to overwrite: {path}")
            if existing["status"] == "timeout" and matching_timeout < args.timeout_seconds:
                pending.append(case)
                retry_source_hashes[(case.kind, case.case_id)] = sha256(path)
            continue
        if (
            args.rerun_all
            or existing is None
            or existing["status"] not in TERMINAL_STATUSES
            or (args.retry_timeouts and existing["status"] == "timeout")
        ):
            pending.append(case)
    print(f"[plan] {len(pending)} to run; {len(cases) - len(pending)} reusable results", flush=True)
    if args.retry_timeouts_only:
        for case in pending:
            previous_limit = results[case.kind, case.case_id]["benchmark"]["timeout_seconds"]
            print(f"[retry] {case.case_id}: {previous_limit}s -> {args.timeout_seconds}s", flush=True)
    if args.dry_run:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    inventory_path = output_root / "input_manifest.json"
    if args.retry_timeouts_only and inventory_path.is_file():
        archive_saved_file(inventory_path, output_root / "timeout_retry_history" / "manifests")
    write_json(
        inventory_path,
        {
            "dataset": DATASET_NAME,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "min_states": args.min_states,
            "max_states": args.max_states,
            "timeout_seconds": args.timeout_seconds,
            "configuration": {
                "optimization": args.optimization,
                "strictness_strategy": "witness",
                "strictness_mode": "finite",
                "nonempty_requirement": NONEMPTY_REQUIREMENT,
                "symmetry_breaking": "bfs",
                "bfs_parent_redundancy": args.bfs_parent_redundancy,
                "product_encoding": args.product_encoding,
                "encoding_revision": (
                    "historical-report-only" if args.report_only else encoding_revision(args.optimization, args.product_encoding)
                ),
                "factor_order": args.factor_order,
                "sat_solver": "g3" if args.optimization in ("sat", "enumeration") else None,
                "timeout_retries_only": args.retry_timeouts_only,
            },
            "cases": [
                {
                    "case_id": case.case_id,
                    "input_kind": case.kind,
                    "states": case.state_count,
                    "input_path": case.benchmark.input_path.as_posix(),
                    "input_dot_sha256": case.benchmark.dot_sha256,
                }
                for case in cases
            ],
        },
    )
    write_reports(cases, results, output_root, args.timeout_seconds, args.rank_by, args.min_states, args.max_states)
    if args.report_only:
        return 0

    limit = args.max_new_runs or len(pending)
    chosen = pending[:limit]
    print(f"[start] executing {len(chosen)} case(s)", flush=True)
    try:
        for index, case in enumerate(chosen, start=1):
            path = result_path(output_root, case)
            prior_archive = None
            if args.retry_timeouts_only:
                if sha256(path) != retry_source_hashes[(case.kind, case.case_id)]:
                    raise RuntimeError(f"Saved attempt changed during retry planning: {path}")
                archive_directory = (
                    output_root / "timeout_retry_history"
                    / f"states_{case.state_count:02d}" / case.kind / case.case_id
                )
                prior_archive = archive_saved_file(path, archive_directory)
                log_path = path.parent / "solver.log"
                if log_path.is_file():
                    archive_saved_file(log_path, archive_directory)
            payload = _run_one(
                case.benchmark,
                path.parent,
                "witness",
                args.optimization,
                "bfs",
                "finite",
                args.factor_order,
                args.timeout_seconds,
                args.heartbeat_seconds,
                index,
                len(chosen),
                inventory_path,
                dataset_name=DATASET_NAME,
                bfs_parent_redundancy=args.bfs_parent_redundancy,
                product_encoding=args.product_encoding,
            )
            if prior_archive is not None:
                payload["timeout_retry"] = {
                    "previous_timeout_seconds": results[case.kind, case.case_id]["benchmark"]["timeout_seconds"],
                    "previous_result_sha256": retry_source_hashes[(case.kind, case.case_id)],
                    "previous_result_archive": prior_archive.resolve().as_posix(),
                    "requested_timeout_seconds": args.timeout_seconds,
                }
                write_json(path, payload)
            results[(case.kind, case.case_id)] = payload
            write_reports(
                cases, results, output_root, args.timeout_seconds, args.rank_by,
                args.min_states, args.max_states,
            )
    except KeyboardInterrupt:
        print("[stopped] current solver terminated; completed results are saved", flush=True)
        return 130
    print(f"[done] reports: {output_root / 'reports'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
