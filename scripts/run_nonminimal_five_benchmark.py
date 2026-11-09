"""Run decomposition benchmarks on five larger DFAs per output-state size."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from dfa_identify.dot_io import dfa_from_dot
from dfa_identify.minimum_edit import normalize_optimization

from run_minimal_five_benchmark import (
    BenchmarkCase,
    _read_matching_result,
    _run_one,
    _sha256,
    result_directory,
)

DATASET_NAME = "dfaminer-nonminimum-five-per-output-size"
DEFAULT_MANIFEST = Path(
    "benchmarks/dfaminer-nonminimal/five_per_output_size_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path("benchmarks/dfaminer-nonminimal/runs-five-per-output-size")


def _recorded_or_fallback_path(
    recorded: str | None,
    fallback: Path,
) -> Path:
    if recorded:
        candidate = Path(recorded)
        if candidate.is_file():
            return candidate.resolve()
    return fallback.resolve()


def load_nonminimal_cases(
    manifest_path: Path,
    *,
    min_states: int = 5,
    max_states: int = 15,
    requested_case_ids: Sequence[str] | None = None,
    validate_dots: bool = True,
) -> list[BenchmarkCase]:
    """Load only the five cases explicitly selected for each output size."""

    manifest_path = manifest_path.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"balanced manifest not found: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid manifest JSON: {manifest_path}: {error}") from error

    if not manifest.get("complete"):
        raise SystemExit(
            "the five-per-output-size manifest is incomplete; finish its "
            "generation first"
        )
    if manifest.get("group_by") != "actual_output_state_count":
        raise SystemExit("manifest must be grouped by actual output-DFA state count")
    if min_states > max_states:
        raise SystemExit("--min-states must not be greater than --max-states")

    requested = (
        {value.removesuffix(".dot") for value in requested_case_ids}
        if requested_case_ids
        else None
    )
    selected = manifest.get("selected")
    if not isinstance(selected, dict):
        raise SystemExit("manifest field 'selected' must be an object")

    cases: list[BenchmarkCase] = []
    found_requested: set[str] = set()
    seen_case_ids: set[str] = set()
    seen_base_cases: set[str] = set()
    for group_key in sorted(selected, key=int):
        state_count = int(group_key)
        if not min_states <= state_count <= max_states:
            continue
        records = selected[group_key]
        if not isinstance(records, list):
            raise SystemExit(f"manifest group {group_key!r} must be a list")
        if requested is None and len(records) != 5:
            raise SystemExit(
                f"output-size group {state_count} contains {len(records)} "
                "selected cases instead of five"
            )

        for record in records:
            case_id = str(record["case_id"])
            if requested is not None and case_id not in requested:
                continue
            if case_id in seen_case_ids:
                raise SystemExit(f"duplicate selected case id: {case_id}")
            seen_case_ids.add(case_id)
            found_requested.add(case_id)

            if record.get("status") != "success":
                raise SystemExit(f"selected case is not successful: {case_id}")
            requested_states = int(record["requested_state_count"])
            actual_states = int(
                record.get("validation", {}).get("actual_state_count", requested_states)
            )
            if requested_states != state_count or actual_states != state_count:
                raise SystemExit(
                    f"{case_id}: group={state_count}, requested="
                    f"{requested_states}, actual={actual_states}"
                )
            minimum_states = int(record["minimum_consistent_states"])
            if minimum_states >= state_count:
                raise SystemExit(
                    f"{case_id}: base minimum {minimum_states} is not smaller "
                    f"than output size {state_count}"
                )

            base_case = str(record["base_case"])
            if base_case in seen_base_cases:
                raise SystemExit(f"selected base case is reused: {base_case}")
            seen_base_cases.add(base_case)
            source_size = int(record["source_size"])
            instance_id = int(record["instance_id"])
            input_path = _recorded_or_fallback_path(
                record.get("output_dot"),
                manifest_path.parent
                / "dot"
                / f"states_{state_count:02d}"
                / f"{case_id}.dot",
            )
            if not input_path.is_file():
                raise SystemExit(f"input DOT not found for {case_id}: {input_path}")
            expected_hash = record.get("validation", {}).get("dot_sha256")
            if validate_dots and expected_hash and _sha256(input_path) != expected_hash:
                raise SystemExit(f"input DOT hash does not match manifest: {case_id}")
            if validate_dots:
                parsed_states = len(tuple(dfa_from_dot(input_path).states()))
                if parsed_states != state_count:
                    raise SystemExit(
                        f"{case_id}: DOT has {parsed_states} states, expected "
                        f"{state_count}"
                    )

            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    input_path=input_path,
                    input_state_count=state_count,
                    source_size=source_size,
                    instance_id=instance_id,
                    dot_sha256=expected_hash,
                    sample=record.get("copied_sample") or record.get("source_sample"),
                    metadata={
                        "base_case": base_case,
                        "base_dot": record.get("base_dot"),
                        "minimum_consistent_states": minimum_states,
                        "extra_states": int(
                            record.get("extra_states", state_count - minimum_states)
                        ),
                        "selection_origin": record.get("selection_origin"),
                        "language_equivalence_to_base_required": False,
                    },
                )
            )

    if requested is not None:
        missing = sorted(requested - found_requested)
        if missing:
            raise SystemExit(
                "requested case(s) are not in the balanced selection or state "
                "range: " + ", ".join(missing)
            )
    if not cases:
        raise SystemExit("no selected cases match the requested filters")

    cases.sort(key=lambda item: (item.input_state_count, item.case_id))
    if requested is None:
        counts = Counter(case.input_state_count for case in cases)
        incomplete = {
            state_count: counts.get(state_count, 0)
            for state_count in range(min_states, max_states + 1)
            if counts.get(state_count, 0) != 5
        }
        if incomplete:
            details = ", ".join(
                f"n={state_count}: {count}" for state_count, count in incomplete.items()
            )
            raise SystemExit(f"expected five selected cases per size; found {details}")
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "run the balanced five-per-output-size non-minimum DFA "
            "decomposition benchmark"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("witness",),
        default=("witness",),
        help="strictness encoding to run (currently: witness only)",
    )
    parser.add_argument(
        "--optimizations",
        nargs="+",
        type=normalize_optimization, choices=("maxsat", "sat", "enumeration"),
        default=("maxsat",),
        help="optimization backend(s) to run (default: maxsat)",
    )
    parser.add_argument("--min-states", type=int, default=5)
    parser.add_argument("--max-states", type=int, default=15)
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE_ID",
        help="run only selected IDs, for example nm-05-from-04-001",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=0,
        help="stop after this many newly executed combinations; 0 is unlimited",
    )
    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument(
        "--retry-failures",
        action="store_true",
        help="rerun saved timeout/error results while retaining SAT/UNSAT",
    )
    retry_group.add_argument(
        "--rerun-all",
        action="store_true",
        help="replace every matching saved result",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate selected inputs and print the plan without a solver",
    )
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
    if args.timeout_seconds <= 0 or args.heartbeat_seconds <= 0:
        raise SystemExit("timeout and heartbeat values must be positive")
    if args.max_new_runs < 0:
        raise SystemExit("--max-new-runs must be zero or positive")
    if tuple(args.strategies) != ("witness",):
        raise SystemExit(
            "all-factor-size benchmarks currently require --strategies witness"
        )
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    if not args.bfs_parent_redundancy:
        output_root /= "without-bfs-parent-redundancy"
    if args.product_encoding != "factored":
        output_root /= f"product-{args.product_encoding}"
    cases = load_nonminimal_cases(
        manifest_path,
        min_states=args.min_states,
        max_states=args.max_states,
        requested_case_ids=args.cases,
    )
    plan = [
        (case, optimization, strategy)
        for strategy in args.strategies
        for optimization in args.optimizations
        for case in cases
    ]
    decisions: list[tuple[BenchmarkCase, str, str, str, str | None]] = []
    for case, optimization, strategy in plan:
        result_path = (
            result_directory(output_root, case, optimization, strategy) / "result.json"
        )
        existing = _read_matching_result(
            result_path,
            case,
            optimization,
            strategy,
            DATASET_NAME,
            require_current_encoding=True,
            bfs_parent_redundancy=args.bfs_parent_redundancy,
            product_encoding=args.product_encoding,
        )
        existing_status = existing.get("status") if existing else None
        if args.rerun_all:
            action = "run"
        elif (
            existing is not None
            and args.retry_failures
            and existing_status in {"timeout", "error"}
        ):
            action = "run"
        elif existing is not None:
            action = "skip"
        else:
            action = "run"
        decisions.append((case, optimization, strategy, action, existing_status))

    counts = Counter(
        (optimization, strategy, action)
        for _, optimization, strategy, action, _ in decisions
    )
    state_distribution = Counter(case.input_state_count for case in cases)
    print(
        "[inputs] selected actual states: "
        + ", ".join(
            f"{state}={state_distribution[state]}"
            for state in sorted(state_distribution)
        ),
        flush=True,
    )
    print(
        f"[plan] {len(cases)} selected input DFAs; {len(plan)} combinations; "
        f"output={output_root}",
        flush=True,
    )
    for strategy in args.strategies:
        for optimization in args.optimizations:
            print(
                f"[plan] {optimization}/{strategy}: "
                f"run={counts[optimization, strategy, 'run']} | "
                f"reuse={counts[optimization, strategy, 'skip']}",
                flush=True,
            )
    if args.dry_run:
        print("[dry-run] validation complete; no solver was started", flush=True)
        return 0

    runnable = [decision for decision in decisions if decision[3] == "run"]
    if not runnable:
        print("[done] every requested result is already saved", flush=True)
        return 0
    limit = args.max_new_runs or len(runnable)
    selected_runs = runnable[:limit]
    print(
        f"[start] executing {len(selected_runs)} new run(s); "
        f"{len(runnable) - len(selected_runs)} left for later",
        flush=True,
    )

    outcome_counts: Counter[str] = Counter()
    try:
        for run_number, (case, optimization, strategy, _, _) in enumerate(
            selected_runs, start=1
        ):
            payload = _run_one(
                case=case,
                output_directory=result_directory(
                    output_root, case, optimization, strategy
                ),
                strategy=strategy,
                optimization=optimization,
                symmetry_breaking="bfs",
                strictness_mode="finite",
                factor_order="sum",
                timeout_seconds=args.timeout_seconds,
                heartbeat_seconds=args.heartbeat_seconds,
                run_number=run_number,
                run_total=len(selected_runs),
                manifest_path=manifest_path,
                dataset_name=DATASET_NAME,
                bfs_parent_redundancy=args.bfs_parent_redundancy,
                product_encoding=args.product_encoding,
            )
            outcome_counts[str(payload.get("status", "error"))] += 1
    except KeyboardInterrupt:
        print(
            "[stopped] interrupted safely; completed results remain reusable",
            flush=True,
        )
        return 130

    summary = ", ".join(
        f"{status}={count}" for status, count in sorted(outcome_counts.items())
    )
    print(f"[done] {summary}", flush=True)
    print(
        "[next] poetry run python " "scripts/build_nonminimal_five_benchmark_report.py",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
