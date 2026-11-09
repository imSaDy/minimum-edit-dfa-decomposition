"""Run decomposition experiments on the minimal five-per-size DFA set.

The input set is read from ``five_per_size_manifest.json`` and is grouped by
the actual state count of each learned minimal DFA.  Results are isolated from
both the older four-per-size benchmark and the non-minimal DFAMiner generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dfa_identify.dot_io import dfa_from_dot
from dfa_identify.minimum_edit import ENCODING_REVISION, encoding_revision, factor_size_pairs, normalize_optimization

DATASET_NAME = "dfaminer-minimal-five-per-actual-size"
DEFAULT_MANIFEST = Path("benchmarks/dfaminer/five_per_size_manifest.json")
DEFAULT_OUTPUT_ROOT = Path("benchmarks/dfaminer/runs-five-minimal")
STRATEGY_DIRECTORIES = {
    "witness": "strictness",
    "blocking": "no-strictness",
}
TERMINAL_STATUSES = {"sat", "unsat", "timeout", "error"}
FACTOR_SIZE_SEARCH_MODE = "all-proper-unordered"
SEARCH_ORDERS = {
    "maxsat": "factor-size-then-optimum",
    "sat": "edit-budget-then-factor-size",
    "enumeration": "edit-distance-then-factor-size-then-repair",
}


@dataclass(frozen=True)
class BenchmarkCase:
    """One selected input DFA and its provenance."""

    case_id: str
    input_path: Path
    input_state_count: int
    source_size: int
    instance_id: int
    dot_sha256: str | None
    sample: str | None
    metadata: dict[str, Any] | None = None


def factor_size_candidates(
    input_state_count: int, factor_order: str = "sum"
) -> tuple[tuple[int, int], ...]:
    """Return the exact-size subproblems covered by one benchmark run."""

    return factor_size_pairs(input_state_count, ordering=factor_order)  # type: ignore[arg-type]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(
    manifest_path: Path,
    *,
    min_states: int = 4,
    max_states: int = 15,
    requested_case_ids: Sequence[str] | None = None,
    validate_dots: bool = True,
) -> list[BenchmarkCase]:
    """Load and validate selected cases from the completed learned-size manifest."""

    manifest_path = manifest_path.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"manifest not found: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid manifest JSON: {manifest_path}: {error}") from error

    if manifest.get("group_by") != "learned":
        raise SystemExit(
            "the manifest must be grouped by the actual learned DFA size "
            "(group_by='learned')"
        )
    if not manifest.get("complete"):
        raise SystemExit(
            "the five-per-size manifest is not complete; finish its preparation first"
        )
    if min_states > max_states:
        raise SystemExit("--min-states must not be greater than --max-states")

    requested = (
        {value.removesuffix(".dot") for value in requested_case_ids}
        if requested_case_ids
        else None
    )
    found_requested: set[str] = set()
    seen_case_ids: set[str] = set()
    cases: list[BenchmarkCase] = []
    selected = manifest.get("selected")
    if not isinstance(selected, dict):
        raise SystemExit("manifest field 'selected' must be an object")

    for group_key, entries in selected.items():
        try:
            group_state_count = int(group_key)
        except (TypeError, ValueError) as error:
            raise SystemExit(
                f"invalid state-count group in manifest: {group_key!r}"
            ) from error
        if not min_states <= group_state_count <= max_states:
            continue
        if not isinstance(entries, list):
            raise SystemExit(f"manifest group {group_key!r} must be a list")

        for record in entries:
            case_id = str(record["case_id"])
            if requested is not None and case_id not in requested:
                continue
            if case_id in seen_case_ids:
                raise SystemExit(f"duplicate selected case in manifest: {case_id}")
            seen_case_ids.add(case_id)
            found_requested.add(case_id)

            learned_state_count = int(record["learned_state_count"])
            if learned_state_count != group_state_count:
                raise SystemExit(
                    f"{case_id}: learned_state_count={learned_state_count}, "
                    f"but it is listed under group {group_state_count}"
                )
            input_path = (manifest_path.parent / record["dot"]).resolve()
            if not input_path.is_file():
                raise SystemExit(f"input DOT not found for {case_id}: {input_path}")

            expected_hash = record.get("dot_sha256")
            if validate_dots and expected_hash and _sha256(input_path) != expected_hash:
                raise SystemExit(
                    f"input DOT hash does not match the manifest: {case_id}"
                )
            if validate_dots:
                parsed_state_count = len(tuple(dfa_from_dot(input_path).states()))
                if parsed_state_count != learned_state_count:
                    raise SystemExit(
                        f"{case_id}: DOT has {parsed_state_count} states, "
                        f"manifest says {learned_state_count}"
                    )

            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    input_path=input_path,
                    input_state_count=learned_state_count,
                    source_size=int(record["source_size"]),
                    instance_id=int(record["instance_id"]),
                    dot_sha256=expected_hash,
                    sample=record.get("sample"),
                )
            )

    if requested is not None:
        missing = sorted(requested - found_requested)
        if missing:
            raise SystemExit(
                "requested case(s) are not selected in this state range: "
                + ", ".join(missing)
            )

    cases.sort(key=lambda item: (item.input_state_count, item.case_id))
    if not cases:
        raise SystemExit("no selected cases match the requested filters")

    counts = Counter(case.input_state_count for case in cases)
    if requested is None:
        incomplete = {
            state_count: counts.get(state_count, 0)
            for state_count in range(min_states, max_states + 1)
            if counts.get(state_count, 0) != 5
        }
        if incomplete:
            details = ", ".join(
                f"n={state_count}: {count}" for state_count, count in incomplete.items()
            )
            raise SystemExit(f"expected five cases per requested size; found {details}")
    return cases


def result_directory(
    output_root: Path,
    case: BenchmarkCase,
    optimization: str,
    strategy: str,
    symmetry_breaking: str = "bfs",
    strictness_mode: str = "finite",
    factor_order: str = "sum",
) -> Path:
    base = (
        output_root
        / optimization
        / STRATEGY_DIRECTORIES[strategy]
    )
    # Preserve the old layout for reading historical baseline results.
    # All new solver runs use the BFS subtree.
    if symmetry_breaking != "none":
        base = base / f"symmetry-{symmetry_breaking}"
    if strictness_mode != "finite":
        base = base / f"strictness-{strictness_mode}"
    if factor_order != "sum":
        base = base / f"factor-order-{factor_order}"
    return base / f"states_{case.input_state_count:02d}" / case.case_id


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=repr) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_matching_result(
    path: Path,
    case: BenchmarkCase,
    optimization: str,
    strategy: str,
    dataset_name: str = DATASET_NAME,
    symmetry_breaking: str = "bfs",
    strictness_mode: str = "finite",
    factor_order: str = "sum",
    require_current_encoding: bool = False,
    bfs_parent_redundancy: bool | None = None,
    product_encoding: str | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        return None
    if (
        require_current_encoding
        and benchmark.get("encoding_revision") != encoding_revision(optimization, product_encoding or "factored")
    ):
        return None
    if (
        benchmark.get("dataset") != dataset_name
        or benchmark.get("case_id") != case.case_id
        or benchmark.get("input_state_count") != case.input_state_count
        or ("enumeration" if benchmark.get("optimization_backend") == "hybrid"
            else benchmark.get("optimization_backend")) != optimization
        or benchmark.get("strategy") != strategy
        or benchmark.get("symmetry_breaking", "none") != symmetry_breaking
        or benchmark.get("strictness_mode", "finite") != strictness_mode
        or benchmark.get("factor_order", "sum") != factor_order
        or benchmark.get("factor_size_search_mode") != FACTOR_SIZE_SEARCH_MODE
        or (
            optimization in ("sat", "enumeration")
            and benchmark.get("factor_size_search_order")
            != SEARCH_ORDERS[optimization]
        )
        or (
            optimization == "maxsat"
            and benchmark.get(
                "factor_size_search_order", SEARCH_ORDERS[optimization]
            )
            != SEARCH_ORDERS[optimization]
        )
        or (bfs_parent_redundancy is not None
            and benchmark.get("bfs_parent_redundancy", benchmark.get("bfs_parent_strengthening", benchmark.get("bfs_converse", True))) is not bfs_parent_redundancy)
        or (product_encoding is not None
            and benchmark.get("product_encoding", "factored") != product_encoding)
        or payload.get("status") not in TERMINAL_STATUSES
    ):
        return None
    return payload


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate only the solver process started by this runner."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _tail(value: str | None, limit: int = 4000) -> str:
    return (value or "")[-limit:]


def _run_one(
    case: BenchmarkCase,
    output_directory: Path,
    strategy: str,
    optimization: str,
    symmetry_breaking: str,
    strictness_mode: str,
    factor_order: str,
    timeout_seconds: int,
    heartbeat_seconds: int,
    run_number: int,
    run_total: int,
    manifest_path: Path,
    dataset_name: str = DATASET_NAME,
    bfs_parent_redundancy: bool = True,
    product_encoding: str = "factored",
) -> dict[str, Any]:
    if symmetry_breaking != "bfs":
        raise ValueError("BFS is always enabled for new benchmark runs")
    factor_candidates = factor_size_candidates(case.input_state_count, factor_order)
    output_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / "result.json"
    pending_result = output_directory / f".solver-result-{os.getpid()}.json"
    if pending_result.exists():
        pending_result.unlink()
    visual_directory = output_directory / "visualizations"
    command = [
        sys.executable,
        "-m",
        "dfa_identify.cli",
        str(case.input_path),
        "--strictness" if strategy == "witness" else "--no-strictness",
        "--optimization",
        optimization,
        "--product-encoding",
        product_encoding,
        "--strictness-mode",
        strictness_mode,
        "--factor-order",
        factor_order,
        "--bfs-parent-redundancy" if bfs_parent_redundancy else "--no-bfs-parent-redundancy",
        "--show-progress",
        "--output",
        str(pending_result),
        "--visualize-dir",
        str(visual_directory),
    ]
    popen_options: dict[str, Any] = {}
    if os.name == "nt":
        popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_options["start_new_session"] = True

    print(
        f"[run {run_number}/{run_total}] {case.case_id} | "
        f"actual states={case.input_state_count} | "
        f"factor pairs={len(factor_candidates)} "
        f"| {optimization}/{strategy}/strictness-{strictness_mode}/"
        f"factor-order-{factor_order}/symmetry-{symmetry_breaking}",
        flush=True,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        **popen_options,
    )
    print(f"[start] PID={process.pid}; timeout={timeout_seconds}s", flush=True)

    output_lines: deque[str] = deque(maxlen=200)
    solver_log = output_directory / "solver.log"

    def forward_solver_output() -> None:
        assert process.stdout is not None
        with solver_log.open("w", encoding="utf-8", buffering=1) as stream:
            for line in process.stdout:
                stream.write(line)
                output_lines.append(line)
                print(f"[solver] {line}", end="", flush=True)

    output_thread = threading.Thread(target=forward_solver_output, daemon=True)
    output_thread.start()

    timed_out = False
    try:
        while True:
            elapsed = time.perf_counter() - started
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                timed_out = True
                _terminate_process_tree(process)
                process.wait()
                break
            try:
                process.wait(timeout=min(heartbeat_seconds, remaining))
                break
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - started
                print(
                    f"[alive] {case.case_id} "
                    f"{optimization}/{strategy}/strictness-{strictness_mode}/"
                    f"factor-order-{factor_order}/symmetry-{symmetry_breaking} | "
                    f"elapsed={elapsed:.0f}s | "
                    f"remaining={max(0, timeout_seconds - elapsed):.0f}s",
                    flush=True,
                )
    except KeyboardInterrupt:
        print(f"[interrupt] stopping only PID={process.pid}", flush=True)
        _terminate_process_tree(process)
        process.wait()
        if pending_result.exists():
            pending_result.unlink()
        raise
    finally:
        output_thread.join(timeout=5)

    solver_output_tail = _tail("".join(output_lines))

    wall_seconds = time.perf_counter() - started
    benchmark = {
        "encoding_revision": encoding_revision(optimization, product_encoding),
        "product_encoding": product_encoding,
        "bfs_parent_redundancy": bfs_parent_redundancy,
        "dataset": dataset_name,
        "manifest": manifest_path.resolve().as_posix(),
        "case_id": case.case_id,
        "input": case.input_path.as_posix(),
        "input_dot_sha256": case.dot_sha256,
        "input_state_count": case.input_state_count,
        "source_size": case.source_size,
        "source_instance_id": case.instance_id,
        "source_sample": case.sample,
        "factor_size_search_mode": FACTOR_SIZE_SEARCH_MODE,
        "factor_size_search_order": SEARCH_ORDERS[optimization],
        "factor_size_candidates": [list(pair) for pair in factor_candidates],
        "strategy": strategy,
        "strictness_mode": strictness_mode,
        "factor_order": factor_order,
        "optimization_backend": optimization,
        "symmetry_breaking": symmetry_breaking,
        "timeout_seconds": timeout_seconds,
        "started_at_utc": started_at,
        "wall_seconds": round(wall_seconds, 6),
        "solver_log": solver_log.resolve().as_posix(),
    }
    if case.metadata:
        benchmark.update(case.metadata)

    if timed_out:
        payload: dict[str, Any] = {
            "status": "timeout",
            "strictness_strategy": strategy,
            "strictness_mode": strictness_mode,
            "optimization_backend": optimization,
            "symmetry_breaking": symmetry_breaking,
        }
        print(
            f"[timeout] {case.case_id} after {wall_seconds:.3f}s",
            flush=True,
        )
    elif process.returncode != 0 or not pending_result.is_file():
        payload = {
            "status": "error",
            "strictness_strategy": strategy,
            "strictness_mode": strictness_mode,
            "optimization_backend": optimization,
            "symmetry_breaking": symmetry_breaking,
            "process_returncode": process.returncode,
            "stdout_tail": solver_output_tail,
            "stderr_tail": solver_output_tail,
        }
        print(
            f"[error] {case.case_id}; return code={process.returncode}",
            flush=True,
        )
    else:
        try:
            payload = json.loads(pending_result.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            payload = {
                "status": "error",
                "strictness_strategy": strategy,
                "strictness_mode": strictness_mode,
                "optimization_backend": optimization,
                "symmetry_breaking": symmetry_breaking,
                "error": f"could not read solver result: {error}",
                "stdout_tail": solver_output_tail,
                "stderr_tail": solver_output_tail,
            }
        else:
            print(
                f"[result] {case.case_id}: {payload.get('status')} | "
                f"edit distance={payload.get('edit_distance', '-')} | "
                f"solve={payload.get('timing', {}).get('solve_seconds', '-')}s | "
                f"wall={wall_seconds:.3f}s",
                flush=True,
            )

    benchmark["factor_sizes"] = payload.get("factor_sizes")
    payload["bfs_parent_redundancy"] = bfs_parent_redundancy
    payload["product_encoding"] = product_encoding
    payload["benchmark"] = benchmark
    _write_json_atomic(result_path, payload)
    if pending_result.exists():
        pending_result.unlink()
    print(f"[saved] {result_path}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the five-minimal-DFA-per-size decomposition benchmark and "
            "save resumable timing/result JSON files."
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
    parser.add_argument(
        "--strictness-modes",
        nargs="+",
        choices=("finite", "infinite"),
        default=("finite",),
        help="strictness semantics to benchmark (default: finite)",
    )
    parser.add_argument(
        "--factor-orders",
        nargs="+",
        choices=("sum", "max", "product"),
        default=("sum",),
        help="factor-size secondary objective(s) to benchmark (default: sum)",
    )
    parser.add_argument("--min-states", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=15)
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE_ID",
        help="run only these selected IDs, for example dfa-04-001",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=0,
        help="stop after this many newly executed combinations; 0 means unlimited",
    )
    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument(
        "--retry-failures",
        action="store_true",
        help="rerun saved timeout/error results while retaining SAT/UNSAT results",
    )
    retry_group.add_argument(
        "--rerun-all",
        action="store_true",
        help="replace every matching saved result in the requested selection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the plan without starting a solver",
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
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.heartbeat_seconds <= 0:
        raise SystemExit("--heartbeat-seconds must be positive")
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
    cases = load_cases(
        manifest_path,
        min_states=args.min_states,
        max_states=args.max_states,
        requested_case_ids=args.cases,
    )

    # Strictness witnesses are encoded directly for every selected backend.
    plan = [
        (
            case,
            optimization,
            strategy,
            symmetry_breaking,
            strictness_mode,
            factor_order,
        )
        for strategy in args.strategies
        for optimization in args.optimizations
        for symmetry_breaking in ("bfs",)
        for strictness_mode in args.strictness_modes
        for factor_order in args.factor_orders
        for case in cases
    ]
    decisions: list[
        tuple[BenchmarkCase, str, str, str, str, str, str, str | None]
    ] = []
    for (
        case,
        optimization,
        strategy,
        symmetry_breaking,
        strictness_mode,
        factor_order,
    ) in plan:
        result_path = (
            result_directory(
                output_root,
                case,
                optimization,
                strategy,
                symmetry_breaking,
                strictness_mode,
                factor_order,
            )
            / "result.json"
        )
        existing = _read_matching_result(
            result_path,
            case,
            optimization,
            strategy,
            symmetry_breaking=symmetry_breaking,
            strictness_mode=strictness_mode,
            factor_order=factor_order,
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
        decisions.append(
            (
                case,
                optimization,
                strategy,
                symmetry_breaking,
                strictness_mode,
                factor_order,
                action,
                existing_status,
            )
        )

    counts = Counter(
        (
            optimization,
            strategy,
            symmetry_breaking,
            strictness_mode,
            factor_order,
            action,
        )
        for (
            _,
            optimization,
            strategy,
            symmetry_breaking,
            strictness_mode,
            factor_order,
            action,
            _,
        ) in decisions
    )
    print(
        f"[plan] {len(cases)} input DFAs; {len(plan)} requested combinations; "
        f"output={output_root}",
        flush=True,
    )
    for strategy in args.strategies:
        for optimization in args.optimizations:
            for symmetry_breaking in ("bfs",):
                for strictness_mode in args.strictness_modes:
                    for factor_order in args.factor_orders:
                        key = (
                            optimization,
                            strategy,
                            symmetry_breaking,
                            strictness_mode,
                            factor_order,
                        )
                        print(
                            f"[plan] {optimization}/{strategy}/"
                            f"strictness-{strictness_mode}/"
                            f"factor-order-{factor_order}/"
                            f"symmetry-{symmetry_breaking}: "
                            f"run={counts[(*key, 'run')]} | "
                            f"reuse={counts[(*key, 'skip')]}",
                            flush=True,
                        )
    if args.dry_run:
        print("[dry-run] validation complete; no solver was started", flush=True)
        return 0

    runnable = [decision for decision in decisions if decision[6] == "run"]
    if not runnable:
        print("[done] every requested result is already saved", flush=True)
        return 0
    limit = args.max_new_runs or len(runnable)
    selected_runs = runnable[:limit]
    print(
        f"[start] executing {len(selected_runs)} new run(s); "
        f"{len(runnable) - len(selected_runs)} left for a later invocation",
        flush=True,
    )

    outcome_counts: Counter[str] = Counter()
    try:
        for run_number, (
            case,
            optimization,
            strategy,
            symmetry_breaking,
            strictness_mode,
            factor_order,
            _,
            _,
        ) in enumerate(
            selected_runs, start=1
        ):
            output_directory = result_directory(
                output_root,
                case,
                optimization,
                strategy,
                symmetry_breaking,
                strictness_mode,
                factor_order,
            )
            payload = _run_one(
                case,
                output_directory,
                strategy,
                optimization,
                symmetry_breaking,
                strictness_mode,
                factor_order,
                args.timeout_seconds,
                args.heartbeat_seconds,
                run_number,
                len(selected_runs),
                manifest_path,
                bfs_parent_redundancy=args.bfs_parent_redundancy,
                product_encoding=args.product_encoding,
            )
            outcome_counts[str(payload.get("status", "error"))] += 1
    except KeyboardInterrupt:
        print(
            "[stopped] interrupted safely; completed result files remain reusable",
            flush=True,
        )
        return 130

    summary = ", ".join(
        f"{status}={count}" for status, count in sorted(outcome_counts.items())
    )
    print(f"[done] {summary}", flush=True)
    print(
        "[next] build CSV files and charts with: poetry run python "
        "scripts/build_minimal_five_benchmark_report.py",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
