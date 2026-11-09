"""Build tables and charts for the minimal five-per-size benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt

from run_minimal_five_benchmark import (
    DATASET_NAME,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    BenchmarkCase,
    _read_matching_result,
    factor_size_candidates,
    load_cases,
    result_directory,
)

STRATEGIES = {
    "witness": "Strictness Note",
}
OPTIMIZATIONS = {
    "maxsat": "MaxSAT",
    "sat": "Ordinary SAT",
    "enumeration": "Edit-distance enumeration + SAT",
}
SYMMETRY_MODES = {
    "none": "Historical baseline (no BFS)",
    "bfs": "BFS symmetry",
}
COLORS = {
    ("maxsat", "none"): "#2563eb",
    ("maxsat", "bfs"): "#7c3aed",
    ("sat", "none"): "#0891b2",
    ("sat", "bfs"): "#ea580c",
    ("enumeration", "none"): "#15803d",
    ("enumeration", "bfs"): "#16a34a",
}
MARKERS = {"none": "o", "bfs": "s"}
LINESTYLES = {"maxsat": "-", "sat": "--", "enumeration": ":"}


def _mean(values: Iterable[float | int | None]) -> float | None:
    items = [float(value) for value in values if value is not None]
    return statistics.fmean(items) if items else None


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _load_records(
    cases: list[BenchmarkCase],
    output_root: Path,
    dataset_name: str = DATASET_NAME,
    symmetry_modes: Sequence[str] = ("bfs",),
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        candidates = factor_size_candidates(case.input_state_count)
        for optimization in OPTIMIZATIONS:
            for strategy in STRATEGIES:
              for symmetry_breaking in symmetry_modes:
                result_path = (
                    result_directory(
                        output_root,
                        case,
                        optimization,
                        strategy,
                        symmetry_breaking,
                    )
                    / "result.json"
                )
                if optimization == "enumeration" and not result_path.is_file():
                    legacy_path = result_directory(
                        output_root, case, "hybrid", strategy, symmetry_breaking,
                    ) / "result.json"
                    if legacy_path.is_file():
                        result_path = legacy_path
                payload = _read_matching_result(
                    result_path,
                    case,
                    optimization,
                    strategy,
                    dataset_name,
                    symmetry_breaking,
                )
                if payload is None:
                    status = "invalid_result" if result_path.is_file() else "not_run"
                    payload = {}
                else:
                    status = str(payload.get("status", "invalid_result"))
                benchmark = payload.get("benchmark", {})
                timing = payload.get("timing", {})
                search = payload.get("search", {})
                selected_sizes = payload.get("factor_sizes") or [None, None]
                error_text = payload.get("error") or payload.get("stderr_tail")
                records.append(
                    {
                        "case": case.case_id,
                        "input_state_count": case.input_state_count,
                        "source_size": case.source_size,
                        "source_instance_id": case.instance_id,
                        "base_case": (case.metadata or {}).get("base_case"),
                        "minimum_consistent_states": (case.metadata or {}).get(
                            "minimum_consistent_states"
                        ),
                        "extra_states": (case.metadata or {}).get("extra_states"),
                        "input_dot": case.input_path.as_posix(),
                        "factor_1_states": selected_sizes[0],
                        "factor_2_states": selected_sizes[1],
                        "factor_size_candidate_count": len(candidates),
                        "factor_size_attempt_count": len(
                            search.get("factor_size_attempts", [])
                        ),
                        "optimization_backend": optimization,
                        "strategy": strategy,
                        "symmetry_breaking": symmetry_breaking,
                        "bfs_parent_redundancy": benchmark.get("bfs_parent_redundancy", benchmark.get("bfs_parent_strengthening", benchmark.get("bfs_converse", True))),
                        "product_encoding": benchmark.get("product_encoding", "factored"),
                        "encoding_variables": payload.get("encoding_statistics", {}).get("variables"),
                        "encoding_clauses": payload.get("encoding_statistics", {}).get("clauses"),
                        "encoding_literals": payload.get("encoding_statistics", {}).get("literals"),
                        "encoding_revision": benchmark.get(
                            "encoding_revision", "legacy-unversioned"
                        ),
                        "status": status,
                        "solve_seconds": timing.get("solve_seconds"),
                        "wall_seconds": benchmark.get("wall_seconds"),
                        "edit_distance": payload.get("edit_distance"),
                        "solver_calls": search.get("solver_calls"),
                        "checked_models": search.get("checked_models"),
                        "blocked_models": search.get("blocked_models"),
                        "error": error_text,
                        "result": (
                            result_path.resolve().as_posix()
                            if result_path.is_file()
                            else None
                        ),
                    }
                )
    return records


def _write_detailed_csv(records: list[dict[str, Any]], report_directory: Path) -> None:
    fields = (
        "case",
        "input_state_count",
        "source_size",
        "source_instance_id",
        "base_case",
        "minimum_consistent_states",
        "extra_states",
        "input_dot",
        "factor_1_states",
        "factor_2_states",
        "factor_size_candidate_count",
        "factor_size_attempt_count",
        "optimization_backend",
        "strategy",
        "symmetry_breaking",
        "bfs_parent_redundancy",
        "product_encoding",
        "encoding_variables",
        "encoding_clauses",
        "encoding_literals",
        "encoding_revision",
        "status",
        "solve_seconds",
        "wall_seconds",
        "edit_distance",
        "solver_calls",
        "checked_models",
        "blocked_models",
        "error",
        "result",
    )
    with (report_directory / "benchmark_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _summaries(
    records: list[dict[str, Any]],
    cases: list[BenchmarkCase],
    symmetry_modes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    state_counts = sorted({case.input_state_count for case in cases})
    selected_counts = Counter(case.input_state_count for case in cases)
    for state_count in state_counts:
        candidates = factor_size_candidates(state_count)
        for optimization in OPTIMIZATIONS:
            for strategy in STRATEGIES:
              for symmetry_breaking in symmetry_modes:
                group = [
                    record
                    for record in records
                    if record["input_state_count"] == state_count
                    and record["optimization_backend"] == optimization
                    and record["strategy"] == strategy
                    and record["symmetry_breaking"] == symmetry_breaking
                ]
                successful = [record for record in group if record["status"] == "sat"]
                edit_distances = [
                    record["edit_distance"]
                    for record in successful
                    if record["edit_distance"] is not None
                ]
                row = {
                    "input_state_count": state_count,
                    "factor_sizes": "all proper unordered pairs",
                    "factor_size_candidate_count": len(candidates),
                    "optimization_backend": optimization,
                    "strategy": strategy,
                    "symmetry_breaking": symmetry_breaking,
                    "selected_cases": selected_counts[state_count],
                    "recorded_cases": sum(
                        record["status"] != "not_run" for record in group
                    ),
                    "sat_cases": len(successful),
                    "unsat_cases": sum(record["status"] == "unsat" for record in group),
                    "timeout_cases": sum(
                        record["status"] == "timeout" for record in group
                    ),
                    "error_cases": sum(
                        record["status"] in {"error", "invalid_result"}
                        for record in group
                    ),
                    "not_run_cases": sum(
                        record["status"] == "not_run" for record in group
                    ),
                    "mean_solve_seconds": _rounded(
                        _mean(record["solve_seconds"] for record in successful)
                    ),
                    "median_solve_seconds": _rounded(
                        statistics.median(
                            record["solve_seconds"]
                            for record in successful
                            if record["solve_seconds"] is not None
                        )
                        if any(
                            record["solve_seconds"] is not None for record in successful
                        )
                        else None
                    ),
                    "min_solve_seconds": _rounded(
                        min(
                            (
                                record["solve_seconds"]
                                for record in successful
                                if record["solve_seconds"] is not None
                            ),
                            default=None,
                        )
                    ),
                    "max_solve_seconds": _rounded(
                        max(
                            (
                                record["solve_seconds"]
                                for record in successful
                                if record["solve_seconds"] is not None
                            ),
                            default=None,
                        )
                    ),
                    "mean_wall_seconds": _rounded(
                        _mean(record["wall_seconds"] for record in successful)
                    ),
                    "mean_edit_distance": _rounded(_mean(edit_distances)),
                    "min_edit_distance": min(edit_distances, default=None),
                    "max_edit_distance": max(edit_distances, default=None),
                    "mean_solver_calls": _rounded(
                        _mean(record["solver_calls"] for record in successful)
                    ),
                    "mean_checked_models": _rounded(
                        _mean(record["checked_models"] for record in successful)
                    ),
                    "mean_blocked_models": _rounded(
                        _mean(record["blocked_models"] for record in successful)
                    ),
                    "max_blocked_models": max(
                        (
                            record["blocked_models"]
                            for record in successful
                            if record["blocked_models"] is not None
                        ),
                        default=None,
                    ),
                }
                rows.append(row)
    return rows


def _write_summary(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    report_directory: Path,
    manifest_path: Path,
) -> None:
    with (report_directory / "benchmark_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(record["status"] for record in records)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_path.resolve().as_posix(),
        "planned_run_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "groups": rows,
    }
    (report_directory / "benchmark_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _series(
    rows: list[dict[str, Any]],
    optimization: str,
    strategy: str,
    symmetry_breaking: str,
    field: str,
) -> tuple[list[int], list[float]]:
    selected = sorted(
        (
            row
            for row in rows
            if row["optimization_backend"] == optimization
            and row["strategy"] == strategy
            and row["symmetry_breaking"] == symmetry_breaking
            and row[field] is not None
        ),
        key=lambda row: row["input_state_count"],
    )
    return (
        [row["input_state_count"] for row in selected],
        [row[field] for row in selected],
    )


def _line_chart(
    rows: list[dict[str, Any]],
    state_counts: list[int],
    report_directory: Path,
    field: str,
    filename: str,
    ylabel: str,
    *,
    scale: str = "linear",
    symmetry_modes: Sequence[str] = ("bfs",),
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    plotted = False
    positive = False
    for optimization, optimization_label in OPTIMIZATIONS.items():
        for strategy, strategy_label in STRATEGIES.items():
            for symmetry_breaking in symmetry_modes:
                x_values, y_values = _series(
                    rows,
                    optimization,
                    strategy,
                    symmetry_breaking,
                    field,
                )
                if not x_values:
                    continue
                plotted = True
                positive = positive or any(value > 0 for value in y_values)
                axis.plot(
                    x_values,
                    y_values,
                    label=(
                        f"{optimization_label} / {strategy_label} / "
                        f"{SYMMETRY_MODES[symmetry_breaking]}"
                    ),
                    color=COLORS[optimization, symmetry_breaking],
                    marker=MARKERS[symmetry_breaking],
                    linestyle=LINESTYLES[optimization],
                    linewidth=2.2,
                    markersize=6,
                )
    if not plotted:
        axis.text(
            0.5,
            0.5,
            "No completed SAT runs yet",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#64748b",
        )
    axis.set_xlabel("Actual input DFA states")
    axis.set_ylabel(ylabel)
    axis.set_xticks(state_counts)
    axis.set_xlim(min(state_counts) - 0.5, max(state_counts) + 0.5)
    axis.grid(True, which="both", alpha=0.25)
    if scale == "log" and positive:
        axis.set_yscale("log")
    elif scale == "symlog":
        axis.set_yscale("symlog", linthresh=1)
    if plotted:
        axis.legend(frameon=False)
    fig.savefig(report_directory / filename, dpi=180)
    plt.close(fig)


def _runtime_chart(
    records: list[dict[str, Any]],
    state_counts: list[int],
    report_directory: Path,
    symmetry_modes: Sequence[str],
) -> None:
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    series_keys = [
        (optimization, strategy, symmetry_breaking)
        for optimization in OPTIMIZATIONS
        for strategy in STRATEGIES
        for symmetry_breaking in symmetry_modes
    ]
    offsets = {
        key: -0.24 + (0.48 * index / max(1, len(series_keys) - 1))
        for index, key in enumerate(series_keys)
    }
    plotted = False
    positive = False
    for optimization, optimization_label in OPTIMIZATIONS.items():
        for strategy, strategy_label in STRATEGIES.items():
          for symmetry_breaking in symmetry_modes:
            key = (optimization, strategy, symmetry_breaking)
            label = (
                f"{optimization_label} / {strategy_label} / "
                f"{SYMMETRY_MODES[symmetry_breaking]}"
            )
            group = [
                record
                for record in records
                if record["optimization_backend"] == optimization
                and record["strategy"] == strategy
                and record["symmetry_breaking"] == symmetry_breaking
            ]
            successful = [
                record
                for record in group
                if record["status"] == "sat"
                and record["solve_seconds"] is not None
                and record["solve_seconds"] > 0
            ]
            if successful:
                plotted = True
                positive = True
                axis.scatter(
                    [
                        record["input_state_count"] + offsets[key]
                        for record in successful
                    ],
                    [record["solve_seconds"] for record in successful],
                    color=COLORS[optimization, symmetry_breaking],
                    marker=MARKERS[symmetry_breaking],
                    alpha=0.5,
                    label=f"{label} runs",
                )
                mean_points = []
                for state_count in state_counts:
                    state_runs = [
                        record
                        for record in successful
                        if record["input_state_count"] == state_count
                    ]
                    if state_runs:
                        mean_points.append(
                            (
                                state_count + offsets[key],
                                statistics.fmean(
                                    record["solve_seconds"] for record in state_runs
                                ),
                            )
                        )
                axis.plot(
                    [point[0] for point in mean_points],
                    [point[1] for point in mean_points],
                    color=COLORS[optimization, symmetry_breaking],
                    linestyle=LINESTYLES[optimization],
                    linewidth=2,
                    marker=MARKERS[symmetry_breaking],
                    markersize=5,
                    label=f"{label} mean",
                )
            timed_out = [
                record
                for record in group
                if record["status"] == "timeout"
                and record["wall_seconds"] is not None
                and record["wall_seconds"] > 0
            ]
            if timed_out:
                plotted = True
                positive = True
                axis.scatter(
                    [
                        record["input_state_count"] + offsets[key]
                        for record in timed_out
                    ],
                    [record["wall_seconds"] for record in timed_out],
                    color=COLORS[optimization, symmetry_breaking],
                    marker="x",
                    s=70,
                    label=f"{label} timeout",
                )
    if not plotted:
        axis.text(
            0.5,
            0.5,
            "No completed runs yet",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#64748b",
        )
    axis.set_xlabel("Actual input DFA states")
    axis.set_ylabel("Solve time (seconds, log scale)")
    axis.set_xticks(state_counts)
    axis.set_xlim(min(state_counts) - 0.5, max(state_counts) + 0.5)
    if positive:
        axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    if plotted:
        axis.legend(frameon=False, ncol=2, fontsize=8.5)
    fig.savefig(report_directory / "solve_time_individual.png", dpi=180)
    plt.close(fig)


def _blocked_models_chart(
    rows: list[dict[str, Any]],
    state_counts: list[int],
    report_directory: Path,
    symmetry_modes: Sequence[str] = ("bfs",),
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    plotted = False
    for optimization, optimization_label in OPTIMIZATIONS.items():
        for symmetry_breaking in symmetry_modes:
            blocking = sorted(
                (
                    row
                    for row in rows
                    if row["optimization_backend"] == optimization
                    and row["strategy"] == "blocking"
                    and row["symmetry_breaking"] == symmetry_breaking
                    and row["mean_blocked_models"] is not None
                ),
                key=lambda row: row["input_state_count"],
            )
            if not blocking:
                continue
            plotted = True
            axis.plot(
                [row["input_state_count"] for row in blocking],
                [row["mean_blocked_models"] for row in blocking],
                color=COLORS[optimization, symmetry_breaking],
                marker=MARKERS[symmetry_breaking],
                linestyle=LINESTYLES[optimization],
                linewidth=2.2,
                markersize=6,
                label=(
                    f"{optimization_label} / Manual blocking / "
                    f"{SYMMETRY_MODES[symmetry_breaking]}"
                ),
            )
    if not plotted:
        axis.text(
            0.5,
            0.5,
            "No completed manual-blocking runs yet",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#64748b",
        )
    axis.set_xlabel("Actual input DFA states")
    axis.set_ylabel("Mean blocked non-strict models (symlog scale)")
    axis.set_xticks(state_counts)
    axis.set_xlim(min(state_counts) - 0.5, max(state_counts) + 0.5)
    axis.set_yscale("symlog", linthresh=1)
    axis.grid(True, which="both", alpha=0.25)
    if plotted:
        axis.legend(frameon=False)
    fig.savefig(report_directory / "mean_blocked_models.png", dpi=180)
    plt.close(fig)


def _coverage_chart(
    rows: list[dict[str, Any]],
    state_counts: list[int],
    report_directory: Path,
    symmetry_modes: Sequence[str],
) -> None:
    series = [
        (optimization, strategy, symmetry_breaking)
        for optimization in OPTIMIZATIONS
        for strategy in STRATEGIES
        for symmetry_breaking in symmetry_modes
    ]
    fig, axes = plt.subplots(
        1,
        len(series),
        figsize=(max(7, 5.5 * len(series)), 5.4),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    status_fields = (
        ("sat_cases", "SAT", "#16a34a"),
        ("timeout_cases", "Timeout", "#f59e0b"),
        ("unsat_cases", "UNSAT", "#7c3aed"),
        ("error_cases", "Error", "#ef4444"),
        ("not_run_cases", "Not run", "#cbd5e1"),
    )
    for axis, (optimization, strategy, symmetry_breaking) in zip(
        axes.flat, series
    ):
        selected = sorted(
            (
                row
                for row in rows
                if row["optimization_backend"] == optimization
                and row["strategy"] == strategy
                and row["symmetry_breaking"] == symmetry_breaking
            ),
            key=lambda row: row["input_state_count"],
        )
        states = [row["input_state_count"] for row in selected]
        bottoms = [0] * len(selected)
        for field, label, color in status_fields:
            values = [row[field] for row in selected]
            axis.bar(states, values, bottom=bottoms, color=color, label=label)
            bottoms = [old + value for old, value in zip(bottoms, values)]
        axis.set_ylabel("Cases")
        axis.set_title(
            f"{OPTIMIZATIONS[optimization]} / {STRATEGIES[strategy]} / "
            f"{SYMMETRY_MODES[symmetry_breaking]}",
            loc="left",
            fontsize=11,
        )
        maximum_cases = max(
            (row["selected_cases"] for row in rows),
            default=5,
        )
        step = max(1, (maximum_cases + 4) // 5)
        axis.set_yticks(range(0, maximum_cases + step, step))
        axis.grid(True, axis="y", alpha=0.25)
    for axis in axes.flat:
        axis.set_xlabel("Actual input DFA states")
        axis.set_xticks(state_counts)
    axes.flat[0].legend(frameon=False, ncol=3, fontsize=8.5, loc="upper right")
    fig.savefig(report_directory / "benchmark_coverage.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build CSV/JSON summaries and charts from saved minimal-five runs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--report-directory",
        type=Path,
        help="default: <output-root>/reports",
    )
    parser.add_argument("--min-states", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=15)
    parser.add_argument(
        "--symmetry-modes",
        nargs="+",
        choices=tuple(SYMMETRY_MODES),
        default=("bfs",),
        help="saved modes to report (default: bfs); 'none' reads historical runs only",
    )
    return parser


def build_report(
    cases: list[BenchmarkCase],
    output_root: Path,
    report_directory: Path,
    manifest_path: Path,
    dataset_name: str = DATASET_NAME,
    symmetry_modes: Sequence[str] = ("bfs",),
) -> Counter[str]:
    """Write the shared detailed tables and charts for a benchmark dataset."""

    records = _load_records(cases, output_root, dataset_name, symmetry_modes)
    records.sort(
        key=lambda record: (
            record["input_state_count"],
            record["case"],
            record["optimization_backend"],
            record["strategy"],
            record["symmetry_breaking"],
        )
    )
    rows = _summaries(records, cases, symmetry_modes)
    state_counts = sorted({case.input_state_count for case in cases})

    report_directory.mkdir(parents=True, exist_ok=True)
    _write_detailed_csv(records, report_directory)
    _write_summary(rows, records, report_directory, manifest_path)
    _runtime_chart(records, state_counts, report_directory, symmetry_modes)
    _coverage_chart(rows, state_counts, report_directory, symmetry_modes)
    _line_chart(
        rows,
        state_counts,
        report_directory,
        "mean_solve_seconds",
        "mean_solve_time.png",
        "Mean solve time over SAT runs (seconds, log scale)",
        scale="log",
        symmetry_modes=symmetry_modes,
    )
    _line_chart(
        rows,
        state_counts,
        report_directory,
        "mean_wall_seconds",
        "mean_wall_time.png",
        "Mean end-to-end time over SAT runs (seconds, log scale)",
        scale="log",
        symmetry_modes=symmetry_modes,
    )
    _line_chart(
        rows,
        state_counts,
        report_directory,
        "mean_edit_distance",
        "mean_edit_distance.png",
        "Mean minimum edit distance over SAT runs",
        symmetry_modes=symmetry_modes,
    )
    _line_chart(
        rows,
        state_counts,
        report_directory,
        "mean_solver_calls",
        "mean_solver_calls.png",
        "Mean solver calls over SAT runs (symlog scale)",
        scale="symlog",
        symmetry_modes=symmetry_modes,
    )
    status_counts = Counter(record["status"] for record in records)
    status_text = ", ".join(
        f"{status}={count}" for status, count in sorted(status_counts.items())
    )
    print(
        f"wrote {len(records)} planned-run rows to {report_directory}",
        flush=True,
    )
    print(f"status: {status_text}", flush=True)
    return status_counts


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    report_directory = (
        args.report_directory.resolve()
        if args.report_directory
        else output_root / "reports"
    )
    cases = load_cases(
        manifest_path,
        min_states=args.min_states,
        max_states=args.max_states,
    )
    build_report(
        cases,
        output_root,
        report_directory,
        manifest_path,
        DATASET_NAME,
        args.symmetry_modes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
