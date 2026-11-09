"""Command-line interface for minimum-edit DFA decomposition."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from dfa import DFA
from pysat.solvers import NoSuchSolverError

from dfa_identify.dot_io import (
    dfa_from_dot,
    write_decomposition_visualisations,
)
from dfa_identify.minimum_edit import (
    ENCODING_REVISION,
    encoding_revision,
    DecompositionResult,
    factor_size_pairs,
    find_minimum_edit_decomposition,
    normalize_optimization,
    _resolve_bfs_parent_redundancy,
)


def _lookup_declared(value: Any, declared: Sequence[Any], field: str) -> Any:
    for candidate in declared:
        if value == candidate:
            return candidate
    raise ValueError(f"unknown {field}: {value!r}")


def _dfa_from_json(payload: dict[str, Any]) -> DFA:
    states = tuple(payload["states"])
    alphabet = tuple(payload["alphabet"])
    if not states:
        raise ValueError("states must not be empty")
    if len(set(map(repr, states))) != len(states):
        raise ValueError("states must be unique")
    if len(set(map(repr, alphabet))) != len(alphabet):
        raise ValueError("alphabet symbols must be unique")

    start = _lookup_declared(payload["start"], states, "start state")
    accepting = {
        _lookup_declared(state, states, "accepting state")
        for state in payload.get("accepting", [])
    }

    transition_map: dict[tuple[Any, Any], Any] = {}
    transition_data = payload["transitions"]
    if isinstance(transition_data, list):
        for item in transition_data:
            source = _lookup_declared(item["from"], states, "transition source")
            symbol = _lookup_declared(item["symbol"], alphabet, "transition symbol")
            target = _lookup_declared(item["to"], states, "transition target")
            key = (source, symbol)
            if key in transition_map:
                raise ValueError(f"duplicate transition for {key!r}")
            transition_map[key] = target
    elif isinstance(transition_data, dict):
        # Nested object syntax is convenient when all state/symbol names are
        # strings.  The list form above supports arbitrary JSON scalar values.
        for source_name, row in transition_data.items():
            source = _lookup_declared(source_name, states, "transition source")
            for symbol_name, target_name in row.items():
                symbol = _lookup_declared(symbol_name, alphabet, "transition symbol")
                target = _lookup_declared(target_name, states, "transition target")
                transition_map[source, symbol] = target
    else:
        raise ValueError("transitions must be a list or a nested object")

    missing = [
        (state, symbol)
        for state in states
        for symbol in alphabet
        if (state, symbol) not in transition_map
    ]
    if missing:
        raise ValueError(f"missing transitions: {missing!r}")

    return DFA(
        start=start,
        inputs=alphabet,
        outputs={False, True},
        states=states,
        label=lambda state: state in accepting,
        transition=lambda state, symbol: transition_map[state, symbol],
    )


def _dfa_to_json(dfa: DFA, alphabet: Sequence[Any]) -> dict[str, Any]:
    states = sorted(dfa.states(), key=lambda value: (type(value).__name__, repr(value)))
    return {
        "states": states,
        "alphabet": list(alphabet),
        "start": dfa.start,
        "accepting": [state for state in states if bool(dfa._label(state))],
        "transitions": [
            {
                "from": state,
                "symbol": symbol,
                "to": dfa._transition(state, symbol),
            }
            for state in states
            for symbol in alphabet
        ],
    }


def _result_to_json(result: DecompositionResult) -> dict[str, Any]:
    alphabet = sorted(
        result.repaired.inputs,
        key=lambda value: (type(value).__name__, repr(value)),
    )
    return {
        "status": "sat",
        "edit_distance": result.edit_distance,
        "factor_sizes": [len(tuple(factor.states())) for factor in result.factors],
        "strictness_strategy": result.strictness_strategy,
        "strictness_mode": result.strictness_mode,
        "optimization_backend": result.optimization_backend,
        "symmetry_breaking": result.symmetry_breaking,
        "bfs_parent_redundancy": result.bfs_parent_redundancy,
        "bfs_encoding": result.bfs_encoding,
        "product_encoding": result.product_encoding,
        "encoding_statistics": {
            "scope": "base hard CNF for the returned factor-size pair; excludes edit bounds, assumptions, lazy blocks, soft clauses and native solver transformations",
            "variables": result.base_encoding_variables,
            "clauses": result.base_encoding_clauses,
            "literals": result.base_encoding_literals,
            "max_clause_length": result.base_encoding_max_clause_length,
        },
        "transition_edits": [
            {
                "state": edit.state,
                "symbol": edit.symbol,
                "old_target": edit.old_target,
                "new_target": edit.new_target,
            }
            for edit in result.transition_edits
        ],
        "acceptance_edits": [
            {
                "state": edit.state,
                "old_value": edit.old_value,
                "new_value": edit.new_value,
            }
            for edit in result.acceptance_edits
        ],
        "strictness_witnesses": [list(word) for word in result.witnesses],
        "common_accepting_witness": (
            None if result.common_witness is None else list(result.common_witness)
        ),
        "search": {
            "costs_seen": list(result.costs_seen),
            "checked_models": result.checked_models,
            "blocked_models": result.blocked_models,
            "solver_calls": result.solver_calls,
            "repair_candidates_checked": result.repair_candidates_checked,
            "empty_repair_candidates_skipped": result.empty_repair_candidates_skipped,
            "factor_size_attempts": [
                {
                    "factor_sizes": list(attempt.factor_sizes),
                    "edit_distance": attempt.edit_distance,
                    "elapsed_seconds": round(attempt.elapsed_seconds, 6),
                    "edit_budget": attempt.edit_budget,
                    "repair_candidates_checked": attempt.repair_candidates_checked,
                    "empty_repair_candidates_skipped": attempt.empty_repair_candidates_skipped,
                }
                for attempt in result.factor_size_attempts
            ],
            "factor_size_search_complete": result.factor_size_search_complete,
            "stopped_on_zero": result.stopped_on_zero,
        },
        "repaired": _dfa_to_json(result.repaired, alphabet),
        "factors": [_dfa_to_json(factor, alphabet) for factor in result.factors],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minimum-edit-dfa",
        description="Find a minimum-edit exact strict decomposition A' = A1 intersect A2.",
    )
    parser.add_argument("input", type=Path, help="input DFA DOT file")
    parser.add_argument(
        "--factor-sizes",
        type=int,
        nargs=2,
        metavar=("N1", "N2"),
        help=(
            "solve one exact factor-size pair; when omitted, search every "
            "unique pair 2 <= N1 <= N2 < the input state count"
        ),
    )
    parser.add_argument(
        "--strictness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "encode bounded strictness witnesses in SAT; --no-strictness still "
            "enforces strictness by blocking decoded non-strict factor pairs"
        ),
    )
    parser.add_argument(
        "--strictness-mode",
        choices=("finite", "infinite"),
        default=None,
        help=(
            "strictness semantics: 'finite' requires one witness in each "
            "direction (default); 'infinite' requires both directional "
            "language differences to be infinite"
        ),
    )
    parser.add_argument(
        "--optimization",
        type=normalize_optimization,
        choices=("maxsat", "sat", "enumeration"),
        default=None,
        help=(
            "minimize edits with Exact MaxSAT (default) or ordinary SAT "
            "iterative deepening, or enumerate repairs externally and check "
            "each fixed repair with SAT ('enumeration')"
        ),
    )
    parser.add_argument(
        "--product-encoding",
        choices=("factored", "direct"),
        default=None,
        help="product-relation closure: 'factored' uses U/W auxiliaries (default); 'direct' uses five-literal clauses without U/W",
    )
    parser.add_argument(
        "--sat-solver",
        default=None,
        metavar="NAME",
        help=(
            "PySAT solver used by --optimization sat or enumeration, for example g3, "
            "cadical195, cadical300, or maplecm (default: g3)"
        ),
    )
    parser.add_argument(
        "--bfs-parent-redundancy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include logically redundant BFS parent clauses (default); --no-bfs-parent-redundancy omits them while retaining BFS",
    )
    parser.add_argument(
        "--bfs-encoding", choices=("classic", "compact"), default=None,
        help="BFS predicates: classic (default) or compact Section 3.2 prefix encoding",
    )
    parser.add_argument(
        "--bfs-converse", "--bfs-parent-strengthening", dest="bfs_parent_redundancy",
        action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--factor-order",
        choices=("sum", "max", "product"),
        default=None,
        help=(
            "automatic factor-pair order and secondary objective: 'sum' "
            "minimizes n1+n2 (default); 'max' first minimizes max(n1,n2); "
            "'product' first minimizes n1*n2"
        ),
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="print the factor-size pair before and after each subproblem",
    )
    parser.add_argument("-o", "--output", type=Path, help="write result JSON here")
    parser.add_argument(
        "--visualize-dir",
        type=Path,
        help=(
            "directory for A, A_prime, A1, and A2 DOT/PNG files; by default "
            "a test-specific directory is created next to the input"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.input.suffix.lower() == ".dot":
            payload: dict[str, Any] = {}
            target = dfa_from_dot(args.input)
        elif args.input.suffix.lower() == ".json":
            with args.input.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            target = _dfa_from_json(payload)
            removed = {
                "max_edits", "incremental_sat", "symmetry_breaking"
            } & payload.keys()
            if removed:
                raise ValueError("removed configuration options: " + ", ".join(sorted(removed)))
        else:
            raise ValueError("input must be a .dot file")
        factor_sizes_value = args.factor_sizes or payload.get("factor_sizes")
        factor_sizes: tuple[int, int] | None
        if factor_sizes_value is None:
            factor_sizes = None
        else:
            parsed_factor_sizes = tuple(factor_sizes_value)
            if len(parsed_factor_sizes) != 2:
                raise ValueError("factor_sizes must contain exactly two integers")
            factor_sizes = (parsed_factor_sizes[0], parsed_factor_sizes[1])

        configured_strictness = payload.get("strictness", True)
        if not isinstance(configured_strictness, bool):
            raise ValueError("strictness must be true or false")
        strictness = (
            args.strictness if args.strictness is not None else configured_strictness
        )
        configured_strictness_mode = payload.get("strictness_mode", "finite")
        if configured_strictness_mode not in ("finite", "infinite"):
            raise ValueError("strictness_mode must be 'finite' or 'infinite'")
        strictness_mode = args.strictness_mode or configured_strictness_mode
        configured_optimization = normalize_optimization(payload.get("optimization", "maxsat"))
        optimization = args.optimization or configured_optimization
        configured_product_encoding = payload.get("product_encoding", "factored")
        if configured_product_encoding not in ("factored", "direct"):
            raise ValueError("product_encoding must be 'factored' or 'direct'")
        product_encoding = args.product_encoding or configured_product_encoding
        configured_sat_solver = payload.get("sat_solver", "g3")
        if not isinstance(configured_sat_solver, str) or not configured_sat_solver:
            raise ValueError("sat_solver must be a non-empty PySAT solver name")
        sat_solver = args.sat_solver or configured_sat_solver
        for name in ("bfs_parent_redundancy", "bfs_parent_strengthening", "bfs_converse"):
            if name in payload and not isinstance(payload[name], bool):
                raise ValueError(f"{name} must be true or false")
        configured_bfs_parent_redundancy = _resolve_bfs_parent_redundancy(
            payload.get("bfs_parent_redundancy"), payload.get("bfs_converse"),
            payload.get("bfs_parent_strengthening"),
        )
        if not isinstance(configured_bfs_parent_redundancy, bool):
            raise ValueError("bfs_parent_redundancy must be true or false")
        bfs_parent_redundancy = (
            args.bfs_parent_redundancy
            if args.bfs_parent_redundancy is not None
            else configured_bfs_parent_redundancy
        )
        configured_factor_order = payload.get("factor_order", "sum")
        configured_bfs_encoding = payload.get("bfs_encoding", "classic")
        if configured_bfs_encoding not in ("classic", "compact"):
            raise ValueError("bfs_encoding must be 'classic' or 'compact'")
        bfs_encoding = args.bfs_encoding or configured_bfs_encoding
        if configured_factor_order not in ("sum", "max", "product"):
            raise ValueError("factor_order must be 'sum', 'max', or 'product'")
        factor_order = args.factor_order or configured_factor_order
        solve_started = time.perf_counter()
        result = find_minimum_edit_decomposition(
            target,
            factor_sizes,
            strictness=strictness,
            strictness_mode=strictness_mode,
            optimization=optimization,
            sat_solver=sat_solver,
            factor_order=factor_order,
            progress=(
                (lambda message: print(message, flush=True))
                if args.show_progress
                else None
            ),
            bfs_parent_redundancy=bfs_parent_redundancy,
            bfs_encoding=bfs_encoding,
            product_encoding=product_encoding,
        )
        solve_seconds = time.perf_counter() - solve_started
        output = (
            _result_to_json(result)
            if result is not None
            else {
                "status": "unsat",
                "strictness_strategy": "witness" if strictness else "blocking",
                "strictness_mode": strictness_mode,
                "factor_order": factor_order,
                "optimization_backend": optimization,
                "symmetry_breaking": "bfs",
            }
        )
        output["encoding_revision"] = encoding_revision(optimization, product_encoding, bfs_encoding)
        output["bfs_encoding"] = bfs_encoding
        output["product_encoding"] = product_encoding
        output["bfs_parent_redundancy"] = bfs_parent_redundancy
        output["sat_solver"] = sat_solver if optimization in ("sat", "enumeration") else None
        output["factor_order"] = factor_order
        if factor_sizes is None:
            candidate_pairs = factor_size_pairs(
                len(tuple(target.states())), ordering=factor_order
            )
            output["factor_size_search"] = {
                "mode": "all-proper-unordered",
                "candidates": [list(pair) for pair in candidate_pairs],
                "factor_order": factor_order,
                "completed": (
                    True if result is None else result.factor_size_search_complete
                ),
                "stopped_on_zero": (
                    False if result is None else result.stopped_on_zero
                ),
                "search_order": (
                    "edit-distance-then-factor-size-then-repair"
                    if optimization == "enumeration"
                    else "edit-budget-then-factor-size"
                    if optimization == "sat"
                    else "factor-size-then-optimum"
                ),
            }
        else:
            output["factor_size_search"] = {
                "mode": "fixed",
                "candidates": [list(factor_sizes)],
                "factor_order": factor_order,
                "completed": True,
                "stopped_on_zero": False,
            }
        output["timing"] = {"solve_seconds": round(solve_seconds, 6)}
        if result is not None:
            strategy = "witness" if strictness else "blocking"
            selected_factor_sizes = tuple(
                len(tuple(factor.states())) for factor in result.factors
            )
            default_visualise_directory = (
                args.input.parent
                / "visualizations"
                / args.input.stem
                / f"{selected_factor_sizes[0]}x{selected_factor_sizes[1]}"
                / strategy
            )
            if strictness_mode != "finite":
                default_visualise_directory /= f"strictness-{strictness_mode}"
            default_visualise_directory /= optimization
            if factor_order != "sum":
                default_visualise_directory /= f"factor-order-{factor_order}"
            if optimization in ("sat", "enumeration") and sat_solver != "g3":
                default_visualise_directory /= f"solver-{sat_solver}"
            default_visualise_directory /= "symmetry-bfs"
            if bfs_encoding != "classic":
                default_visualise_directory /= f"encoding-{bfs_encoding}"
            if not bfs_parent_redundancy:
                default_visualise_directory /= "without-bfs-parent-redundancy"
            if product_encoding != "factored":
                default_visualise_directory /= f"product-{product_encoding}"
            visualise_directory = (
                args.visualize_dir or default_visualise_directory
            )
            output["visualizations"] = write_decomposition_visualisations(
                target, result, visualise_directory
            )
    except (
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
        NoSuchSolverError,
    ) as error:
        raise SystemExit(f"invalid input: {error}") from error

    rendered = json.dumps(output, ensure_ascii=False, indent=2, default=repr) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
