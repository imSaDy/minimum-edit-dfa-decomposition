import contextlib
import io
import json
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

from dfa import DFA
from pysat.solvers import Solver

from dfa_identify.cli import build_parser, main as cli_main
from dfa_identify.minimum_edit import (
    _DirectEncoding, _normalize_target, encoding_revision,
    find_minimum_edit_decomposition,
)
from test_minimum_edit import (
    assert_exact_and_strict, brute_force_unary_optimum,
    even_a_and_b_mod_3, even_a_and_even_b, trailing_b_mod_3,
    unary_two_state_dfa,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_all_small_dfaminer as all_small
import run_minimal_five_benchmark as minimal


class ProductOnly(_DirectEncoding):
    def _build(self):
        self._encode_automata()
        self._encode_exact_product()


def intersection_equals(repaired, first, second):
    """Independent traversal of the synchronized automata, with no SAT variables."""
    pending = [(repaired.start, first.start, second.start)]
    visited = set()
    while pending:
        q, i, j = pending.pop()
        if (q, i, j) in visited:
            continue
        visited.add((q, i, j))
        if bool(repaired._label(q)) != (bool(first._label(i)) and bool(second._label(j))):
            return False
        for symbol in repaired.inputs:
            pending.append((repaired._transition(q, symbol),
                            first._transition(i, symbol), second._transition(j, symbol)))
    return True


class ProductEncodingTests(unittest.TestCase):
    def test_product_clause_counts_and_auxiliary_variable_difference(self):
        target = _normalize_target(even_a_and_b_mod_3())
        formulas = {}
        for mode in ("factored", "direct"):
            encoding = ProductOnly(target, (4, 4), True, "finite", product_encoding=mode)
            formulas[mode] = encoding
            relation_literals = {encoding.product_relation(q, i, j)
                                 for q in range(6) for i in range(4) for j in range(4)}
            families = {key[0] for key in encoding.pool.obj2id}
            if mode == "direct":
                self.assertTrue({"U", "W"}.isdisjoint(families))
                closure = [clause for clause in encoding.cnf.clauses if len(clause) == 5]
                self.assertEqual(len(closure), 18432)
                self.assertTrue(all(clause[0] < 0 and -clause[0] in relation_literals
                                    and clause[-1] in relation_literals for clause in closure))
            else:
                auxiliary_literals = {value for key, value in encoding.pool.obj2id.items()
                                      if key[0] in ("U", "W")}
                closure = [clause for clause in encoding.cnf.clauses
                           if any(abs(lit) in auxiliary_literals for lit in clause)]
                self.assertEqual(len(closure), 2688)
                self.assertTrue(all(len(clause) == 3 for clause in closure))
                self.assertEqual(len(auxiliary_literals), 384)
        self.assertEqual(formulas["factored"].base_encoding_variables
                         - formulas["direct"].base_encoding_variables, 384)
        self.assertEqual(formulas["direct"].base_encoding_clauses
                         - formulas["factored"].base_encoding_clauses, 18432 - 2688)

    def test_both_representations_match_exhaustive_language_equality(self):
        automata = [unary_two_state_dfa(edges, mask)
                    for edges in product(range(2), repeat=2) for mask in range(4)]
        factors = [dfa for dfa in automata if dfa._transition(0, "a") == 1]
        for mode in ("factored", "direct"):
            encoding = ProductOnly(_normalize_target(automata[0]), (2, 2),
                                   True, "finite", product_encoding=mode)
            with Solver(name="g3", bootstrap_with=encoding.cnf.clauses) as solver:
                for repaired, first, second in product(automata, factors, factors):
                    assumptions = []
                    for q in range(2):
                        assumptions.append(encoding.repaired_transition(q, 0, repaired._transition(q, "a")))
                        literal = encoding.repaired_accepting(q)
                        assumptions.append(literal if repaired._label(q) else -literal)
                    for h, factor in enumerate((first, second)):
                        for q in range(2):
                            assumptions.append(encoding.factor_transition(h, q, 0, factor._transition(q, "a")))
                            literal = encoding.factor_accepting(h, q)
                            assumptions.append(literal if factor._label(q) else -literal)
                    self.assertEqual(solver.solve(assumptions=assumptions),
                                     intersection_equals(repaired, first, second),
                                     (mode, repaired, first, second))

    def test_direct_mode_matches_independent_minimum_edit_optima(self):
        for edges in product(range(2), repeat=2):
            for mask in range(4):
                target = unary_two_state_dfa(edges, mask)
                expected = brute_force_unary_optimum(target)
                for method in ("sat", "maxsat", "enumeration"):
                    result = find_minimum_edit_decomposition(
                        target, (2, 2), optimization=method, product_encoding="direct")
                    self.assertEqual(None if result is None else result.edit_distance, expected)

    def test_modes_preserve_strictness_search_order_and_parent_settings(self):
        for method, mode, strengthened in product(
                ("sat", "maxsat", "enumeration"), ("factored", "direct"), (True, False)):
            result = find_minimum_edit_decomposition(
                trailing_b_mod_3(), optimization=method, product_encoding=mode,
                bfs_parent_redundancy=strengthened)
            self.assertEqual(result.edit_distance, 1)
            self.assertEqual(result.product_encoding, mode)
            self.assertEqual(result.bfs_parent_redundancy, strengthened)
            self.assertGreater(result.base_encoding_clauses, 0)
            assert_exact_and_strict(result)
        for method, mode in product(("sat", "maxsat", "enumeration"), ("factored", "direct")):
            infinite = find_minimum_edit_decomposition(
                even_a_and_even_b(), (2, 2), optimization=method,
                product_encoding=mode, strictness_mode="infinite")
            self.assertEqual(infinite.edit_distance, 0)
            assert_exact_and_strict(infinite)
            automatic = find_minimum_edit_decomposition(
                even_a_and_b_mod_3(), optimization=method, product_encoding=mode)
            self.assertEqual(automatic.edit_distance, 0)
            self.assertEqual(tuple(len(f.states()) for f in automatic.factors), (2, 3))

    def test_renamed_options_and_legacy_aliases(self):
        parsed = build_parser().parse_args([
            "sample.dot", "--optimization", "hybrid", "--no-bfs-converse",
            "--product-encoding", "direct"])
        self.assertEqual(parsed.optimization, "enumeration")
        self.assertFalse(parsed.bfs_parent_redundancy)
        result = find_minimum_edit_decomposition(
            trailing_b_mod_3(), (2, 2), optimization="hybrid", bfs_converse=False)
        self.assertEqual(result.optimization_backend, "enumeration")
        self.assertFalse(result.bfs_parent_redundancy)
        self.assertFalse(result.bfs_converse)
        self.assertFalse(result.bfs_parent_strengthening)
        intermediate = build_parser().parse_args(["sample.dot", "--no-bfs-parent-strengthening"])
        self.assertFalse(intermediate.bfs_parent_redundancy)
        aliased = find_minimum_edit_decomposition(
            trailing_b_mod_3(), (2, 2), bfs_parent_strengthening=False)
        self.assertFalse(aliased.bfs_parent_redundancy)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            find_minimum_edit_decomposition(
                trailing_b_mod_3(), bfs_converse=True, bfs_parent_redundancy=False)
        with self.assertRaisesRegex(ValueError, "product_encoding"):
            find_minimum_edit_decomposition(trailing_b_mod_3(), product_encoding="invalid")

    def test_json_selection_cli_override_and_statistics(self):
        target = trailing_b_mod_3()
        payload = {
            "states": list(target.states()), "alphabet": ["a", "b"], "start": 0,
            "accepting": [2], "factor_sizes": [2, 2],
            "transitions": [{"from": q, "symbol": a, "to": target._transition(q, a)}
                            for q in target.states() for a in ("a", "b")],
            "optimization": "enumeration", "product_encoding": "direct",
            "bfs_parent_redundancy": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.json", Path(directory) / "result.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            for flags, expected in (([], "direct"), (["--product-encoding", "factored"], "factored")):
                with patch("dfa_identify.cli.write_decomposition_visualisations", return_value={}):
                    self.assertEqual(cli_main([str(source), "-o", str(output), *flags]), 0)
                record = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(record["product_encoding"], expected)
                self.assertEqual(record["encoding_revision"], encoding_revision("enumeration", expected))
                self.assertGreater(record["encoding_statistics"]["clauses"], 0)
                self.assertNotIn("bfs_converse", record)

    def test_runner_passes_mode_isolates_output_and_rejects_mismatched_resume(self):
        case = all_small.InputCase("minimal", minimal.BenchmarkCase(
            "sample", Path("sample.dot"), 3, 3, 1, None, None))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(all_small, "scan_cases", return_value=[case]):
                with patch.object(all_small, "_run_one", return_value={"status": "unsat"}) as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(all_small.main([
                            "--min-states", "3", "--max-states", "3", "--factor-order", "max",
                            "--output-root", directory, "--product-encoding", "direct"]), 0)
            self.assertEqual(run.call_args.kwargs["product_encoding"], "direct")
            self.assertEqual(run.call_args.args[1], Path(directory) / "product-direct/runs/states_03/minimal/sample")
            record = {"status": "unsat", "benchmark": {
                "dataset": all_small.DATASET_NAME, "case_id": "sample", "input_state_count": 3,
                "input_kind": "minimal", "input_dot_sha256": None, "optimization_backend": "sat",
                "strictness_mode": "finite", "strategy": "witness", "symmetry_breaking": "bfs",
                "factor_order": "max", "timeout_seconds": 30,
                "nonempty_requirement": all_small.NONEMPTY_REQUIREMENT,
                "factor_size_search_mode": minimal.FACTOR_SIZE_SEARCH_MODE,
                "factor_size_search_order": minimal.SEARCH_ORDERS["sat"],
                "product_encoding": "direct", "encoding_revision": encoding_revision("sat", "direct"),
            }}
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNotNone(all_small.matching_result(path, case, 30, "max", product_encoding="direct"))
            self.assertIsNone(all_small.matching_result(path, case, 30, "max", product_encoding="factored"))
            record["benchmark"]["product_encoding"] = "factored"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNone(all_small.matching_result(path, case, 30, "max", product_encoding="direct"))
