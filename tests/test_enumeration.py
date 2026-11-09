import contextlib
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from itertools import product
from pathlib import Path
from unittest.mock import patch

from dfa import DFA
from pysat.solvers import Solver

from dfa_identify.cli import main as cli_main
from dfa_identify.minimum_edit import (
    _FixedRepairEncoding,
    _has_nonempty_language,
    _normalize_target,
    _repairs_at_distance,
    encoding_revision,
    find_minimum_edit_decomposition,
)
from test_minimum_edit import (
    assert_exact_and_strict,
    brute_force_unary_optimum,
    even_a_and_b_mod_3,
    even_a_and_even_b,
    trailing_b_mod_3,
    unary_two_state_dfa,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_all_small_dfaminer as all_small
from build_minimal_five_benchmark_report import _load_records, _write_detailed_csv
from run_minimal_five_benchmark import (
    BenchmarkCase, ENCODING_REVISION, SEARCH_ORDERS, _read_matching_result,
    result_directory,
)


class EnumerationTests(unittest.TestCase):
    def test_repair_layers_match_all_transition_tables_without_duplicates(self):
        for state_count, alphabet_size in ((1, 1), (2, 2), (3, 1)):
            with self.subTest(states=state_count, alphabet=alphabet_size):
                target = DFA(
                    start=0, inputs=set(range(alphabet_size)), outputs={False, True},
                    states=tuple(range(state_count)), label=lambda q: q == 0,
                    transition=lambda q, _symbol: q)
                original = _normalize_target(target)
                seen = set()
                layer_counts = Counter()
                for distance in range(state_count * (alphabet_size + 1) + 1):
                    for repaired in _repairs_at_distance(original, distance):
                        key = (repaired.transitions, repaired.accepting)
                        self.assertNotIn(key, seen)
                        seen.add(key)
                        self.assertEqual(repaired.original_states, original.original_states)
                        actual = sum(a != b for x, y in zip(repaired.transitions, original.transitions)
                                     for a, b in zip(x, y))
                        actual += sum(a != b for a, b in zip(repaired.accepting, original.accepting))
                        self.assertEqual(actual, distance)
                        layer_counts[distance] += 1
                expected = {
                    (tuple(tuple(entries[q * alphabet_size:(q + 1) * alphabet_size])
                           for q in range(state_count)), accepting)
                    for entries in product(range(state_count), repeat=state_count * alphabet_size)
                    for accepting in product((False, True), repeat=state_count)
                }
                self.assertEqual(seen, expected)
                for distance in range(state_count * (alphabet_size + 1) + 1):
                    count = sum(
                        math.comb(state_count * alphabet_size, transition_edits)
                        * (state_count - 1) ** transition_edits
                        * math.comb(state_count, distance - transition_edits)
                        for transition_edits in range(state_count * alphabet_size + 1)
                        if 0 <= distance - transition_edits <= state_count
                    )
                    self.assertEqual(layer_counts[distance], count)

    def test_fixed_repair_formula_pins_every_cell_and_has_no_edit_variables(self):
        class AutomataOnly(_FixedRepairEncoding):
            def _build(self):
                self._encode_automata()

        original = _normalize_target(trailing_b_mod_3())
        encoding = AutomataOnly(original, (2, 2), True, "finite")
        with Solver(name="g3", bootstrap_with=encoding.cnf.clauses) as solver:
            for repaired in _repairs_at_distance(original, 1):
                assumptions = encoding.repair_assumptions(repaired)
                self.assertEqual(len(assumptions), 9)
                self.assertTrue(solver.solve(assumptions=assumptions))
                decoded, _factors = encoding.decode(solver.get_model())
                for state, row in enumerate(repaired.transitions):
                    self.assertEqual(bool(decoded._label(state)), repaired.accepting[state])
                    for symbol_index, destination in enumerate(row):
                        self.assertEqual(decoded._transition(state, original.alphabet[symbol_index]), destination)
        full = _FixedRepairEncoding(original, (2, 2), True, "finite")
        self.assertEqual(full.edit_literals, [])
        self.assertFalse(any(key[0].startswith("E-") for key in full.pool.obj2id))
        self.assertFalse(any(key[0].startswith("nonempty-") for key in full.pool.obj2id))

    def test_enumeration_uses_no_sat_edit_counter_or_maxsat_and_checks_semantics(self):
        with patch("dfa_identify.minimum_edit.CardEnc.atmost", side_effect=AssertionError("SAT edit counter used")):
            for converse in (True, False):
                for strictness in (True, False):
                    with self.subTest(converse=converse, strictness=strictness):
                        result = find_minimum_edit_decomposition(
                            trailing_b_mod_3(), (2, 2), optimization="enumeration",
                            bfs_parent_redundancy=converse, strictness=strictness,
                            maxsat_factory=lambda _formula: self.fail("MaxSAT invoked"))
                        self.assertEqual(result.edit_distance, 1)
                        self.assertEqual(result.optimization_backend, "enumeration")
                        self.assertEqual(result.costs_seen, (0, 1))
                        self.assertEqual(sum(a.repair_candidates_checked for a in result.factor_size_attempts),
                                         result.repair_candidates_checked)
                        self.assertGreater(result.repair_candidates_checked, 1)
                        self.assertGreaterEqual(
                            result.solver_calls,
                            result.repair_candidates_checked - result.empty_repair_candidates_skipped)
                        assert_exact_and_strict(result)
            infinite = find_minimum_edit_decomposition(
                even_a_and_even_b(), (2, 2), optimization="enumeration", strictness_mode="infinite")
            self.assertEqual(infinite.edit_distance, 0)
            self.assertEqual(infinite.repair_candidates_checked, 1)
            assert_exact_and_strict(infinite)

    def test_automatic_search_prefers_zero_edit_later_pair_over_earlier_positive_cost(self):
        for ordering in ("sum", "max", "product"):
            result = find_minimum_edit_decomposition(
                even_a_and_b_mod_3(), optimization="enumeration", factor_order=ordering)
            self.assertEqual(result.edit_distance, 0)
            self.assertEqual(tuple(len(f.states()) for f in result.factors), (2, 3))
            self.assertEqual(tuple(a.factor_sizes for a in result.factor_size_attempts), ((2, 2), (2, 3)))
            self.assertEqual(result.repair_candidates_checked, 2)
            self.assertEqual(result.solver_calls, 2)
            self.assertTrue(result.stopped_on_zero)

    def test_enumeration_matches_independent_exhaustive_unary_optima(self):
        for transitions in product(range(2), repeat=2):
            for accepting_mask in range(4):
                with self.subTest(transitions=transitions, mask=accepting_mask):
                    target = unary_two_state_dfa(transitions, accepting_mask)
                    expected = brute_force_unary_optimum(target)
                    result = find_minimum_edit_decomposition(target, (2, 2), optimization="enumeration")
                    self.assertEqual(None if result is None else result.edit_distance, expected)

    def test_multiple_edit_layers_preserve_declared_unreachable_states(self):
        target = DFA(start=0, inputs={"a", "b"}, outputs={False, True},
                     states=(0, 1, 2), label=lambda _q: False,
                     transition=lambda q, _symbol: q)
        baseline = find_minimum_edit_decomposition(target, (2, 2), optimization="sat")
        enumeration = find_minimum_edit_decomposition(target, (2, 2), optimization="enumeration")
        self.assertEqual(enumeration.edit_distance, baseline.edit_distance)
        self.assertEqual(enumeration.edit_distance, 3)
        self.assertEqual(enumeration.costs_seen, (0, 1, 2, 3))
        self.assertEqual(len(enumeration.repaired.states()), 3)
        self.assertEqual(tuple(a.repair_candidates_checked for a in enumeration.factor_size_attempts)[:3], (1, 15, 99))
        assert_exact_and_strict(enumeration)

    def test_cli_enumeration_json_settings_and_search_statistics(self):
        target = trailing_b_mod_3()
        payload = {
            "states": list(target.states()), "alphabet": ["a", "b"], "start": target.start,
            "accepting": [q for q in target.states() if target._label(q)],
            "transitions": [{"from": q, "symbol": symbol, "to": target._transition(q, symbol)}
                            for q in target.states() for symbol in ("a", "b")],
            "optimization": "enumeration", "factor_order": "max", "bfs_parent_redundancy": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.json", Path(directory) / "result.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            with patch("dfa_identify.cli.write_decomposition_visualisations", return_value={}):
                self.assertEqual(cli_main([str(source), "-o", str(output)]), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["edit_distance"], 1)
            self.assertEqual(result["optimization_backend"], "enumeration")
            self.assertEqual(result["factor_size_search"]["search_order"], SEARCH_ORDERS["enumeration"])
            self.assertEqual(result["sat_solver"], "g3")
            self.assertEqual(
                result["search"]["repair_candidates_checked"],
                result["search"]["solver_calls"] + result["search"]["empty_repair_candidates_skipped"])
            self.assertEqual(result["encoding_revision"], encoding_revision("enumeration"))
            self.assertFalse(result["bfs_parent_redundancy"])

    def test_all_eligible_runner_routes_enumeration_and_isolates_results(self):
        case = all_small.InputCase("minimal", BenchmarkCase("sample", Path("sample.dot"), 3, 3, 1, None, None))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(all_small, "scan_cases", return_value=[case]):
                with patch.object(all_small, "_run_one", return_value={"status": "unsat"}) as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(all_small.main([
                            "--optimization", "enumeration", "--min-states", "3", "--max-states", "7",
                            "--factor-order", "max", "--bfs-parent-redundancy", "--output-root", directory]), 0)
            self.assertEqual(run.call_args.args[3:7], ("enumeration", "bfs", "finite", "max"))
            self.assertEqual(run.call_args.args[1], Path(directory) / "enumeration/runs/states_03/minimal/sample")
            self.assertTrue(run.call_args.kwargs["bfs_parent_redundancy"])

    def test_resume_and_report_include_enumeration_backend(self):
        case = BenchmarkCase("sample", Path("sample.dot"), 3, 3, 1, None, None)
        payload = {"status": "unsat", "benchmark": {
            "dataset": "dfaminer-minimal-five-per-actual-size", "case_id": "sample",
            "input_state_count": 3, "strategy": "witness", "symmetry_breaking": "bfs",
            "optimization_backend": "enumeration", "encoding_revision": encoding_revision("enumeration"),
            "factor_size_search_mode": "all-proper-unordered",
            "factor_size_search_order": SEARCH_ORDERS["enumeration"],
        }}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = result_directory(root, case, "enumeration", "witness") / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(_read_matching_result(path, case, "enumeration", "witness", require_current_encoding=True))
            self.assertIsNone(_read_matching_result(path, case, "sat", "witness", require_current_encoding=True))
            payload["benchmark"]["encoding_revision"] = ENCODING_REVISION
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(_read_matching_result(path, case, "enumeration", "witness", require_current_encoding=True))
            self.assertIsNotNone(_read_matching_result(path, case, "enumeration", "witness"))
            payload["benchmark"]["encoding_revision"] = encoding_revision("enumeration")
            path.write_text(json.dumps(payload), encoding="utf-8")
            records = _load_records([case], root)
            selected = next(row for row in records if row["optimization_backend"] == "enumeration")
            self.assertEqual(selected["status"], "unsat")
            _write_detailed_csv(records, root)
            with (root / "benchmark_results.csv").open(newline="", encoding="utf-8") as stream:
                self.assertTrue(any(row["optimization_backend"] == "enumeration" for row in csv.DictReader(stream)))

    def test_empty_alphabet_empty_size_domain_and_single_state_factors(self):
        empty_alphabet = DFA(start=0, inputs=set(), outputs={False, True},
                             states=(0, 1, 2), label=lambda _q: True,
                             transition=lambda q, _symbol: q)
        self.assertIsNone(find_minimum_edit_decomposition(empty_alphabet, optimization="enumeration"))
        self.assertIsNone(find_minimum_edit_decomposition(unary_two_state_dfa((1, 0), 1), optimization="enumeration"))
        self.assertIsNone(find_minimum_edit_decomposition(trailing_b_mod_3(), (1, 2), optimization="enumeration"))

    def test_nonemptiness_uses_reachability_and_handles_the_empty_word(self):
        def normalized(accepting, transitions, alphabet=("a",)):
            target = DFA(start=0, inputs=alphabet, outputs={False, True},
                         states=(0, 1, 2), label=lambda q: q in accepting,
                         transition=lambda q, _a: transitions[q])
            return _normalize_target(target)

        self.assertFalse(_has_nonempty_language(normalized(set(), (1, 2, 2))))
        self.assertFalse(_has_nonempty_language(normalized({2}, (1, 1, 2))))
        self.assertTrue(_has_nonempty_language(normalized({2}, (1, 2, 2))))
        self.assertTrue(_has_nonempty_language(normalized({0}, (0, 1, 2), alphabet=())))
        self.assertFalse(_has_nonempty_language(normalized({1, 2}, (0, 1, 2), alphabet=())))

    def test_empty_candidates_are_skipped_before_sat(self):
        target = DFA(start=0, inputs={"a", "b"}, outputs={False, True},
                     states=(0, 1, 2), label=lambda _q: False,
                     transition=lambda q, _symbol: q)
        normalized = _normalize_target(target)
        encoding = _FixedRepairEncoding(normalized, (2, 2), True, "finite")
        # Without the precheck, the remaining CNF permits disjoint strict
        # factors whose intersection equals this empty repair.
        with Solver(name="g3", bootstrap_with=encoding.cnf.clauses) as solver:
            self.assertTrue(solver.solve(assumptions=encoding.repair_assumptions(normalized)))

        original = _FixedRepairEncoding.repair_assumptions
        passed_to_sat = []

        def checked_assumptions(instance, repaired):
            self.assertTrue(_has_nonempty_language(repaired))
            passed_to_sat.append(repaired)
            return original(instance, repaired)

        with patch.object(_FixedRepairEncoding, "repair_assumptions", checked_assumptions):
            result = find_minimum_edit_decomposition(target, (2, 2), optimization="enumeration")
        self.assertEqual(result.edit_distance, 3)
        self.assertGreater(result.empty_repair_candidates_skipped, 0)
        self.assertEqual(result.factor_size_attempts[0].empty_repair_candidates_skipped, 1)
        self.assertEqual(result.empty_repair_candidates_skipped,
                         sum(a.empty_repair_candidates_skipped for a in result.factor_size_attempts))
        self.assertEqual(result.solver_calls, len(passed_to_sat))
        self.assertEqual(result.repair_candidates_checked,
                         result.solver_calls + result.empty_repair_candidates_skipped)
        assert_exact_and_strict(result)
