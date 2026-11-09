import contextlib
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from dfa import DFA
from pysat.solvers import Solver

from dfa_identify.cli import main as cli_main
from dfa_identify.minimum_edit import (
    ENCODING_REVISION,
    _DirectEncoding,
    _normalize_target,
    find_minimum_edit_decomposition,
)
from test_minimum_edit import (
    assert_exact_and_strict,
    even_a_and_even_b,
    trailing_b_mod_3,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_all_small_dfaminer as all_small
import run_nonminimal_five_benchmark as nonminimal
from run_minimal_five_benchmark import (
    BenchmarkCase,
    DATASET_NAME,
    FACTOR_SIZE_SEARCH_MODE,
    SEARCH_ORDERS,
    _read_matching_result,
    result_directory,
)


class BfsParentRedundancyTests(unittest.TestCase):
    def test_only_converse_clauses_are_removed_and_are_entailed(self):
        class AutomataEncoding(_DirectEncoding):
            def _build(self):
                self._encode_automata()

        for alphabet_size in (1, 2, 3):
            target = DFA(start=0, inputs=set(range(alphabet_size)),
                         outputs={False, True}, label=lambda _q: False,
                         transition=lambda q, _a: q)
            for sizes in ((1, 1), (2, 3), (6, 7)):
                with self.subTest(alphabet_size=alphabet_size, sizes=sizes):
                    normalized = _normalize_target(target)
                    baseline = AutomataEncoding(normalized, sizes, True, "finite")
                    reduced = AutomataEncoding(normalized, sizes, True, "finite", False)
                    converse = []
                    for factor, size in enumerate(sizes):
                        for state in range(1, size):
                            # Accessibility still follows from retained parent existence.
                            self.assertIn(
                                [reduced.enumeration_parent(factor, parent, state)
                                 for parent in range(state)], reduced.cnf.clauses)
                            for parent in range(state):
                                converse.append([
                                    -reduced.transition_relation(factor, parent, state),
                                    reduced.enumeration_parent(factor, parent, state),
                                    *[reduced.transition_relation(factor, earlier, state)
                                      for earlier in range(parent)],
                                ])
                    self.assertEqual(baseline.pool.obj2id, reduced.pool.obj2id)
                    self.assertEqual(
                        Counter(map(tuple, baseline.cnf.clauses)),
                        Counter(map(tuple, reduced.cnf.clauses)) + Counter(map(tuple, converse)))
                    with Solver(name="g3", bootstrap_with=reduced.cnf.clauses) as solver:
                        self.assertTrue(solver.solve())
                        for clause in converse:
                            self.assertFalse(solver.solve(assumptions=[-literal for literal in clause]))

    def test_reduced_encoding_preserves_optima_and_semantics(self):
        for optimization in ("sat", "maxsat"):
            for mode, target, expected in (
                ("finite", trailing_b_mod_3(), 1),
                ("infinite", even_a_and_even_b(), 0),
            ):
                with self.subTest(optimization=optimization, mode=mode):
                    result = find_minimum_edit_decomposition(
                        target, (2, 2), optimization=optimization,
                        strictness_mode=mode, bfs_parent_redundancy=False)
                    self.assertIsNotNone(result)
                    self.assertEqual(result.edit_distance, expected)
                    self.assertEqual(result.symmetry_breaking, "bfs")
                    self.assertFalse(result.bfs_parent_redundancy)
                    assert_exact_and_strict(result)
        automatic = find_minimum_edit_decomposition(
            trailing_b_mod_3(), optimization="sat", bfs_parent_redundancy=False)
        blocking = find_minimum_edit_decomposition(
            trailing_b_mod_3(), (2, 2), optimization="sat",
            strictness=False, bfs_parent_redundancy=False)
        for result in (automatic, blocking):
            self.assertEqual(result.edit_distance, 1)
            self.assertFalse(result.bfs_parent_redundancy)
            assert_exact_and_strict(result)

    def test_cli_and_json_select_converse_without_disabling_bfs(self):
        payload = {"states": [0], "alphabet": ["a"], "start": 0,
                   "accepting": [0],
                   "transitions": [{"from": 0, "symbol": "a", "to": 0}]}
        configurations = (({}, [], True), ({"bfs_parent_redundancy": False}, [], False),
                          ({"bfs_parent_redundancy": False}, ["--bfs-parent-redundancy"], True),
                          ({"bfs_parent_redundancy": True}, ["--no-bfs-parent-redundancy"], False))
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.json", Path(directory) / "result.json"
            for settings, flags, expected in configurations:
                with self.subTest(settings=settings, flags=flags):
                    source.write_text(json.dumps({**payload, **settings}), encoding="utf-8")
                    with patch("dfa_identify.cli.find_minimum_edit_decomposition", return_value=None) as solve:
                        self.assertEqual(cli_main([str(source), "-o", str(output), *flags]), 0)
                    self.assertIs(solve.call_args.kwargs["bfs_parent_redundancy"], expected)
                    record = json.loads(output.read_text(encoding="utf-8"))
                    self.assertIs(record["bfs_parent_redundancy"], expected)
                    self.assertEqual(record["symmetry_breaking"], "bfs")

    def test_resume_does_not_reuse_the_other_converse_configuration(self):
        case = BenchmarkCase("sample", Path("sample.dot"), 3, 3, 1, None, None)
        record = {"status": "sat", "benchmark": {
            "encoding_revision": ENCODING_REVISION,
            "dataset": DATASET_NAME, "case_id": case.case_id,
            "input_state_count": 3, "optimization_backend": "sat",
            "strategy": "witness", "symmetry_breaking": "bfs",
            "factor_size_search_mode": FACTOR_SIZE_SEARCH_MODE,
            "factor_size_search_order": SEARCH_ORDERS["sat"],
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            for recorded in (None, True, False):
                if recorded is not None:
                    record["benchmark"]["bfs_parent_redundancy"] = recorded
                path.write_text(json.dumps(record), encoding="utf-8")
                for requested in (True, False):
                    found = _read_matching_result(
                        path, case, "sat", "witness", require_current_encoding=True,
                        bfs_parent_redundancy=requested)
                    self.assertEqual(found is not None, requested == (True if recorded is None else recorded))
            record["benchmark"].update({
                "dataset": all_small.DATASET_NAME, "input_kind": "minimal",
                "nonempty_requirement": all_small.NONEMPTY_REQUIREMENT,
                "input_dot_sha256": None, "timeout_seconds": 30,
                "strictness_mode": "finite", "factor_order": "max",
            })
            path.write_text(json.dumps(record), encoding="utf-8")
            wrapped = all_small.InputCase("minimal", case)
            self.assertIsNone(all_small.matching_result(path, wrapped, 30, "max"))
            self.assertIsNotNone(all_small.matching_result(path, wrapped, 30, "max", bfs_parent_redundancy=False))

    def test_nonminimal_runner_passes_variant_and_separates_output(self):
        case = BenchmarkCase("sample", Path("sample.dot"), 3, 3, 1, None, None)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(nonminimal, "load_nonminimal_cases", return_value=[case]):
                with patch.object(nonminimal, "_run_one", return_value={"status": "sat"}) as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(nonminimal.main([
                            "--output-root", directory, "--no-bfs-parent-redundancy"]), 0)
            self.assertFalse(run.call_args.kwargs["bfs_parent_redundancy"])
            self.assertEqual(run.call_args.kwargs["output_directory"], result_directory(
                Path(directory) / "without-bfs-parent-redundancy", case, "maxsat", "witness"))

    def test_removed_api_options_and_invalid_converse_are_rejected(self):
        for keyword, value in (("max_edits", 0), ("incremental_sat", True),
                               ("symmetry_breaking", "bfs")):
            with self.subTest(keyword=keyword), self.assertRaisesRegex(TypeError, keyword):
                find_minimum_edit_decomposition(trailing_b_mod_3(), **{keyword: value})
        for sizes in (None, (2, 2)):
            with self.assertRaisesRegex(ValueError, "bfs_parent_redundancy"):
                find_minimum_edit_decomposition(trailing_b_mod_3(), sizes, bfs_parent_redundancy=1)
