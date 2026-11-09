"""Independent graph checks and SAT entailment for the compact predicates."""
from collections import Counter, deque
import contextlib
import io
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from dfa import DFA
from pysat.solvers import Solver

from dfa_identify.cli import main as cli_main
from dfa_identify.minimum_edit import (
    _DirectEncoding, _normalize_target, encoding_revision,
    find_minimum_edit_decomposition,
)
from test_minimum_edit import assert_exact_and_strict, even_a_and_even_b, trailing_b_mod_3


class AutomataEncoding(_DirectEncoding):
    def _build(self):
        self._encode_automata()


def target_for(alphabet_size):
    return _normalize_target(DFA(start=0, inputs=set(range(alphabet_size)),
                                outputs={False, True}, label=lambda q: False,
                                transition=lambda q, a: q))


def bfs_order(table, alphabet_size):
    """Graph traversal independent of all SAT auxiliaries."""
    discovered, queue = [0], deque([0])
    while queue:
        state = queue.popleft()
        for symbol in range(alphabet_size):
            successor = table[state * alphabet_size + symbol]
            if successor not in discovered:
                discovered.append(successor)
                queue.append(successor)
    return discovered


class CompactBfsTests(unittest.TestCase):
    def test_projected_models_agree_with_independent_bfs_traversal(self):
        # Exhaust every total transition table in these small domains, then
        # test larger seeded tables and their BFS renumberings.
        rng = random.Random(2019)
        for size, alphabet_size in ((1, 0), (2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (2, 3), (6, 3)):
            tables = itertools.product(range(size), repeat=size * alphabet_size)
            if size == 6:
                tables = []
                for _ in range(80):
                    table = [rng.randrange(size) for _ in range(size * alphabet_size)]
                    tables.append(table)
                    order = bfs_order(table, alphabet_size)
                    if len(order) == size:
                        indices = {state: index for index, state in enumerate(order)}
                        tables.append([indices[table[state * alphabet_size + symbol]]
                                       for state in order for symbol in range(alphabet_size)])
            encodings = [AutomataEncoding(target_for(alphabet_size), (size, 1), True, "finite",
                                         redundancy, bfs_encoding=kind)
                         for kind in ("classic", "compact") for redundancy in (True, False)]
            with contextlib.ExitStack() as stack:
                solvers = [stack.enter_context(Solver(name="g3", bootstrap_with=e.cnf.clauses))
                           for e in encodings]
                for table in tables:
                    expected = bfs_order(table, alphabet_size) == list(range(size))
                    for encoding, solver in zip(encodings, solvers):
                        assumptions = [encoding.factor_transition(0, state, symbol,
                                                                  table[state * alphabet_size + symbol])
                                       for state in range(size) for symbol in range(alphabet_size)]
                        self.assertEqual(solver.solve(assumptions=assumptions), expected,
                                         (size, alphabet_size, table, encoding.bfs_encoding,
                                          encoding.bfs_parent_redundancy))

    def test_only_parent_converses_are_removed_and_each_is_entailed(self):
        for alphabet_size in (0, 1, 2, 3):
            for sizes in ((1, 1), (2, 3), (6, 7)):
                full = AutomataEncoding(target_for(alphabet_size), sizes, True, "finite",
                                        True, bfs_encoding="compact")
                reduced = AutomataEncoding(target_for(alphabet_size), sizes, True, "finite",
                                           False, bfs_encoding="compact")
                removed = [[-reduced.transition_relation(factor, parent, state),
                            -reduced.var("sb-nt", factor, state, parent),
                            reduced.enumeration_parent(factor, parent, state)]
                           for factor, size in enumerate(sizes) for state in range(1, size)
                           for parent in range(state)]
                self.assertEqual(full.pool.obj2id, reduced.pool.obj2id)
                self.assertEqual(Counter(map(tuple, full.cnf.clauses)),
                                 Counter(map(tuple, reduced.cnf.clauses)) + Counter(map(tuple, removed)))
                with Solver(name="g3", bootstrap_with=reduced.cnf.clauses) as solver:
                    self.assertEqual(solver.solve(), alphabet_size > 0 or max(sizes) == 1)
                    for clause in removed:
                        self.assertFalse(solver.solve(assumptions=[-literal for literal in clause]))

    def test_all_four_preserve_optima_and_decoded_languages(self):
        for optimization in ("sat", "maxsat", "enumeration"):
            for kind in ("classic", "compact"):
                for redundancy in (True, False):
                    for mode, target, expected in (("finite", trailing_b_mod_3(), 1),
                                                   ("infinite", even_a_and_even_b(), 0)):
                        result = find_minimum_edit_decomposition(
                            target, (2, 2), optimization=optimization, strictness_mode=mode,
                            bfs_encoding=kind, bfs_parent_redundancy=redundancy)
                        self.assertIsNotNone(result)
                        self.assertEqual(result.edit_distance, expected)
                        self.assertEqual(result.bfs_encoding, kind)
                        assert_exact_and_strict(result)
        for redundancy in (True, False):
            for product_encoding in ("factored", "direct"):
                automatic = find_minimum_edit_decomposition(
                    trailing_b_mod_3(), optimization="sat", factor_order="max",
                    bfs_encoding="compact", bfs_parent_redundancy=redundancy,
                    product_encoding=product_encoding)
                blocking = find_minimum_edit_decomposition(
                    trailing_b_mod_3(), (2, 2), optimization="sat", strictness=False,
                    bfs_encoding="compact", bfs_parent_redundancy=redundancy,
                    product_encoding=product_encoding)
                for result in (automatic, blocking):
                    self.assertEqual(result.edit_distance, 1)
                    assert_exact_and_strict(result)

    def test_missing_first_prefix_recurrence_breaks_converse_entailment(self):
        encoding = AutomataEncoding(target_for(2), (3, 1), True, "finite",
                                    False, bfs_encoding="compact")
        prefix0 = encoding.var("sb-nt", 0, 2, 0)
        prefix1 = encoding.var("sb-nt", 0, 2, 1)
        edge0 = encoding.transition_relation(0, 0, 2)
        first_recurrence = {(-prefix1, prefix0), (-prefix1, -edge0),
                            (prefix1, -prefix0, edge0)}
        incomplete = [clause for clause in encoding.cnf.clauses
                      if tuple(clause) not in first_recurrence]
        assumptions = [edge0, -encoding.enumeration_parent(0, 0, 2),
                       encoding.enumeration_parent(0, 1, 2)]
        with Solver(name="g3", bootstrap_with=incomplete) as solver:
            self.assertTrue(solver.solve(assumptions=assumptions))
        with Solver(name="g3", bootstrap_with=encoding.cnf.clauses) as solver:
            self.assertFalse(solver.solve(assumptions=assumptions))

    def test_cli_metadata_defaults_overrides_and_invalid_options(self):
        with tempfile.TemporaryDirectory() as name:
            source, output = Path(name) / "input.json", Path(name) / "result.json"
            for configured, flags, expected in ((None, [], "classic"), ("compact", [], "compact"),
                                                ("classic", ["--bfs-encoding", "compact"], "compact"),
                                                ("compact", ["--bfs-encoding", "classic"], "classic")):
                payload = {"states": [0], "alphabet": [], "start": 0,
                           "accepting": [], "transitions": []}
                if configured:
                    payload["bfs_encoding"] = configured
                source.write_text(json.dumps(payload), encoding="utf-8")
                with patch("dfa_identify.cli.find_minimum_edit_decomposition", return_value=None) as solve:
                    with contextlib.redirect_stdout(io.StringIO()):
                        cli_main([str(source), "-o", str(output), *flags])
                record = json.loads(output.read_text())
                self.assertEqual(solve.call_args.kwargs["bfs_encoding"], expected)
                self.assertEqual(record["bfs_encoding"], expected)
                self.assertEqual(record["encoding_revision"], encoding_revision("maxsat", "factored", expected))
        for invalid in ("unknown", None, False):
            with self.assertRaisesRegex(ValueError, "bfs_encoding"):
                find_minimum_edit_decomposition(trailing_b_mod_3(), bfs_encoding=invalid)
        self.assertNotEqual(encoding_revision("sat"), encoding_revision("sat", bfs_encoding="compact"))


if __name__ == "__main__":
    unittest.main()
