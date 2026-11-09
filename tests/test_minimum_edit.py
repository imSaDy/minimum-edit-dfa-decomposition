import unittest
from itertools import product
from unittest.mock import patch

from dfa import DFA
from pysat.solvers import Solver

from dfa_identify.minimum_edit import (
    DecompositionResult,
    factor_size_pairs,
    find_minimum_edit_decomposition,
    _find_minimum_edit_decomposition_for_sizes,
    _DirectEncoding,
    _normalize_target,
)


def trailing_b_mod_3() -> DFA:
    transitions = {
        (0, "a"): 0,
        (0, "b"): 1,
        (1, "a"): 0,
        (1, "b"): 2,
        (2, "a"): 0,
        (2, "b"): 0,
    }
    return DFA(
        start=0,
        inputs={"a", "b"},
        outputs={False, True},
        states=(0, 1, 2),
        label=lambda state: state == 2,
        transition=lambda state, symbol: transitions[state, symbol],
    )


def unary_two_state_dfa(transitions, accepting_mask) -> DFA:
    return DFA(
        start=0,
        inputs={"a"},
        outputs={False, True},
        states=(0, 1),
        label=lambda state: bool(accepting_mask & (1 << state)),
        transition=lambda state, _symbol: transitions[state],
    )


def dfa_with_state_count(size: int) -> DFA:
    return DFA(
        start=0,
        inputs={"a"},
        outputs={False, True},
        states=tuple(range(size)),
        label=lambda _state: False,
        transition=lambda state, _symbol: (state + 1) % size,
    )


def even_a_and_even_b() -> DFA:
    states = tuple(product(range(2), repeat=2))
    return DFA(
        start=(0, 0),
        inputs={"a", "b"},
        outputs={False, True},
        states=states,
        label=lambda state: state == (0, 0),
        transition=lambda state, symbol: (
            (1 - state[0], state[1]) if symbol == "a" else (state[0], 1 - state[1])
        ),
    )


def even_a_and_b_mod_3() -> DFA:
    states = tuple(product(range(2), range(3)))
    return DFA(
        start=(0, 0),
        inputs={"a", "b"},
        outputs={False, True},
        states=states,
        label=lambda state: state == (0, 0),
        transition=lambda state, symbol: (
            (1 - state[0], state[1])
            if symbol == "a"
            else (state[0], (state[1] + 1) % 3)
        ),
    )


def binary_cofinite_except_length_one() -> DFA:
    """Recognize every binary word except the two words of length one."""

    return DFA(
        start=0,
        inputs={"a", "b"},
        outputs={False, True},
        states=(0, 1, 2),
        label=lambda state: state != 1,
        transition=lambda state, _symbol: min(state + 1, 2),
    )


def synthetic_result(
    pair: tuple[int, int],
    cost: int,
    *,
    optimization: str = "maxsat",
    symmetry_breaking: str = "bfs",
) -> DecompositionResult:
    return DecompositionResult(
        repaired=dfa_with_state_count(5),
        factors=(dfa_with_state_count(pair[0]), dfa_with_state_count(pair[1])),
        edit_distance=cost,
        transition_edits=(),
        acceptance_edits=(),
        witnesses=((), ("a",)),
        strictness_strategy="witness",
        strictness_mode="finite",
        optimization_backend=optimization,  # type: ignore[arg-type]
        symmetry_breaking=symmetry_breaking,  # type: ignore[arg-type]
        blocked_models=0,
        checked_models=1,
        solver_calls=1,
        costs_seen=(cost,),
    )


def brute_force_unary_optimum(original: DFA) -> int | None:
    automata = [
        unary_two_state_dfa(transitions, accepting_mask)
        for transitions in product(range(2), repeat=2)
        for accepting_mask in range(4)
    ]
    # An exact two-state factor must make state 1 reachable from start state 0.
    factors = [dfa for dfa in automata if dfa._transition(0, "a") == 1]
    witness_words = [tuple("a" for _ in range(length)) for length in range(4)]

    optimum = None
    for repaired in automata:
        transition_cost = sum(
            original._transition(state, "a") != repaired._transition(state, "a")
            for state in range(2)
        )
        acceptance_cost = sum(
            bool(original._label(state)) != bool(repaired._label(state))
            for state in range(2)
        )
        cost = transition_cost + acceptance_cost
        if optimum is not None and cost >= optimum:
            continue
        for first in factors:
            for second in factors:
                if repaired != (first & second):
                    continue
                if not any(
                    first.label(word) and second.label(word)
                    for word in witness_words
                ):
                    continue
                first_strict = any(
                    first.label(word) and not second.label(word)
                    for word in witness_words
                )
                second_strict = any(
                    second.label(word) and not first.label(word)
                    for word in witness_words
                )
                if first_strict and second_strict:
                    optimum = cost
                    break
            if optimum == cost:
                break
    return optimum


def assert_exact_and_strict(result) -> None:
    repaired = result.repaired
    first, second = result.factors
    assert len(first.states()) == 2
    assert len(second.states()) == 2
    assert repaired == (first & second)
    first_witness, second_witness = result.witnesses
    assert first.label(first_witness) is True
    assert second.label(first_witness) is False
    assert second.label(second_witness) is True
    assert first.label(second_witness) is False
    assert result.common_witness is not None
    assert first.label(result.common_witness) is True
    assert second.label(result.common_witness) is True
    assert repaired.label(result.common_witness) is True


class MinimumEditTests(unittest.TestCase):
    def test_factor_size_pairs_are_unique_and_ordered_by_total_states(self) -> None:
        self.assertEqual(
            factor_size_pairs(5),
            ((2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)),
        )

    def test_factor_size_pairs_support_max_primary_order(self) -> None:
        self.assertEqual(
            factor_size_pairs(5, ordering="max"),
            ((2, 2), (2, 3), (3, 3), (2, 4), (3, 4), (4, 4)),
        )

    def test_product_order_uses_sum_to_break_product_ties(self) -> None:
        pairs = factor_size_pairs(8, ordering="product")
        self.assertLess(pairs.index((2, 7)), pairs.index((3, 5)))
        self.assertLess(pairs.index((3, 4)), pairs.index((2, 6)))
        self.assertGreater(
            factor_size_pairs(8, ordering="sum").index((2, 7)),
            factor_size_pairs(8, ordering="sum").index((3, 5)),
        )

    def test_product_order_breaks_equal_edit_costs_by_factor_product(self) -> None:
        def solve(_target, pair, **_kwargs):
            return synthetic_result(
                pair, 1 if pair in ((2, 7), (3, 5)) else 2
            )

        with patch(
            "dfa_identify.minimum_edit._find_minimum_edit_decomposition_for_sizes",
            side_effect=solve,
        ):
            product_result = find_minimum_edit_decomposition(
                dfa_with_state_count(8), factor_order="product"
            )
            sum_result = find_minimum_edit_decomposition(
                dfa_with_state_count(8), factor_order="sum"
            )

        assert product_result is not None and sum_result is not None
        self.assertEqual(tuple(len(f.states()) for f in product_result.factors), (2, 7))
        self.assertEqual(tuple(len(f.states()) for f in sum_result.factors), (3, 5))
        self.assertEqual(product_result.edit_distance, sum_result.edit_distance)

    def test_max_primary_order_breaks_size_ties_by_larger_factor(self) -> None:
        costs = {
            (2, 2): 2,
            (2, 3): 2,
            (2, 4): 1,
            (3, 3): 1,
            (3, 4): 2,
            (4, 4): 2,
        }

        def solve(_target, pair, **_kwargs):
            return synthetic_result(pair, costs[pair])

        with patch(
            "dfa_identify.minimum_edit._find_minimum_edit_decomposition_for_sizes",
            side_effect=solve,
        ) as mocked_solve:
            result = find_minimum_edit_decomposition(
                dfa_with_state_count(5), factor_order="max"
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(tuple(len(f.states()) for f in result.factors), (3, 3))
        self.assertEqual(
            [call.args[1] for call in mocked_solve.call_args_list],
            list(factor_size_pairs(5, ordering="max")),
        )

    def test_all_size_search_checks_every_pair_and_breaks_ties_by_state_sum(
        self,
    ) -> None:
        costs = {
            (2, 2): 3,
            (2, 3): 1,
            (2, 4): 1,
            (3, 3): 1,
            (3, 4): 2,
            (4, 4): 2,
        }

        def solve(_target, pair, **_kwargs):
            return synthetic_result(pair, costs[pair])

        with patch(
            "dfa_identify.minimum_edit._find_minimum_edit_decomposition_for_sizes",
            side_effect=solve,
        ) as mocked_solve:
            result = find_minimum_edit_decomposition(dfa_with_state_count(5))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(tuple(len(f.states()) for f in result.factors), (2, 3))
        self.assertEqual(result.edit_distance, 1)
        self.assertEqual(mocked_solve.call_count, 6)
        self.assertTrue(result.factor_size_search_complete)
        self.assertFalse(result.stopped_on_zero)

    def test_all_size_search_stops_at_the_first_zero_cost(self) -> None:
        costs = {(2, 2): 2, (2, 3): 1, (2, 4): 0}

        def solve(_target, pair, **_kwargs):
            return synthetic_result(pair, costs[pair])

        with patch(
            "dfa_identify.minimum_edit._find_minimum_edit_decomposition_for_sizes",
            side_effect=solve,
        ) as mocked_solve:
            result = find_minimum_edit_decomposition(dfa_with_state_count(5))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(tuple(len(f.states()) for f in result.factors), (2, 4))
        self.assertEqual(result.edit_distance, 0)
        self.assertEqual(mocked_solve.call_count, 3)
        self.assertFalse(result.factor_size_search_complete)
        self.assertTrue(result.stopped_on_zero)

    def test_all_size_sat_searches_budgets_before_factor_pairs(self) -> None:
        calls: list[tuple[int, tuple[int, int]]] = []

        def solve(_target, pair, **kwargs):
            budget = kwargs["sat_budget"]
            calls.append((budget, pair))
            if budget == 1 and pair == (2, 3):
                return synthetic_result(pair, 1, optimization="sat")
            return None

        with patch(
            "dfa_identify.minimum_edit._find_minimum_edit_decomposition_for_sizes",
            side_effect=solve,
        ):
            result = find_minimum_edit_decomposition(
                dfa_with_state_count(5), optimization="sat"
            )

        self.assertIsNotNone(result)
        assert result is not None
        pairs = factor_size_pairs(5)
        self.assertEqual(
            calls,
            [*( (0, pair) for pair in pairs ), (1, (2, 2)), (1, (2, 3))],
        )
        self.assertEqual(result.edit_distance, 1)
        self.assertEqual(result.costs_seen, (0, 1))
        self.assertEqual(result.solver_calls, len(calls))
        self.assertTrue(result.factor_size_search_complete)
        self.assertFalse(result.stopped_on_zero)
        self.assertEqual(
            tuple(attempt.edit_budget for attempt in result.factor_size_attempts),
            (0, 0, 0, 0, 0, 0, 1, 1),
        )

    def test_real_zero_cost_decomposition_stops_at_two_by_two(self) -> None:
        result = find_minimum_edit_decomposition(even_a_and_even_b())

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.edit_distance, 0)
        self.assertEqual(tuple(len(f.states()) for f in result.factors), (2, 2))
        self.assertEqual(
            tuple(attempt.factor_sizes for attempt in result.factor_size_attempts),
            ((2, 2),),
        )
        self.assertTrue(result.stopped_on_zero)
        assert_exact_and_strict(result)

    def test_all_size_search_requires_encoded_strictness(self) -> None:
        with self.assertRaisesRegex(ValueError, "automatic factor-size search"):
            find_minimum_edit_decomposition(trailing_b_mod_3(), strictness=False)

    def test_all_size_search_supports_sat_and_maxsat(self) -> None:
        for optimization in ("maxsat", "sat"):
            result = find_minimum_edit_decomposition(
                trailing_b_mod_3(), optimization=optimization
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.edit_distance, 1)
            self.assertEqual(result.optimization_backend, optimization)
            self.assertTrue(result.factor_size_search_complete)
            assert_exact_and_strict(result)

    def test_all_size_search_reports_the_current_pair(self) -> None:
        messages: list[str] = []
        result = find_minimum_edit_decomposition(
            even_a_and_even_b(), progress=messages.append
        )

        self.assertIsNotNone(result)
        self.assertTrue(any("solving 2x2" in message for message in messages))
        self.assertTrue(any("finished 2x2: cost=0" in message for message in messages))
        self.assertTrue(any("stopping on zero cost" in message for message in messages))

    def test_note_witness_encoding_finds_slide_example_optimum(self) -> None:
        result = find_minimum_edit_decomposition(
            trailing_b_mod_3(), (2, 2), strictness=True
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.edit_distance, 1)
        self.assertEqual(result.strictness_strategy, "witness")
        self.assertEqual(result.optimization_backend, "maxsat")
        self.assertEqual(result.blocked_models, 0)
        self.assertEqual(result.costs_seen, (1,))
        assert_exact_and_strict(result)

    def test_ordinary_sat_iterative_deepening_finds_same_optimum(self) -> None:
        result = find_minimum_edit_decomposition(
            trailing_b_mod_3(),
            (2, 2),
            strictness=True,
            optimization="sat",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.edit_distance, 1)
        self.assertEqual(result.optimization_backend, "sat")
        self.assertEqual(result.costs_seen, (0, 1))
        self.assertGreaterEqual(result.solver_calls, 2)
        assert_exact_and_strict(result)





    def test_bfs_is_permanent_and_preserves_the_known_optimum(self) -> None:
        for optimization in ("maxsat", "sat"):
            result = find_minimum_edit_decomposition(
                trailing_b_mod_3(),
                (2, 2),
                strictness=True,
                optimization=optimization,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.edit_distance, 1)
            self.assertEqual(result.symmetry_breaking, "bfs")
            assert_exact_and_strict(result)

    def test_bfs_cannot_be_disabled(self) -> None:
        for sizes in (None, (2, 2)):
            with self.subTest(sizes=sizes):
                with self.assertRaisesRegex(TypeError, "symmetry_breaking"):
                    find_minimum_edit_decomposition(
                        trailing_b_mod_3(), sizes,
                        symmetry_breaking="none",  # type: ignore[arg-type]
                    )

    def test_bfs_matches_independent_three_state_traversal(self) -> None:
        class AutomataEncoding(_DirectEncoding):
            def _build(self):
                # Test numbering without equality or witness constraints
                # making a rejected transition table infeasible for other reasons.
                self._encode_automata()

        encoding = AutomataEncoding(
            _normalize_target(trailing_b_mod_3()), (3, 1), True, "finite"
        )
        with Solver(name="g3", bootstrap_with=encoding.cnf.clauses) as solver:
            self.assertTrue(solver.solve())
            for state in (1, 2):
                # BFS alone entails an incoming edge from a preceding state.
                incoming = [
                    encoding.factor_transition(0, predecessor, symbol, state)
                    for predecessor in range(state)
                    for symbol in range(2)
                ]
                parents = [encoding.enumeration_parent(0, parent, state)
                           for parent in range(state)]
                self.assertIn(parents, encoding.cnf.clauses)
                self.assertNotIn(incoming, encoding.cnf.clauses)
                self.assertFalse(solver.solve(assumptions=[-edge for edge in incoming]))

            for entries in product(range(3), repeat=6):
                # A plain queue independently numbers all states by BFS,
                # exploring symbol a before b.
                discovered = [0]
                for state in discovered:
                    for symbol in range(2):
                        target = entries[2 * state + symbol]
                        if target not in discovered:
                            discovered.append(target)
                expected = discovered == [0, 1, 2]
                transitions = [
                    encoding.factor_transition(0, state, symbol,
                                               entries[2 * state + symbol])
                    for state in range(3) for symbol in range(2)
                ]
                self.assertEqual(
                    solver.solve(assumptions=transitions), expected, entries
                )

    def test_infinite_strictness_finds_two_infinite_differences(self) -> None:
        for optimization in ("sat",):
            with self.subTest(optimization=optimization):
                result = _find_minimum_edit_decomposition_for_sizes(
                    even_a_and_even_b(),
                    (2, 2),
                    strictness=True,
                    strictness_mode="infinite",
                    optimization=optimization,
                    sat_budget=0,
                )

                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.edit_distance, 0)
                self.assertEqual(result.strictness_mode, "infinite")
                self.assertTrue(all(len(word) >= 4 for word in result.witnesses))
                assert_exact_and_strict(result)

    def test_infinite_strictness_rejects_cofinite_directional_differences(
        self,
    ) -> None:
        target = binary_cofinite_except_length_one()
        finite = _find_minimum_edit_decomposition_for_sizes(
            target,
            (3, 3),
            strictness=True,
            strictness_mode="finite",
            optimization="sat",
            sat_budget=0,
        )
        infinite = _find_minimum_edit_decomposition_for_sizes(
            target,
            (3, 3),
            strictness=True,
            strictness_mode="infinite",
            optimization="sat",
            sat_budget=0,
        )

        self.assertIsNotNone(finite)
        self.assertIsNone(infinite)

    def test_manual_blocking_finds_the_same_optimum(self) -> None:
        result = find_minimum_edit_decomposition(
            trailing_b_mod_3(), (2, 2), strictness=False
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.edit_distance, 1)
        self.assertEqual(result.strictness_strategy, "blocking")
        self.assertGreaterEqual(result.checked_models, 1)
        assert_exact_and_strict(result)


    def test_ordinary_sat_supports_manual_strictness_blocking(self) -> None:
        result = find_minimum_edit_decomposition(
            trailing_b_mod_3(),
            (2, 2),
            strictness=False,
            optimization="sat",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.edit_distance, 1)
        self.assertEqual(result.strictness_strategy, "blocking")
        self.assertEqual(result.optimization_backend, "sat")
        assert_exact_and_strict(result)

    def test_two_single_state_factors_cannot_both_be_strict(self) -> None:
        result = find_minimum_edit_decomposition(
            trailing_b_mod_3(), (1, 1), strictness=True
        )
        self.assertIsNone(result)

    def test_empty_repaired_language_is_rejected_at_zero_edits(self) -> None:
        for optimization in ("sat",):
            for strictness in (True, False):
                with self.subTest(optimization=optimization, strictness=strictness):
                    result = _find_minimum_edit_decomposition_for_sizes(
                        dfa_with_state_count(3),
                        (2, 2),
                        strictness=strictness,
                        optimization=optimization,
                        sat_budget=0,
                    )
                    self.assertIsNone(result)

    def test_exact_maxsat_matches_exhaustive_unary_search(self) -> None:
        for transitions in product(range(2), repeat=2):
            for accepting_mask in range(4):
                target = unary_two_state_dfa(transitions, accepting_mask)
                expected = brute_force_unary_optimum(target)
                for optimization in ("maxsat", "sat"):
                    result = find_minimum_edit_decomposition(
                        target,
                        (2, 2),
                        strictness=True,
                        optimization=optimization,
                    )
                    actual = None if result is None else result.edit_distance
                    self.assertEqual(
                        actual,
                        expected,
                        msg=(transitions, accepting_mask, optimization),
                    )


    def test_rejects_unknown_optimization_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimization"):
            find_minimum_edit_decomposition(
                trailing_b_mod_3(),
                (2, 2),
                optimization="unknown",  # type: ignore[arg-type]
            )


    def test_rejects_unknown_strictness_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictness_mode"):
            find_minimum_edit_decomposition(
                trailing_b_mod_3(),
                (2, 2),
                strictness_mode="eventual",  # type: ignore[arg-type]
            )

    def test_rejects_unknown_factor_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "factor_order"):
            find_minimum_edit_decomposition(
                trailing_b_mod_3(),
                factor_order="volume",  # type: ignore[arg-type]
            )



    def test_rejects_kissat_for_sat_optimization(self) -> None:
        with self.assertRaisesRegex(ValueError, "not supported"):
            find_minimum_edit_decomposition(
                trailing_b_mod_3(),
                (2, 2),
                optimization="sat",
                sat_solver="kissat404",
            )


if __name__ == "__main__":
    unittest.main()
