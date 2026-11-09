"""Minimum-edit decomposition of a DFA into two strict intersection factors.

This module implements the direct formulation from
``Minimum_Edit_DFA_Decomposition.pdf`` with either Exact MaxSAT or ordinary
SAT iterative deepening, or external repair enumeration with SAT checks.
Exact language equality is encoded over the
complete reachable product, rather than over a finite sample.  The public
entry point can solve one prescribed factor-size pair or optimize across all
non-trivial proper pairs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from itertools import combinations, product
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, Literal, Sequence

from dfa import DFA
from pysat.card import CardEnc, EncType
from pysat.examples.rc2 import RC2
from pysat.formula import CNF, IDPool, WCNF
from pysat.solvers import Solver, SolverNames

Word = tuple[Any, ...]
StrictnessStrategy = Literal["witness", "blocking"]
StrictnessMode = Literal["finite", "infinite"]
OptimizationBackend = Literal["maxsat", "sat", "enumeration"]
ProductEncoding = Literal["factored", "direct"]
BfsEncoding = Literal["classic", "compact"]
SymmetryBreaking = Literal["bfs"]
FactorSizeOrder = Literal["sum", "max", "product"]
ENCODING_REVISION = "permanent-bfs-v3"
ENUMERATION_ENCODING_REVISION = "enumeration-external-nonempty-v3"


def normalize_optimization(optimization: str) -> OptimizationBackend:
    """Accept the former method name when reading older configurations."""

    if optimization == "hybrid":
        return "enumeration"
    if optimization not in ("maxsat", "sat", "enumeration"):
        raise ValueError("optimization must be 'maxsat', 'sat', or 'enumeration'")
    return optimization


def _validate_product_encoding(product_encoding: str) -> None:
    if product_encoding not in ("factored", "direct"):
        raise ValueError("product_encoding must be 'factored' or 'direct'")


def _resolve_bfs_parent_redundancy(
    setting: bool | None, legacy_setting: bool | None,
    previous_setting: bool | None = None,
) -> bool:
    values = []
    for name, value in (("bfs_parent_redundancy", setting),
                        ("bfs_converse", legacy_setting),
                        ("bfs_parent_strengthening", previous_setting)):
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")
        if value is not None:
            values.append(value)
    if len(set(values)) > 1:
        raise ValueError("conflicting BFS parent redundancy settings or aliases")
    return values[0] if values else True


def encoding_revision(
    optimization: OptimizationBackend, product_encoding: ProductEncoding = "factored",
    bfs_encoding: BfsEncoding = "classic",
) -> str:
    """Identify the method and product representation for safe resuming."""

    optimization = normalize_optimization(optimization)
    _validate_product_encoding(product_encoding)
    _validate_bfs_encoding(bfs_encoding)
    base = ENUMERATION_ENCODING_REVISION if optimization == "enumeration" else ENCODING_REVISION
    if bfs_encoding == "compact":
        base += "-bfs-compact-v1"
    return base if product_encoding == "factored" else f"{base}-product-direct-v1"


def _validate_bfs_encoding(bfs_encoding: str) -> None:
    if bfs_encoding not in ("classic", "compact"):
        raise ValueError("bfs_encoding must be 'classic' or 'compact'")


@dataclass(frozen=True)
class TransitionEdit:
    state: Any
    symbol: Any
    old_target: Any
    new_target: Any


@dataclass(frozen=True)
class AcceptanceEdit:
    state: Any
    old_value: bool
    new_value: bool


@dataclass(frozen=True)
class FactorSizeAttempt:
    """The outcome of one exact factor-size subproblem."""

    factor_sizes: tuple[int, int]
    edit_distance: int | None
    elapsed_seconds: float
    edit_budget: int | None = None
    repair_candidates_checked: int = 0
    empty_repair_candidates_skipped: int = 0


@dataclass(frozen=True)
class DecompositionResult:
    """A minimum-edit nonempty repaired DFA and its two strict factors."""

    repaired: DFA
    factors: tuple[DFA, DFA]
    edit_distance: int
    transition_edits: tuple[TransitionEdit, ...]
    acceptance_edits: tuple[AcceptanceEdit, ...]
    witnesses: tuple[Word, Word]
    strictness_strategy: StrictnessStrategy
    optimization_backend: OptimizationBackend
    symmetry_breaking: SymmetryBreaking
    blocked_models: int
    checked_models: int
    solver_calls: int
    costs_seen: tuple[int, ...]
    strictness_mode: StrictnessMode = "finite"
    factor_size_attempts: tuple[FactorSizeAttempt, ...] = ()
    factor_size_search_complete: bool = False
    stopped_on_zero: bool = False
    common_witness: Word | None = None
    bfs_parent_redundancy: bool = True
    bfs_encoding: BfsEncoding = "classic"
    product_encoding: ProductEncoding = "factored"
    base_encoding_variables: int = 0
    base_encoding_clauses: int = 0
    base_encoding_literals: int = 0
    base_encoding_max_clause_length: int = 0
    repair_candidates_checked: int = 0
    empty_repair_candidates_skipped: int = 0

    @property
    def bfs_converse(self) -> bool:
        """Compatibility accessor for the former option name."""

        return self.bfs_parent_redundancy

    @property
    def bfs_parent_strengthening(self) -> bool:
        """Compatibility accessor for the intermediate option name."""

        return self.bfs_parent_redundancy


@dataclass(frozen=True)
class _NormalizedDFA:
    original_states: tuple[Any, ...]
    alphabet: tuple[Any, ...]
    transitions: tuple[tuple[int, ...], ...]
    accepting: tuple[bool, ...]

    @property
    def n_states(self) -> int:
        return len(self.original_states)


def _stable_order(values: Iterable[Any]) -> tuple[Any, ...]:
    values = tuple(values)
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(
            sorted(values, key=lambda value: (type(value).__name__, repr(value)))
        )


def _factor_size_order_key(
    sizes: tuple[int, int], ordering: FactorSizeOrder
) -> tuple[Any, ...]:
    if ordering == "sum":
        return (sum(sizes), sizes)
    if ordering == "max":
        return (max(sizes), sum(sizes), sizes)
    if ordering == "product":
        return (sizes[0] * sizes[1], sum(sizes), sizes)
    raise ValueError("factor_order must be 'sum', 'max', or 'product'")


def factor_size_pairs(
    target_state_count: int,
    ordering: FactorSizeOrder = "sum",
) -> tuple[tuple[int, int], ...]:
    """Return every unique non-trivial proper factor-size pair, small first.

    Both factors have at least two states and strictly fewer states than the
    input DFA.  Only ``n1 <= n2`` is retained because swapping the two factors
    does not change their intersection.  ``ordering="sum"`` uses increasing
    total factor states.  ``ordering="max"`` first minimizes the larger factor
    and then the total number of factor states.
    ``ordering="product"`` first minimizes the product of the factor sizes,
    then their sum.
    """

    if (
        not isinstance(target_state_count, int)
        or isinstance(target_state_count, bool)
        or target_state_count < 1
    ):
        raise ValueError("target_state_count must be a positive integer")
    if ordering not in ("sum", "max", "product"):
        raise ValueError("factor_order must be 'sum', 'max', or 'product'")
    return tuple(
        sorted(
            (
                (first, second)
                for first in range(2, target_state_count)
                for second in range(first, target_state_count)
            ),
            key=lambda sizes: _factor_size_order_key(sizes, ordering),
        )
    )


def _normalize_target(target: DFA) -> _NormalizedDFA:
    if target.inputs is None:
        raise ValueError("target.inputs must explicitly define a finite alphabet")

    alphabet = _stable_order(target.inputs)
    known_states = set(target.states())
    if target.start not in known_states:
        known_states.add(target.start)

    # Put reachable states first, in deterministic breadth-first order.  Any
    # explicitly supplied unreachable states follow and still count as part of
    # the edit domain.
    ordered_states: list[Any] = []
    queued = {target.start}
    queue = deque([target.start])
    while queue:
        state = queue.popleft()
        ordered_states.append(state)
        for symbol in alphabet:
            successor = target._transition(state, symbol)
            if successor not in known_states:
                raise ValueError(
                    f"transition from {state!r} on {symbol!r} leaves the declared state set"
                )
            if successor not in queued:
                queued.add(successor)
                queue.append(successor)

    ordered_states.extend(_stable_order(known_states - queued))
    state_to_index = {state: index for index, state in enumerate(ordered_states)}

    transitions: list[tuple[int, ...]] = []
    accepting: list[bool] = []
    for state in ordered_states:
        row = tuple(
            state_to_index[target._transition(state, symbol)] for symbol in alphabet
        )
        transitions.append(row)
        label = target._label(state)
        if label not in (True, False):
            raise ValueError("the target DFA must have Boolean state labels")
        accepting.append(bool(label))

    return _NormalizedDFA(
        original_states=tuple(ordered_states),
        alphabet=alphabet,
        transitions=tuple(transitions),
        accepting=tuple(accepting),
    )


def _one_hot(cnf: CNF, literals: Sequence[int]) -> None:
    if not literals:
        cnf.append([])
        return
    cnf.append(list(literals))
    for left_index, left in enumerate(literals):
        for right in literals[left_index + 1 :]:
            cnf.append([-left, -right])


class _DirectEncoding:
    def __init__(
        self,
        target: _NormalizedDFA,
        factor_sizes: tuple[int, int],
        encode_strictness: bool,
        strictness_mode: StrictnessMode,
        bfs_parent_redundancy: bool = True,
        product_encoding: ProductEncoding = "factored",
        bfs_encoding: BfsEncoding = "classic",
    ) -> None:
        if not isinstance(bfs_parent_redundancy, bool):
            raise ValueError("bfs_parent_redundancy must be true or false")
        self.bfs_parent_redundancy = bfs_parent_redundancy
        _validate_bfs_encoding(bfs_encoding)
        self.bfs_encoding = bfs_encoding
        _validate_product_encoding(product_encoding)
        self.product_encoding = product_encoding
        self.target = target
        self.factor_sizes = factor_sizes
        self.encode_strictness = encode_strictness
        self.strictness_mode = strictness_mode
        self.pool = IDPool(start_from=1)
        self.cnf = CNF()
        self.edit_literals: list[int] = []
        self._build()
        # Count the application CNF before SAT edit bounds, blocking clauses,
        # assumptions, or transformations internal to the native solver.
        self.base_encoding_variables = self.cnf.nv
        self.base_encoding_clauses = len(self.cnf.clauses)
        self.base_encoding_literals = sum(map(len, self.cnf.clauses))
        self.base_encoding_max_clause_length = max(map(len, self.cnf.clauses), default=0)

    def var(self, family: str, *indices: Any) -> int:
        return self.pool.id((family, *indices))

    def repaired_transition(self, state: int, symbol: int, target: int) -> int:
        return self.var("D", state, symbol, target)

    def repaired_accepting(self, state: int) -> int:
        return self.var("F", state)

    def factor_transition(
        self, factor: int, state: int, symbol: int, target: int
    ) -> int:
        return self.var("T", factor, state, symbol, target)

    def factor_accepting(self, factor: int, state: int) -> int:
        return self.var("G", factor, state)

    def product_relation(self, state: int, first: int, second: int) -> int:
        return self.var("R", state, first, second)

    def enumeration_parent(self, factor: int, parent: int, state: int) -> int:
        if not parent < state:
            raise ValueError("a BFS enumeration parent must have a smaller index")
        return self.var("sb-parent", factor, parent, state)

    def transition_relation(self, factor: int, parent: int, state: int) -> int:
        if not parent < state:
            raise ValueError("a BFS transition relation requires parent < state")
        return self.var("sb-transition", factor, parent, state)

    def enumeration_label(self, factor: int, symbol: int, state: int) -> int:
        if state <= 0:
            raise ValueError("only non-initial states have BFS enumeration labels")
        return self.var("sb-label", factor, symbol, state)

    def _encode_factor_bfs_symmetry(self, factor: int, size: int) -> None:
        """Keep one breadth-first numbering from each reachable DFA isomorph."""

        if self.bfs_encoding == "compact":
            self._encode_factor_compact_bfs_symmetry(factor, size)
            return

        alphabet_size = len(self.target.alphabet)

        # Each noninitial state has a preceding BFS parent.  Together with
        # the parent/transition definitions, this implies accessibility from
        # state 0 without separate incoming-transition clauses.  t(i,j)
        # means some symbol transitions i to j; p(i,j) selects the least source.
        for state in range(1, size):
            self.cnf.append(
                [
                    self.enumeration_parent(factor, parent, state)
                    for parent in range(state)
                ]
            )
            for parent in range(state):
                parent_literal = self.enumeration_parent(factor, parent, state)
                transition_literal = self.transition_relation(
                    factor, parent, state
                )
                transitions = [
                    self.factor_transition(factor, parent, symbol, state)
                    for symbol in range(alphabet_size)
                ]

                # transition_literal <-> OR(transitions)
                self.cnf.append([-transition_literal, *transitions])
                for transition in transitions:
                    self.cnf.append([transition_literal, -transition])

                # The selected parent has a transition to this state and is
                # the smallest source state that does so.
                self.cnf.append([transition_literal, -parent_literal])
                for earlier_parent in range(parent):
                    self.cnf.append(
                        [
                            -parent_literal,
                            -self.transition_relation(
                                factor, earlier_parent, state
                            ),
                        ]
                    )
                # This converse follows from parent existence and the
                # forward parent implications. Retain it by default
                # for propagation; allow controlled timing experiments.
                if self.bfs_parent_redundancy:
                    self.cnf.append(
                        [
                            -transition_literal,
                            parent_literal,
                            *[
                                self.transition_relation(
                                    factor, earlier_parent, state
                                )
                                for earlier_parent in range(parent)
                            ],
                        ]
                    )

                # When parent is selected, m(symbol,state) selects the least
                # alphabet symbol on a parent -> state transition.
                for symbol, transition in enumerate(transitions):
                    label = self.enumeration_label(factor, symbol, state)
                    self.cnf.append([-parent_literal, -label, transition])
                    self.cnf.append(
                        [
                            -transition,
                            -parent_literal,
                            label,
                            *transitions[:symbol],
                        ]
                    )
                    for earlier_transition in transitions[:symbol]:
                        self.cnf.append(
                            [-parent_literal, -label, -earlier_transition]
                        )

        # BFS numbering requires nondecreasing parent indices.  Consecutive
        # children of the same parent are ordered by their transition labels.
        for state in range(1, size - 1):
            next_state = state + 1
            for parent in range(state):
                current_parent = self.enumeration_parent(factor, parent, state)
                for earlier_parent in range(parent):
                    self.cnf.append(
                        [
                            -current_parent,
                            -self.enumeration_parent(
                                factor, earlier_parent, next_state
                            ),
                        ]
                    )

                next_parent = self.enumeration_parent(factor, parent, next_state)
                for later_symbol in range(alphabet_size):
                    for earlier_symbol in range(later_symbol):
                        self.cnf.append(
                            [
                                -current_parent,
                                -next_parent,
                                -self.enumeration_label(
                                    factor, later_symbol, state
                                ),
                                -self.enumeration_label(
                                    factor, earlier_symbol, next_state
                                ),
                            ]
                        )

    def _encode_factor_compact_bfs_symmetry(self, factor: int, size: int) -> None:
        """Adapt Zakirzyanov et al. (LATA 2019), Section 3.2, to factors.

        Prefixes count entries strictly before their zero-based index. The
        preprint's (10) needs its first recurrence, (12) needs conjunction,
        (13)/(15) need true empty prefixes, and (15) refers to q+1.
        Keep (9)'s Boolean recurrence: with exactly-one parents its asserted
        endpoint enforces nondecreasing parent indices. This is an adaptation
        of the intended predicates, not a reproduction of DFA-Inductor 2 or
        its additional Section 3.3 pruning constraints.
        """
        alphabet_size = len(self.target.alphabet)

        def conjunction(output: int, left: int, right: int) -> None:
            self.cnf.extend([[-output, left], [-output, right],
                             [output, -left, -right]])

        for state in range(1, size):
            parents = [self.enumeration_parent(factor, parent, state)
                       for parent in range(state)]
            self.cnf.append(parents)
            # (8): a linear-size AMO, rather than a quadratic pairwise AMO.
            self.cnf.extend(CardEnc.atmost(parents, bound=1, vpool=self.pool,
                                         encoding=EncType.seqcounter).clauses)
            prefix = self.var("sb-nt", factor, state, 0)
            self.cnf.append([prefix])
            for parent, selected in enumerate(parents):
                edge = self.transition_relation(factor, parent, state)
                transitions = [self.factor_transition(factor, parent, symbol, state)
                               for symbol in range(alphabet_size)]
                self.cnf.append([-edge, *transitions])
                self.cnf.extend([[edge, -transition] for transition in transitions])
                self.cnf.extend([[-selected, edge], [-selected, prefix]])
                if self.bfs_parent_redundancy:
                    self.cnf.append([-edge, -prefix, selected])
                next_prefix = self.var("sb-nt", factor, state, parent + 1)
                conjunction(next_prefix, prefix, -edge)
                prefix = next_prefix

                no_edge = self.var("sb-ne", factor, parent, state, 0)
                no_label = self.var("sb-zm", factor, parent, state, 0)
                self.cnf.extend([[no_edge], [no_label]])
                for symbol, transition in enumerate(transitions):
                    label = self.var("sb-compact-label", factor, parent, state, symbol)
                    conjunction(label, transition, no_edge)
                    if symbol + 1 < alphabet_size:
                        next_no_edge = self.var("sb-ne", factor, parent, state, symbol + 1)
                        next_no_label = self.var("sb-zm", factor, parent, state, symbol + 1)
                        conjunction(next_no_edge, no_edge, -transition)
                        conjunction(next_no_label, no_label, -label)
                        no_edge, no_label = next_no_edge, next_no_label

        for state in range(1, size - 1):
            # (9), with the implicit current diagonal parent fixed false.
            next_order = self.var("sb-ng", factor, state, state)
            self.cnf.append([next_order])
            for parent in reversed(range(state)):
                current = self.enumeration_parent(factor, parent, state)
                following = self.enumeration_parent(factor, parent, state + 1)
                equal = self.var("sb-eq", factor, state, parent)
                order = self.var("sb-ng", factor, state, parent)
                # equal <-> (current <-> following)
                self.cnf.extend([[-equal, -current, following],
                                 [-equal, current, -following],
                                 [equal, current, following],
                                 [equal, -current, -following]])
                # order <-> (next_order & equal) | (current & !following)
                self.cnf.extend([[-order, next_order, current],
                                 [-order, next_order, -following],
                                 [-order, equal, current],
                                 [-order, equal, -following],
                                 [order, -next_order, -equal],
                                 [order, -current, following]])
                next_order = order
                for symbol in range(alphabet_size):
                    label = self.var("sb-compact-label", factor, parent, state, symbol)
                    no_earlier_label = self.var("sb-zm", factor, parent, state + 1, symbol)
                    self.cnf.append([-current, -following, -label, no_earlier_label])
            self.cnf.append([next_order])

    def _encode_automata(self) -> None:
        n_states = self.target.n_states
        alphabet_size = len(self.target.alphabet)

        for state in range(n_states):
            for symbol in range(alphabet_size):
                _one_hot(
                    self.cnf,
                    [
                        self.repaired_transition(state, symbol, target)
                        for target in range(n_states)
                    ],
                )

        for factor, size in enumerate(self.factor_sizes):
            for state in range(size):
                for symbol in range(alphabet_size):
                    _one_hot(
                        self.cnf,
                        [
                            self.factor_transition(factor, state, symbol, target)
                            for target in range(size)
                        ],
                    )

            self._encode_factor_bfs_symmetry(factor, size)

    def _encode_edits(self) -> None:
        for state, row in enumerate(self.target.transitions):
            for symbol, original_target in enumerate(row):
                edit = self.var("E-transition", state, symbol)
                unchanged = self.repaired_transition(state, symbol, original_target)
                # edit <-> not unchanged
                self.cnf.append([edit, unchanged])
                self.cnf.append([-edit, -unchanged])
                self.edit_literals.append(edit)

        for state, originally_accepting in enumerate(self.target.accepting):
            edit = self.var("E-acceptance", state)
            accepting = self.repaired_accepting(state)
            if originally_accepting:
                # edit <-> not accepting
                self.cnf.append([edit, accepting])
                self.cnf.append([-edit, -accepting])
            else:
                # edit <-> accepting
                self.cnf.append([-edit, accepting])
                self.cnf.append([edit, -accepting])
            self.edit_literals.append(edit)

    def _encode_exact_product(self) -> None:
        n_states = self.target.n_states
        first_size, second_size = self.factor_sizes
        alphabet_size = len(self.target.alphabet)

        # The initial states of the repaired target and both factors are 0.
        self.cnf.append([self.product_relation(0, 0, 0)])

        # Acceptance agrees at every reachable product triple.
        for state in range(n_states):
            target_accepting = self.repaired_accepting(state)
            for first in range(first_size):
                first_accepting = self.factor_accepting(0, first)
                for second in range(second_size):
                    relation = self.product_relation(state, first, second)
                    second_accepting = self.factor_accepting(1, second)
                    self.cnf.append([-relation, -target_accepting, first_accepting])
                    self.cnf.append([-relation, -target_accepting, second_accepting])
                    self.cnf.append(
                        [
                            -relation,
                            target_accepting,
                            -first_accepting,
                            -second_accepting,
                        ]
                    )

        if self.product_encoding == "direct":
            self._encode_direct_product_closure()
            return

        # Factored closure (R -> U -> W -> R) is logically the same as the
        # direct five-literal implication, but grows additively in the three
        # automaton sizes instead of multiplicatively in their destinations.
        for state in range(n_states):
            for first in range(first_size):
                for second in range(second_size):
                    relation = self.product_relation(state, first, second)
                    for symbol in range(alphabet_size):
                        for target_state in range(n_states):
                            auxiliary = self.var(
                                "U", target_state, first, second, symbol
                            )
                            self.cnf.append(
                                [
                                    -relation,
                                    -self.repaired_transition(
                                        state, symbol, target_state
                                    ),
                                    auxiliary,
                                ]
                            )

        for target_state in range(n_states):
            for first in range(first_size):
                for second in range(second_size):
                    for symbol in range(alphabet_size):
                        auxiliary = self.var("U", target_state, first, second, symbol)
                        for next_first in range(first_size):
                            next_auxiliary = self.var(
                                "W", target_state, next_first, second, symbol
                            )
                            self.cnf.append(
                                [
                                    -auxiliary,
                                    -self.factor_transition(
                                        0, first, symbol, next_first
                                    ),
                                    next_auxiliary,
                                ]
                            )

        for target_state in range(n_states):
            for next_first in range(first_size):
                for second in range(second_size):
                    for symbol in range(alphabet_size):
                        auxiliary = self.var(
                            "W", target_state, next_first, second, symbol
                        )
                        for next_second in range(second_size):
                            self.cnf.append(
                                [
                                    -auxiliary,
                                    -self.factor_transition(
                                        1, second, symbol, next_second
                                    ),
                                    self.product_relation(
                                        target_state, next_first, next_second
                                    ),
                                ]
                            )

    def _encode_direct_product_closure(self) -> None:
        """Close R with five-literal clauses, without U/W auxiliaries."""

        n_states = self.target.n_states
        first_size, second_size = self.factor_sizes
        for state in range(n_states):
            for first in range(first_size):
                for second in range(second_size):
                    relation = self.product_relation(state, first, second)
                    for symbol in range(len(self.target.alphabet)):
                        for target_state in range(n_states):
                            repaired_edge = self.repaired_transition(state, symbol, target_state)
                            for next_first in range(first_size):
                                first_edge = self.factor_transition(0, first, symbol, next_first)
                                for next_second in range(second_size):
                                    self.cnf.append([
                                        -relation,
                                        -repaired_edge,
                                        -first_edge,
                                        -self.factor_transition(1, second, symbol, next_second),
                                        self.product_relation(target_state, next_first, next_second),
                                    ])

    def _encode_witness(self, direction: int) -> None:
        """Encode a finite or pumpable directional strictness witness."""

        first_size, second_size = self.factor_sizes
        alphabet_size = len(self.target.alphabet)
        product_size = first_size * second_size
        if self.strictness_mode == "finite":
            min_length = 0
            max_length = product_size - 1
            if alphabet_size == 0:
                max_length = 0
        else:
            # A language recognized by an N-state DFA is infinite iff it has
            # an accepted word of length N..2N-1.  Here the directional
            # difference DFA is the n1*n2-state product of the two factors.
            min_length = product_size
            max_length = 2 * product_size - 1
            if alphabet_size == 0:
                self.cnf.append([])
                return

        for position in range(max_length + 1):
            _one_hot(
                self.cnf,
                [
                    self.var("w-state", direction, 0, position, state)
                    for state in range(first_size)
                ],
            )
            _one_hot(
                self.cnf,
                [
                    self.var("w-state", direction, 1, position, state)
                    for state in range(second_size)
                ],
            )

        self.cnf.append([self.var("w-state", direction, 0, 0, 0)])
        self.cnf.append([self.var("w-state", direction, 1, 0, 0)])

        for position in range(max_length):
            symbols = [
                self.var("w-symbol", direction, position, symbol)
                for symbol in range(alphabet_size)
            ]
            _one_hot(self.cnf, symbols)
            for symbol, symbol_literal in enumerate(symbols):
                for factor, size in enumerate(self.factor_sizes):
                    for state in range(size):
                        current = self.var(
                            "w-state", direction, factor, position, state
                        )
                        for next_state in range(size):
                            self.cnf.append(
                                [
                                    -current,
                                    -symbol_literal,
                                    -self.factor_transition(
                                        factor, state, symbol, next_state
                                    ),
                                    self.var(
                                        "w-state",
                                        direction,
                                        factor,
                                        position + 1,
                                        next_state,
                                    ),
                                ]
                            )

        witness_prefixes: list[int] = []
        for position in range(min_length, max_length + 1):
            observed_acceptance: list[int] = []
            for factor, size in enumerate(self.factor_sizes):
                observed = self.var("w-accepting", direction, factor, position)
                observed_acceptance.append(observed)
                for state in range(size):
                    active = self.var("w-state", direction, factor, position, state)
                    accepting = self.factor_accepting(factor, state)
                    # active -> (observed <-> accepting)
                    self.cnf.append([-active, -observed, accepting])
                    self.cnf.append([-active, observed, -accepting])

            is_witness = self.var("w-prefix", direction, position)
            self.cnf.append([-is_witness, observed_acceptance[direction]])
            self.cnf.append([-is_witness, -observed_acceptance[1 - direction]])
            witness_prefixes.append(is_witness)

        self.cnf.append(witness_prefixes)

    def _encode_nonempty_witness(self) -> None:
        """Reach an accepting A' state; exact equality makes both factors accept."""

        state_count = self.target.n_states
        alphabet_size = len(self.target.alphabet)
        max_length = state_count - 1 if alphabet_size else 0
        for position in range(max_length + 1):
            _one_hot(
                self.cnf,
                [self.var("nonempty-state", position, state) for state in range(state_count)],
            )
        self.cnf.append([self.var("nonempty-state", 0, 0)])

        for position in range(max_length):
            symbols = [
                self.var("nonempty-symbol", position, symbol)
                for symbol in range(alphabet_size)
            ]
            _one_hot(self.cnf, symbols)
            for symbol, symbol_literal in enumerate(symbols):
                for state in range(state_count):
                    active = self.var("nonempty-state", position, state)
                    for next_state in range(state_count):
                        self.cnf.append(
                            [
                                -active,
                                -symbol_literal,
                                -self.repaired_transition(state, symbol, next_state),
                                self.var("nonempty-state", position + 1, next_state),
                            ]
                        )

        witness_prefixes: list[int] = []
        for position in range(max_length + 1):
            is_witness = self.var("nonempty-prefix", position)
            for state in range(state_count):
                self.cnf.append(
                    [
                        -is_witness,
                        -self.var("nonempty-state", position, state),
                        self.repaired_accepting(state),
                    ]
                )
            witness_prefixes.append(is_witness)
        self.cnf.append(witness_prefixes)

    def _build(self) -> None:
        self._encode_automata()
        self._encode_edits()
        self._encode_exact_product()
        if self.encode_strictness:
            self._encode_witness(0)
            self._encode_witness(1)
        self._encode_nonempty_witness()

    def _decode_dfa(
        self,
        model_true: set[int],
        size: int,
        transition_variable: Callable[[int, int, int], int],
        acceptance_variable: Callable[[int], int],
    ) -> DFA:
        transitions: dict[tuple[int, Any], int] = {}
        labels: dict[int, bool] = {}
        for state in range(size):
            labels[state] = acceptance_variable(state) in model_true
            for symbol_index, symbol in enumerate(self.target.alphabet):
                chosen = [
                    target
                    for target in range(size)
                    if transition_variable(state, symbol_index, target) in model_true
                ]
                if len(chosen) != 1:
                    raise RuntimeError("SAT model did not encode a deterministic DFA")
                transitions[state, symbol] = chosen[0]

        return DFA(
            start=0,
            inputs=self.target.alphabet,
            outputs={False, True},
            states=tuple(range(size)),
            label=labels.__getitem__,
            transition=lambda state, symbol: transitions[state, symbol],
        )

    def decode(self, model: Sequence[int]) -> tuple[DFA, tuple[DFA, DFA]]:
        model_true = {literal for literal in model if literal > 0}
        repaired = self._decode_dfa(
            model_true,
            self.target.n_states,
            self.repaired_transition,
            self.repaired_accepting,
        )
        factors = tuple(
            self._decode_dfa(
                model_true,
                size,
                lambda state, symbol, target, factor=factor: self.factor_transition(
                    factor, state, symbol, target
                ),
                lambda state, factor=factor: self.factor_accepting(factor, state),
            )
            for factor, size in enumerate(self.factor_sizes)
        )
        return repaired, (factors[0], factors[1])

    def block_factors(self, model: Sequence[int]) -> list[int]:
        """Return one clause blocking exactly the decoded ordered factor pair."""

        model_true = {literal for literal in model if literal > 0}
        clause: list[int] = []
        for factor, size in enumerate(self.factor_sizes):
            for state in range(size):
                accepting = self.factor_accepting(factor, state)
                clause.append(-accepting if accepting in model_true else accepting)
                for symbol in range(len(self.target.alphabet)):
                    for target in range(size):
                        transition = self.factor_transition(
                            factor, state, symbol, target
                        )
                        if transition in model_true:
                            clause.append(-transition)
                            break
        return clause


def _shortest_directional_witness(
    accepted_by: DFA, rejected_by: DFA, alphabet: Sequence[Any]
) -> Word | None:
    start = (accepted_by.start, rejected_by.start)
    queue = deque([(start, ())])
    visited = {start}
    while queue:
        (left, right), word = queue.popleft()
        if bool(accepted_by._label(left)) and not bool(rejected_by._label(right)):
            return word
        for symbol in alphabet:
            successor = (
                accepted_by._transition(left, symbol),
                rejected_by._transition(right, symbol),
            )
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, word + (symbol,)))
    return None


def _shortest_common_witness(
    first: DFA, second: DFA, alphabet: Sequence[Any]
) -> Word | None:
    """Find a word accepted by both factors, including the empty word."""

    start = (first.start, second.start)
    queue = deque([(start, ())])
    visited = {start}
    while queue:
        (left, right), word = queue.popleft()
        if bool(first._label(left)) and bool(second._label(right)):
            return word
        for symbol in alphabet:
            successor = (
                first._transition(left, symbol),
                second._transition(right, symbol),
            )
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, word + (symbol,)))
    return None


def _infinite_directional_witness(
    accepted_by: DFA, rejected_by: DFA, alphabet: Sequence[Any]
) -> Word | None:
    """Return a pumpable mismatch word exactly when the difference is infinite.

    The product has at most ``N = |A1|*|A2|`` states.  Its mismatch language is
    infinite iff it accepts a word whose length is between ``N`` and
    ``2*N-1``.  Tracking a product state separately at each depth is necessary:
    revisiting a state is precisely what certifies a pumpable loop.
    """

    product_size = len(tuple(accepted_by.states())) * len(
        tuple(rejected_by.states())
    )
    if not alphabet:
        return None

    start = (accepted_by.start, rejected_by.start)
    frontier: dict[tuple[Any, Any], Word] = {start: ()}
    for length in range(2 * product_size):
        if length >= product_size:
            for (left, right), word in frontier.items():
                if bool(accepted_by._label(left)) and not bool(
                    rejected_by._label(right)
                ):
                    return word
        if length == 2 * product_size - 1:
            break
        next_frontier: dict[tuple[Any, Any], Word] = {}
        for (left, right), word in frontier.items():
            for symbol in alphabet:
                successor = (
                    accepted_by._transition(left, symbol),
                    rejected_by._transition(right, symbol),
                )
                next_frontier.setdefault(successor, word + (symbol,))
        frontier = next_frontier
    return None


def _equality_counterexample(
    repaired: DFA, factors: tuple[DFA, DFA], alphabet: Sequence[Any]
) -> Word | None:
    first, second = factors
    start = (repaired.start, first.start, second.start)
    queue = deque([(start, ())])
    visited = {start}
    while queue:
        (target_state, first_state, second_state), word = queue.popleft()
        target_value = bool(repaired._label(target_state))
        product_value = bool(first._label(first_state)) and bool(
            second._label(second_state)
        )
        if target_value != product_value:
            return word
        for symbol in alphabet:
            successor = (
                repaired._transition(target_state, symbol),
                first._transition(first_state, symbol),
                second._transition(second_state, symbol),
            )
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, word + (symbol,)))
    return None


def _describe_edits(
    target: _NormalizedDFA, repaired: DFA
) -> tuple[tuple[TransitionEdit, ...], tuple[AcceptanceEdit, ...]]:
    transition_edits: list[TransitionEdit] = []
    acceptance_edits: list[AcceptanceEdit] = []
    for state_index, original_state in enumerate(target.original_states):
        for symbol_index, symbol in enumerate(target.alphabet):
            old_index = target.transitions[state_index][symbol_index]
            new_index = repaired._transition(state_index, symbol)
            if old_index != new_index:
                transition_edits.append(
                    TransitionEdit(
                        state=original_state,
                        symbol=symbol,
                        old_target=target.original_states[old_index],
                        new_target=target.original_states[new_index],
                    )
                )
        old_value = target.accepting[state_index]
        new_value = bool(repaired._label(state_index))
        if old_value != new_value:
            acceptance_edits.append(
                AcceptanceEdit(
                    state=original_state,
                    old_value=old_value,
                    new_value=new_value,
                )
            )
    return tuple(transition_edits), tuple(acceptance_edits)


class _FixedRepairEncoding(_DirectEncoding):
    """Decomposition feasibility without edit variables or edit constraints."""

    def _encode_edits(self) -> None:
        # The repair is chosen externally and fixed by assumptions. The
        # unchanged automata/equality/witness clauses still apply.
        pass

    def _encode_nonempty_witness(self) -> None:
        # Empty repairs are rejected by reachability before SAT is called.
        # Exact equality transfers nonemptiness to the factor intersection.
        pass

    def repair_assumptions(self, repaired: _NormalizedDFA) -> list[int]:
        assumptions = [
            self.repaired_transition(state, symbol, destination)
            for state, row in enumerate(repaired.transitions)
            for symbol, destination in enumerate(row)
        ]
        assumptions.extend(
            self.repaired_accepting(state)
            if accepting else -self.repaired_accepting(state)
            for state, accepting in enumerate(repaired.accepting)
        )
        return assumptions


def _has_nonempty_language(target: _NormalizedDFA) -> bool:
    """Return whether an accepting state is reachable from initial state 0."""

    reached = {0}
    queue = deque([0])
    while queue:
        state = queue.popleft()
        if target.accepting[state]:
            return True
        for successor in target.transitions[state]:
            if successor not in reached:
                reached.add(successor)
                queue.append(successor)
    return False


def _repair_options(
    target: _NormalizedDFA,
) -> tuple[tuple[int, int | None, tuple[int | bool, ...]], ...]:
    """Editable cells and their possible values different from the input."""

    transitions = tuple(
        (state, symbol, tuple(destination for destination in range(target.n_states)
                              if destination != original))
        for state, row in enumerate(target.transitions)
        for symbol, original in enumerate(row)
        if target.n_states > 1
    )
    acceptance = tuple(
        (state, None, (not accepting,))
        for state, accepting in enumerate(target.accepting)
    )
    return transitions + acceptance


def _repairs_at_distance(
    target: _NormalizedDFA, distance: int,
) -> Iterator[_NormalizedDFA]:
    """Generate one BFS distance layer without a queue or duplicate repairs.

    Each cell differs from the original at most once. A transition cell has
    one alternative for each other destination; an acceptance cell can only
    be flipped. Selecting exactly k cells therefore enumerates every repair
    at shortest edit distance k exactly once, including unreachable states.
    """

    options = _repair_options(target)
    if distance == 0:
        yield target
        return
    for cells in combinations(options, distance):
        for values in product(*(alternatives for _, _, alternatives in cells)):
            transitions = [list(row) for row in target.transitions]
            accepting = list(target.accepting)
            for (state, symbol, _alternatives), value in zip(cells, values):
                if symbol is None:
                    accepting[state] = bool(value)
                else:
                    transitions[state][symbol] = int(value)
            yield replace(target, transitions=tuple(map(tuple, transitions)),
                          accepting=tuple(accepting))


def _is_kissat_solver(name: str) -> bool:
    aliases = getattr(SolverNames, "kissat404", ())
    return name.lower() in {alias.lower() for alias in aliases}


def _inspect_encoded_model(
    normalized: _NormalizedDFA,
    encoding: _DirectEncoding,
    model: Sequence[int],
    optimum: int,
    *,
    strictness: bool,
    strictness_mode: StrictnessMode,
    optimization: OptimizationBackend,
    blocked_models: int,
    checked_models: int,
    solver_calls: int,
    costs_seen: Sequence[int],
) -> tuple[DecompositionResult | None, list[int] | None, int, int]:
    """Decode and validate a model, or return a factor-blocking clause."""

    checked_models += 1
    repaired, factors = encoding.decode(model)

    counterexample = _equality_counterexample(
        repaired, factors, normalized.alphabet
    )
    if counterexample is not None:
        raise RuntimeError(
            "internal SAT equality encoding error; counterexample: "
            f"{counterexample!r}"
        )

    common_witness = _shortest_common_witness(
        factors[0], factors[1], normalized.alphabet
    )
    if common_witness is None or not bool(repaired.label(common_witness)):
        raise RuntimeError("internal SAT nonempty intersection encoding error")

    if strictness_mode == "finite":
        first_witness = _shortest_directional_witness(
            factors[0], factors[1], normalized.alphabet
        )
        second_witness = _shortest_directional_witness(
            factors[1], factors[0], normalized.alphabet
        )
    else:
        first_witness = _infinite_directional_witness(
            factors[0], factors[1], normalized.alphabet
        )
        second_witness = _infinite_directional_witness(
            factors[1], factors[0], normalized.alphabet
        )
    if first_witness is not None and second_witness is not None:
        transition_edits, acceptance_edits = _describe_edits(normalized, repaired)
        edit_distance = len(transition_edits) + len(acceptance_edits)
        if edit_distance != optimum:
            raise RuntimeError(
                "decoded edit count does not match the optimization optimum"
            )
        return (
            DecompositionResult(
                repaired=repaired,
                factors=factors,
                edit_distance=edit_distance,
                transition_edits=transition_edits,
                acceptance_edits=acceptance_edits,
                witnesses=(first_witness, second_witness),
                common_witness=common_witness,
                bfs_parent_redundancy=encoding.bfs_parent_redundancy,
                bfs_encoding=encoding.bfs_encoding,
                product_encoding=encoding.product_encoding,
                base_encoding_variables=encoding.base_encoding_variables,
                base_encoding_clauses=encoding.base_encoding_clauses,
                base_encoding_literals=encoding.base_encoding_literals,
                base_encoding_max_clause_length=encoding.base_encoding_max_clause_length,
                strictness_strategy="witness" if strictness else "blocking",
                strictness_mode=strictness_mode,
                optimization_backend=optimization,
                symmetry_breaking="bfs",
                blocked_models=blocked_models,
                checked_models=checked_models,
                solver_calls=solver_calls,
                costs_seen=tuple(costs_seen),
            ),
            None,
            blocked_models,
            checked_models,
        )

    if strictness:
        raise RuntimeError("internal SAT strictness encoding error")

    blocked_models += 1
    return (
        None,
        encoding.block_factors(model),
        blocked_models,
        checked_models,
    )




def _find_enumeration_decomposition(
    normalized: _NormalizedDFA,
    pairs: Sequence[tuple[int, int]],
    *,
    strictness: bool,
    strictness_mode: StrictnessMode,
    bfs_parent_redundancy: bool,
    bfs_encoding: BfsEncoding,
    product_encoding: ProductEncoding,
    sat_solver: str,
    progress: Callable[[str], None] | None = None,
) -> DecompositionResult | None:
    """Search edit-distance layers externally and check fixed repairs in SAT."""

    # A one-state complete DFA has language empty or universal. With a
    # nonempty intersection, it cannot be one of two strict factors.
    pairs = tuple(pair for pair in pairs if min(pair) > 1)
    if not pairs:
        return None
    maximum_distance = len(_repair_options(normalized))
    attempts: list[FactorSizeAttempt] = []
    solver_calls = checked_models = blocked_models = candidates_checked = 0
    empty_candidates_skipped = 0
    distances_seen: list[int] = []
    for distance in range(maximum_distance + 1):
        distances_seen.append(distance)
        for pair_index, pair in enumerate(pairs, start=1):
            started = last_update = perf_counter()
            pair_candidates = pair_empty_skipped = 0
            if progress is not None:
                progress(
                    f"[edit-distance {distance}/{maximum_distance}] "
                    f"[factor-pair {pair_index}/{len(pairs)}] "
                    f"checking fixed repairs for {pair[0]}x{pair[1]} with SAT"
                )
            encoding = _FixedRepairEncoding(
                normalized, pair, strictness, strictness_mode, bfs_parent_redundancy,
                product_encoding,
                bfs_encoding,
            )
            result = None
            # A solver is reused within this pair and distance layer. Every
            # call fixes the whole repair; no bound is expressed in SAT.
            with Solver(name=sat_solver, bootstrap_with=encoding.cnf.clauses) as solver:
                for repaired in _repairs_at_distance(normalized, distance):
                    pair_candidates += 1
                    candidates_checked += 1
                    if progress is not None and perf_counter() - last_update >= 10:
                        progress(
                            f"[edit-distance {distance}] {pair[0]}x{pair[1]}: "
                            f"checked {pair_candidates} repairs; "
                            f"skipped {pair_empty_skipped} empty repairs"
                        )
                        last_update = perf_counter()
                    if not _has_nonempty_language(repaired):
                        pair_empty_skipped += 1
                        empty_candidates_skipped += 1
                        continue
                    assumptions = encoding.repair_assumptions(repaired)
                    while True:
                        solver_calls += 1
                        if not solver.solve(assumptions=assumptions):
                            break
                        model = solver.get_model()
                        if model is None:
                            raise RuntimeError("SAT solver returned no model after SAT")
                        result, blocking_clause, blocked_models, checked_models = (
                            _inspect_encoded_model(
                                normalized, encoding, model, distance,
                                strictness=strictness,
                                strictness_mode=strictness_mode,
                                optimization="enumeration",
                                blocked_models=blocked_models,
                                checked_models=checked_models,
                                solver_calls=solver_calls,
                                costs_seen=distances_seen,
                            )
                        )
                        if result is not None:
                            break
                        assert blocking_clause is not None
                        # Strictness depends only on the factor languages;
                        # a rejected pair remains invalid for other repairs.
                        solver.add_clause(blocking_clause)
                    if result is not None:
                        break
            elapsed = perf_counter() - started
            attempts.append(FactorSizeAttempt(
                factor_sizes=pair,
                edit_distance=None if result is None else distance,
                elapsed_seconds=elapsed,
                edit_budget=distance,
                repair_candidates_checked=pair_candidates,
                empty_repair_candidates_skipped=pair_empty_skipped,
            ))
            if progress is not None:
                outcome = "no solution" if result is None else f"cost={distance}"
                progress(
                    f"[edit-distance {distance}] {pair[0]}x{pair[1]}: {outcome}; "
                    f"checked {pair_candidates} repairs; skipped {pair_empty_skipped} "
                    f"empty repairs in {elapsed:.3f}s"
                )
            if result is not None:
                return replace(
                    result,
                    repair_candidates_checked=candidates_checked,
                    empty_repair_candidates_skipped=empty_candidates_skipped,
                    factor_size_attempts=tuple(attempts),
                    factor_size_search_complete=True,
                    stopped_on_zero=distance == 0,
                )
    return None


def _find_minimum_edit_decomposition_for_sizes(
    target: DFA,
    factor_sizes: tuple[int, int],
    *,
    strictness: bool = True,
    strictness_mode: StrictnessMode = "finite",
    bfs_parent_redundancy: bool = True,
    bfs_encoding: BfsEncoding = "classic",
    product_encoding: ProductEncoding = "factored",
    optimization: OptimizationBackend = "maxsat",
    maxsat_factory: Callable[..., Any] = RC2,
    sat_solver: str = "g3",
    sat_budget: int | None = None,
    progress: Callable[[str], None] | None = None,
    statistics: dict[str, int] | None = None,
) -> DecompositionResult | None:
    """Find an exact, strict, minimum-edit two-factor decomposition.

    ``strictness=True`` encodes both bounded witness paths directly in SAT, as
    described in ``Strictness Note.pdf``.  ``strictness_mode="finite"`` asks
    only for a word in each directional language difference, while
    ``strictness_mode="infinite"`` requires both differences to be infinite.
    ``strictness=False`` deliberately omits the witness clauses: decoded
    factor pairs that fail the selected mathematical strictness mode are
    checked and blocked one by one at the same edit budget.

    With ``optimization="maxsat"``, target edits are soft unit clauses and
    Exact MaxSAT minimizes changed transitions plus flipped acceptance labels.
    With ``optimization="sat"``, ordinary SAT is called for increasing bounds
    ``k = 0, 1, ...`` with a cardinality constraint over the same edit
    literals; the first strict solution is therefore minimum-edit.  Factor
    sizes are exact, not upper bounds.
    """

    if not isinstance(bfs_parent_redundancy, bool):
        raise ValueError("bfs_parent_redundancy must be true or false")
    _validate_bfs_encoding(bfs_encoding)
    if len(factor_sizes) != 2 or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1
        for size in factor_sizes
    ):
        raise ValueError("factor_sizes must contain exactly two positive integers")

    normalized = _normalize_target(target)
    optimization = normalize_optimization(optimization)
    _validate_product_encoding(product_encoding)
    if strictness_mode not in ("finite", "infinite"):
        raise ValueError("strictness_mode must be 'finite' or 'infinite'")
    if not isinstance(sat_solver, str) or not sat_solver:
        raise ValueError("sat_solver must be a non-empty PySAT solver name")
    if optimization in ("sat", "enumeration") and _is_kissat_solver(sat_solver):
        raise ValueError(
            "Kissat is non-incremental and is not supported by the SAT "
            "optimization loop; use Glucose, CaDiCaL, or Maple"
        )
    if sat_budget is not None:
        if optimization != "sat":
            raise ValueError("sat_budget is only valid with optimization='sat'")
        if (
            not isinstance(sat_budget, int)
            or isinstance(sat_budget, bool)
            or sat_budget < 0
        ):
            raise ValueError("sat_budget must be a non-negative integer or None")

    if optimization == "enumeration":
        return _find_enumeration_decomposition(
            normalized, (factor_sizes,), strictness=strictness,
            strictness_mode=strictness_mode, bfs_parent_redundancy=bfs_parent_redundancy,
            bfs_encoding=bfs_encoding,
            product_encoding=product_encoding,
            sat_solver=sat_solver, progress=progress,
        )


    blocked_models = 0
    checked_models = 0
    solver_calls = 0
    costs_seen: list[int] = []
    if statistics is not None:
        statistics.update(solver_calls=0, checked_models=0, blocked_models=0)
    encoding = _DirectEncoding(
        normalized,
        factor_sizes,
        encode_strictness=strictness,
        strictness_mode=strictness_mode,
        bfs_parent_redundancy=bfs_parent_redundancy,
        bfs_encoding=bfs_encoding,
        product_encoding=product_encoding,
    )

    def inspect_model(
        model: Sequence[int], optimum: int
    ) -> tuple[DecompositionResult | None, list[int] | None]:
        nonlocal blocked_models, checked_models
        result, blocking_clause, blocked_models, checked_models = (
            _inspect_encoded_model(
                normalized,
                encoding,
                model,
                optimum,
                strictness=strictness,
                strictness_mode=strictness_mode,
                optimization=optimization,
                blocked_models=blocked_models,
                checked_models=checked_models,
                solver_calls=solver_calls,
                costs_seen=costs_seen,
            )
        )
        if statistics is not None:
            statistics.update(checked_models=checked_models, blocked_models=blocked_models)
        return result, blocking_clause

    if optimization == "maxsat":
        formula = WCNF()
        for clause in encoding.cnf.clauses:
            formula.append(clause)
        for edit_literal in encoding.edit_literals:
            formula.append([-edit_literal], weight=1)

        with maxsat_factory(formula) as optimizer:
            while True:
                solver_calls += 1
                if statistics is not None:
                    statistics["solver_calls"] = solver_calls
                model = optimizer.compute()
                if model is None:
                    return None

                optimum = int(optimizer.cost)
                costs_seen.append(optimum)

                result, blocking_clause = inspect_model(model, optimum)
                if result is not None:
                    # inspect_model captures solver_calls by value when it
                    # constructs the frozen result, after this compute call.
                    return result
                assert blocking_clause is not None
                optimizer.add_clause(blocking_clause)

    maximum_cost = len(encoding.edit_literals)
    if sat_budget is not None:
        if sat_budget > maximum_cost:
            return None
        budgets: Iterable[int] = (sat_budget,)
    else:
        budgets = range(maximum_cost + 1)
    permanent_blocks: list[list[int]] = []
    for budget in budgets:
        costs_seen.append(budget)
        cardinality = CardEnc.atmost(
            lits=encoding.edit_literals,
            bound=budget,
            vpool=encoding.pool,
            encoding=EncType.seqcounter,
        )
        clauses = [
            *encoding.cnf.clauses,
            *permanent_blocks,
            *cardinality.clauses,
        ]
        with Solver(name=sat_solver, bootstrap_with=clauses) as solver:
            while True:
                solver_calls += 1
                if statistics is not None:
                    statistics["solver_calls"] = solver_calls
                if not solver.solve():
                    break
                model = solver.get_model()
                if model is None:
                    raise RuntimeError("SAT solver returned no model after SAT")
                result, blocking_clause = inspect_model(model, budget)
                if result is not None:
                    return result
                assert blocking_clause is not None
                solver.add_clause(blocking_clause)
                permanent_blocks.append(blocking_clause)
    return None


def find_minimum_edit_decomposition(
    target: DFA,
    factor_sizes: tuple[int, int] | None = None,
    *,
    strictness: bool = True,
    strictness_mode: StrictnessMode = "finite",
    bfs_parent_redundancy: bool | None = None,
    bfs_encoding: BfsEncoding = "classic",
    product_encoding: ProductEncoding = "factored",
    optimization: OptimizationBackend = "maxsat",
    maxsat_factory: Callable[..., Any] = RC2,
    sat_solver: str = "g3",
    factor_order: FactorSizeOrder = "sum",
    progress: Callable[[str], None] | None = None,
    bfs_converse: bool | None = None,
    bfs_parent_strengthening: bool | None = None,
) -> DecompositionResult | None:
    """Find a globally best strict minimum-edit decomposition.

    When ``factor_sizes`` is supplied, solve that exact-size subproblem for
    backwards compatibility.  When it is omitted, examine every unique pair
    ``2 <= n1 <= n2 < |A|``.  ``factor_order="sum"`` uses increasing
    ``n1+n2``; ``factor_order="max"`` primarily minimizes ``max(n1,n2)`` and
    uses the sum as its tie-breaker. ``factor_order="product"`` minimizes
    ``n1*n2`` first and uses the sum as its tie-breaker. A zero-edit result is returned
    immediately.  Otherwise every pair is examined and the result is selected
    lexicographically by edit distance and the selected factor-size order.
    Exact MaxSAT optimizes each pair and then compares their optima.
    Baseline ordinary SAT iterates edit budgets in the outer loop and factor
    pairs in the inner loop, stopping at the first satisfiable pair.
    With ``strictness=False``, automatic ordinary SAT search uses external
    strictness checks and blocks invalid ordered factor pairs before advancing.
    Nonemptiness remains encoded. Automatic blocking search is currently
    supported by the SAT backend; fixed-size blocking also supports MaxSAT
    and external repair enumeration.
    ``optimization="enumeration"`` instead enumerates repaired DFAs externally
    by increasing edit distance, then asks SAT for a decomposition of each
    fixed repair. It uses no edit variables, cardinality constraints, or
    soft clauses. Distance, factor-pair order, and candidate repair order
    are the successive search levels.
    Permanent BFS symmetry breaking keeps one breadth-first state
    numbering for each accessible factor. ``bfs_parent_redundancy=False``
    omits only the redundant parent clauses for experiments.
    ``bfs_encoding="compact"`` uses the complete prefix predicates adapted
    from Zakirzyanov et al. (2019), Section 3.2. The default is ``"classic"``.
    ``product_encoding="factored"`` uses U/W auxiliaries and three-literal
    closure clauses; ``"direct"`` uses five-literal closure clauses without
    U/W. Both encode the same exact language equality.
    ``progress`` receives human-readable updates.
    """

    bfs_parent_redundancy = _resolve_bfs_parent_redundancy(
        bfs_parent_redundancy, bfs_converse, bfs_parent_strengthening,
    )
    optimization = normalize_optimization(optimization)
    _validate_product_encoding(product_encoding)
    _validate_bfs_encoding(bfs_encoding)
    if factor_order not in ("sum", "max", "product"):
        raise ValueError("factor_order must be 'sum', 'max', or 'product'")
    if strictness_mode not in ("finite", "infinite"):
        raise ValueError("strictness_mode must be 'finite' or 'infinite'")

    if factor_sizes is not None:
        if progress is not None:
            progress(f"[factor-pair 1/1] solving {factor_sizes[0]}x{factor_sizes[1]}")
        pair_started = perf_counter()
        result = _find_minimum_edit_decomposition_for_sizes(
            target,
            factor_sizes,
            strictness=strictness,
            strictness_mode=strictness_mode,
            optimization=optimization,
            maxsat_factory=maxsat_factory,
            sat_solver=sat_solver,
            bfs_parent_redundancy=bfs_parent_redundancy,
            bfs_encoding=bfs_encoding,
            product_encoding=product_encoding,
            progress=progress if optimization == "enumeration" else None,
        )
        pair_elapsed = perf_counter() - pair_started
        if progress is not None:
            outcome = (
                "no solution" if result is None else f"cost={result.edit_distance}"
            )
            progress(
                f"[factor-pair 1/1] finished {factor_sizes[0]}x{factor_sizes[1]}: "
                f"{outcome} in {pair_elapsed:.3f}s"
            )
        return result

    if not strictness and optimization != "sat":
        raise ValueError(
            "automatic factor-size search with blocking requires optimization='sat'"
        )
    if optimization not in ("maxsat", "sat", "enumeration"):
        raise ValueError("optimization must be 'maxsat', 'sat', or 'enumeration'")
    if not isinstance(sat_solver, str) or not sat_solver:
        raise ValueError("sat_solver must be a non-empty PySAT solver name")
    if optimization in ("sat", "enumeration") and _is_kissat_solver(sat_solver):
        raise ValueError(
            "Kissat is non-incremental and is not supported by the SAT "
            "optimization loop; use Glucose, CaDiCaL, or Maple"
        )

    normalized = _normalize_target(target)
    if optimization == "enumeration":
        return _find_enumeration_decomposition(
            normalized,
            factor_size_pairs(normalized.n_states, ordering=factor_order),
            strictness=True, strictness_mode=strictness_mode,
            bfs_parent_redundancy=bfs_parent_redundancy, sat_solver=sat_solver, progress=progress,
            bfs_encoding=bfs_encoding,
            product_encoding=product_encoding,
        )
    attempts: list[FactorSizeAttempt] = []
    best: DecompositionResult | None = None
    total_blocked_models = 0
    total_checked_models = 0
    total_solver_calls = 0
    all_costs_seen: list[int] = []

    def with_search_metadata(
        result: DecompositionResult,
        *,
        complete: bool,
        stopped_on_zero: bool,
    ) -> DecompositionResult:
        return replace(
            result,
            blocked_models=total_blocked_models,
            checked_models=total_checked_models,
            solver_calls=total_solver_calls,
            costs_seen=tuple(all_costs_seen),
            factor_size_attempts=tuple(attempts),
            factor_size_search_complete=complete,
            stopped_on_zero=stopped_on_zero,
        )

    pairs = factor_size_pairs(normalized.n_states, ordering=factor_order)

    if optimization == "sat":
        maximum_cost = normalized.n_states * (len(normalized.alphabet) + 1)


        for budget in range(maximum_cost + 1):
            all_costs_seen.append(budget)
            if progress is not None:
                progress(
                    f"[edit-budget {budget}/{maximum_cost}] "
                    f"checking {len(pairs)} factor pairs with SAT"
                )

            for pair_index, pair in enumerate(pairs, start=1):
                if progress is not None:
                    progress(
                        f"[edit-budget {budget}] "
                        f"[factor-pair {pair_index}/{len(pairs)}] "
                        f"solving {pair[0]}x{pair[1]} with SAT"
                    )
                pair_started = perf_counter()
                attempt_statistics: dict[str, int] = {}
                result = _find_minimum_edit_decomposition_for_sizes(
                    target,
                    pair,
                    strictness=strictness,
                    strictness_mode=strictness_mode,
                    optimization="sat",
                    sat_solver=sat_solver,
                    sat_budget=budget,
                    bfs_parent_redundancy=bfs_parent_redundancy,
                    bfs_encoding=bfs_encoding,
                    product_encoding=product_encoding,
                    statistics=attempt_statistics if not strictness else None,
                )
                pair_elapsed = perf_counter() - pair_started
                attempts.append(
                    FactorSizeAttempt(
                        factor_sizes=pair,
                        edit_distance=(
                            None if result is None else result.edit_distance
                        ),
                        elapsed_seconds=pair_elapsed,
                        edit_budget=budget,
                    )
                )
                if progress is not None:
                    outcome = (
                        "no solution"
                        if result is None
                        else f"cost={result.edit_distance}"
                    )
                    progress(
                        f"[edit-budget {budget}] "
                        f"[factor-pair {pair_index}/{len(pairs)}] "
                        f"finished {pair[0]}x{pair[1]}: "
                        f"{outcome} in {pair_elapsed:.3f}s"
                    )

                if result is None:
                    if strictness:
                        # Encoded strictness needs one call for this bound.
                        total_solver_calls += 1
                    else:
                        # Lazy checking can reject many models before UNSAT;
                        # retain those counts even when no result is returned.
                        total_solver_calls += attempt_statistics["solver_calls"]
                        total_checked_models += attempt_statistics["checked_models"]
                        total_blocked_models += attempt_statistics["blocked_models"]
                    continue

                total_blocked_models += result.blocked_models
                total_checked_models += result.checked_models
                total_solver_calls += result.solver_calls
                if progress is not None:
                    progress(
                        f"[factor-size search] first SAT result at "
                        f"edit budget {budget}: {pair[0]}x{pair[1]}; stopping"
                    )
                return with_search_metadata(
                    result,
                    complete=True,
                    stopped_on_zero=result.edit_distance == 0,
                )

        if progress is not None:
            progress("[factor-size search] no solution found")
        return None

    for pair_index, pair in enumerate(pairs, start=1):
        if progress is not None:
            progress(
                f"[factor-pair {pair_index}/{len(pairs)}] "
                f"solving {pair[0]}x{pair[1]} with {optimization.upper()}"
            )
        pair_started = perf_counter()
        result = _find_minimum_edit_decomposition_for_sizes(
            target,
            pair,
            strictness=True,
            strictness_mode=strictness_mode,
            optimization="maxsat",
            maxsat_factory=maxsat_factory,
            sat_solver=sat_solver,
            bfs_parent_redundancy=bfs_parent_redundancy,
            bfs_encoding=bfs_encoding,
            product_encoding=product_encoding,
        )
        pair_elapsed = perf_counter() - pair_started
        if progress is not None:
            outcome = (
                "no solution" if result is None else f"cost={result.edit_distance}"
            )
            progress(
                f"[factor-pair {pair_index}/{len(pairs)}] "
                f"finished {pair[0]}x{pair[1]}: {outcome} in {pair_elapsed:.3f}s"
            )
        attempts.append(
            FactorSizeAttempt(
                factor_sizes=pair,
                edit_distance=None if result is None else result.edit_distance,
                elapsed_seconds=pair_elapsed,
            )
        )
        if result is None:
            # An UNSAT MaxSAT subproblem is
            # established by one RC2 compute call.
            total_solver_calls += 1
            continue

        total_blocked_models += result.blocked_models
        total_checked_models += result.checked_models
        total_solver_calls += result.solver_calls
        all_costs_seen.extend(result.costs_seen)
        if result.edit_distance == 0:
            if progress is not None:
                progress(
                    f"[factor-size search] stopping on zero cost at "
                    f"{pair[0]}x{pair[1]}"
                )
            return with_search_metadata(
                result,
                complete=False,
                stopped_on_zero=True,
            )

        result_key = (
            result.edit_distance,
            *_factor_size_order_key(pair, factor_order),
        )
        if best is None:
            best = result
        else:
            best_pair = tuple(len(factor.states()) for factor in best.factors)
            best_key = (
                best.edit_distance,
                *_factor_size_order_key(best_pair, factor_order),
            )
            if result_key < best_key:
                best = result

    if best is None:
        if progress is not None:
            progress("[factor-size search] no solution found")
        return None
    if progress is not None:
        best_pair = tuple(len(factor.states()) for factor in best.factors)
        progress(
            f"[factor-size search] selected {best_pair[0]}x{best_pair[1]} "
            f"with cost={best.edit_distance}"
        )
    return with_search_metadata(best, complete=True, stopped_on_zero=False)


__all__ = [
    "ENCODING_REVISION",
    "ENUMERATION_ENCODING_REVISION",
    "encoding_revision",
    "AcceptanceEdit",
    "DecompositionResult",
    "FactorSizeAttempt",
    "FactorSizeOrder",
    "OptimizationBackend",
    "ProductEncoding",
    "normalize_optimization",
    "StrictnessMode",
    "SymmetryBreaking",
    "TransitionEdit",
    "factor_size_pairs",
    "find_minimum_edit_decomposition",
]
