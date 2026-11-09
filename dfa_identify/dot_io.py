"""DOT input and clean Graphviz visualisations for DFA experiments."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pydot
from dfa import DFA

from dfa_identify.minimum_edit import DecompositionResult


_PYDOT_PARSE_LOCK = threading.Lock()


def _unquote(value: Any) -> str:
    """Return a pydot identifier/attribute without DOT quoting."""

    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        try:
            return str(json.loads(text))
        except json.JSONDecodeError:
            return text[1:-1]
    return text


def _attribute(attributes: Mapping[str, Any], name: str) -> str | None:
    value = attributes.get(name)
    return None if value is None else _unquote(value)


def _normalise_symbol(symbol: str) -> str:
    """Use alphabetic names for numeric benchmark symbols."""

    stripped = symbol.strip()
    if stripped.isdigit():
        index = int(stripped)
        if 0 <= index < 26:
            return chr(ord("a") + index)
    return stripped


def _edge_symbols(label: str) -> tuple[str, ...]:
    symbols = tuple(
        _normalise_symbol(part)
        for part in label.split(",")
        if part.strip()
    )
    if not symbols:
        raise ValueError("transition edge has an empty label")
    return symbols


def dfa_from_dot(path: Path) -> DFA:
    """Parse a deterministic, complete DFA from a Graphviz DOT file.

    Initial states can be marked by an unlabelled edge from a point/plaintext
    helper node, ``style=bold``, or ``start=true``.  Accepting states can be
    marked using ``shape=doublecircle`` or ``peripheries=2``.  Numeric alphabet
    labels are normalised to letters (0 -> a, 1 -> b, ...).
    """

    try:
        # pydot lazily creates and caches a global pyparsing grammar.  That
        # initialisation is not thread-safe, while benchmark preparation runs
        # several independent DFAMiner processes concurrently.
        with _PYDOT_PARSE_LOCK:
            graphs = pydot.graph_from_dot_file(str(path), encoding="utf-8")
    except Exception as error:
        raise ValueError(f"could not parse DOT: {error}") from error
    if len(graphs) != 1:
        raise ValueError("DOT input must contain exactly one directed graph")
    graph = graphs[0]
    if graph.get_type() != "digraph":
        raise ValueError("DOT input must be a directed graph (digraph)")

    ignored_names = {"", "\\n", "node", "edge", "graph"}
    node_attributes: dict[str, dict[str, Any]] = {}
    for node in graph.get_nodes():
        name = _unquote(node.get_name())
        if name not in ignored_names:
            node_attributes[name] = node.get_attributes()

    pseudo_states = set()
    for name, attributes in node_attributes.items():
        shape = (_attribute(attributes, "shape") or "").lower()
        label = _attribute(attributes, "label")
        if name.startswith("__start") or (
            shape in {"none", "plaintext", "point"} and label in {None, ""}
        ):
            pseudo_states.add(name)

    raw_edges: list[tuple[str, str, str | None]] = []
    for edge in graph.get_edges():
        source = _unquote(edge.get_source())
        target = _unquote(edge.get_destination())
        label = _attribute(edge.get_attributes(), "label")
        raw_edges.append((source, target, label))
        if label in {None, ""}:
            pseudo_states.add(source)

    start_candidates = {
        target
        for source, target, label in raw_edges
        if source in pseudo_states and label in {None, ""}
    }
    for name, attributes in node_attributes.items():
        style = (_attribute(attributes, "style") or "").lower()
        start = (_attribute(attributes, "start") or "").lower()
        if "bold" in style or start in {"1", "true", "yes"}:
            start_candidates.add(name)

    transition_map: dict[tuple[str, str], str] = {}
    states = set(node_attributes) - pseudo_states
    alphabet = set()
    for source, target, label in raw_edges:
        if label in {None, ""}:
            if source not in pseudo_states:
                raise ValueError(
                    f"unlabelled edge {source!r} -> {target!r} is not a start edge"
                )
            continue
        if source in pseudo_states or target in pseudo_states:
            raise ValueError("start/helper nodes cannot be used in transitions")
        states.update((source, target))
        for symbol in _edge_symbols(label):
            key = (source, symbol)
            if key in transition_map:
                raise ValueError(f"duplicate transition for {key!r}")
            transition_map[key] = target
            alphabet.add(symbol)

    if not states:
        raise ValueError("DOT input contains no DFA states")
    if not alphabet:
        raise ValueError("DOT input contains no labelled transitions")
    if not start_candidates and "0" in states:
        start_candidates.add("0")
    if len(start_candidates) != 1:
        raise ValueError(
            "DOT input must identify exactly one initial state; found "
            f"{sorted(start_candidates)!r}"
        )
    start = next(iter(start_candidates))
    if start not in states:
        raise ValueError(f"initial state {start!r} is not a DFA state")

    accepting = set()
    for state in states:
        attributes = node_attributes.get(state, {})
        shape = (_attribute(attributes, "shape") or "").lower()
        peripheries = _attribute(attributes, "peripheries")
        if shape == "doublecircle" or peripheries == "2":
            accepting.add(state)

    missing = [
        (state, symbol)
        for state in sorted(states)
        for symbol in sorted(alphabet)
        if (state, symbol) not in transition_map
    ]
    if missing:
        raise ValueError(f"DFA must be complete; missing transitions: {missing!r}")

    frozen_states = frozenset(states)
    frozen_alphabet = frozenset(alphabet)
    return DFA(
        start=start,
        inputs=frozen_alphabet,
        outputs={False, True},
        states=frozen_states,
        label=lambda state: state in accepting,
        transition=lambda state, symbol: transition_map[state, symbol],
    )


def _stable_order(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(values, key=lambda value: (type(value).__name__, repr(value))))


def _state_order(dfa: DFA) -> tuple[Any, ...]:
    states = list(_stable_order(dfa.states()))
    if dfa.start in states:
        states.remove(dfa.start)
    return (dfa.start, *states)


def _dot_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def dfa_to_dot(dfa: DFA) -> str:
    """Create an undecorated DOT graph with alphabetic edge labels."""

    states = _state_order(dfa)
    alphabet = _stable_order(dfa.inputs)
    display_symbols = {
        symbol: chr(ord("a") + index) if index < 26 else str(symbol)
        for index, symbol in enumerate(alphabet)
    }
    node_ids = {state: f"q{index}" for index, state in enumerate(states)}

    lines = [
        "digraph DFA {",
        "  rankdir=LR;",
        '  __start [shape=point, label=""];',
        f"  __start -> {node_ids[dfa.start]};",
    ]
    for state in states:
        shape = "doublecircle" if bool(dfa._label(state)) else "circle"
        lines.append(
            f"  {node_ids[state]} [shape={shape}, label={_dot_string(state)}];"
        )
    for state in states:
        for symbol in alphabet:
            target = dfa._transition(state, symbol)
            lines.append(
                f"  {node_ids[state]} -> {node_ids[target]} "
                f"[label={_dot_string(display_symbols[symbol])}];"
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_decomposition_visualisations(
    target: DFA,
    result: DecompositionResult,
    output_directory: Path,
    *,
    image_format: str = "png",
) -> dict[str, dict[str, str]]:
    """Write clean DOT and rendered images for A, A', A1, and A2."""

    renderer = shutil.which("dot")
    if renderer is None:
        raise RuntimeError(
            "Graphviz 'dot' executable is required to render visualisations"
        )
    if not image_format.isalnum():
        raise ValueError("image format must be alphanumeric")

    output_directory.mkdir(parents=True, exist_ok=True)
    automata: Sequence[tuple[str, DFA]] = (
        ("A", target),
        ("A_prime", result.repaired),
        ("A1", result.factors[0]),
        ("A2", result.factors[1]),
    )
    generated: dict[str, dict[str, str]] = {}
    for name, automaton in automata:
        dot_path = output_directory / f"{name}.dot"
        image_path = output_directory / f"{name}.{image_format}"
        dot_path.write_text(dfa_to_dot(automaton), encoding="utf-8")
        subprocess.run(
            [renderer, f"-T{image_format}", str(dot_path), "-o", str(image_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        generated[name] = {
            "dot": str(dot_path),
            "image": str(image_path),
        }
    return generated


__all__ = [
    "dfa_from_dot",
    "dfa_to_dot",
    "write_decomposition_visualisations",
]
