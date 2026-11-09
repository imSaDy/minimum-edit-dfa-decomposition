import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dfa_identify.dot_io import (
    dfa_from_dot,
    dfa_to_dot,
    write_decomposition_visualisations,
)
from dfa_identify.cli import build_parser
from dfa_identify.minimum_edit import find_minimum_edit_decomposition


NUMERIC_DOT = """\
digraph DFA {
  3 [shape=plaintext, label=""];
  3 -> 0 [label=""];
  0 [shape=circle];
  1 [shape=doublecircle];
  0 -> 0 [label=0];
  0 -> 1 [label=1];
  1 -> 0 [label=0];
  1 -> 1 [label=1];
}
"""


class DotIOTests(unittest.TestCase):
    def test_cli_exposes_sat_solver(self) -> None:
        args = build_parser().parse_args(
            ["input.dot", "--optimization", "sat", "--sat-solver", "cadical195"]
        )
        self.assertEqual(args.sat_solver, "cadical195")

    def test_numeric_labels_are_normalised_to_a_and_b(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.dot"
            path.write_text(NUMERIC_DOT, encoding="utf-8")
            dfa = dfa_from_dot(path)

        self.assertEqual(dfa.start, "0")
        self.assertEqual(dfa.inputs, frozenset({"a", "b"}))
        self.assertFalse(dfa._label("0"))
        self.assertTrue(dfa._label("1"))
        self.assertEqual(dfa._transition("0", "a"), "0")
        self.assertEqual(dfa._transition("0", "b"), "1")

    def test_visual_dot_has_only_alphabetic_transition_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.dot"
            path.write_text(NUMERIC_DOT, encoding="utf-8")
            rendered = dfa_to_dot(dfa_from_dot(path))

        self.assertIn('[label="a"]', rendered)
        self.assertIn('[label="b"]', rendered)
        self.assertNotIn('[label="0"]', rendered)
        self.assertNotIn('[label="1"]', rendered)
        self.assertNotIn("graph [label=", rendered)

    def test_dot_parser_is_safe_for_parallel_benchmark_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.dot"
            path.write_text(NUMERIC_DOT, encoding="utf-8")
            with ThreadPoolExecutor(max_workers=4) as executor:
                parsed = list(executor.map(dfa_from_dot, [path] * 12))

        self.assertTrue(all(dfa.inputs == frozenset({"a", "b"}) for dfa in parsed))

    @unittest.skipUnless(shutil.which("dot"), "Graphviz is not installed")
    def test_four_dot_and_png_visualisations_are_created(self) -> None:
        source = Path("examples/trailing_b_mod_3.dot")
        target = dfa_from_dot(source)
        result = find_minimum_edit_decomposition(
            target, (2, 2), strictness=True
        )
        self.assertIsNotNone(result)

        with tempfile.TemporaryDirectory() as directory:
            generated = write_decomposition_visualisations(
                target, result, Path(directory)
            )
            self.assertEqual(set(generated), {"A", "A_prime", "A1", "A2"})
            for name in generated:
                dot_path = Path(generated[name]["dot"])
                image_path = Path(generated[name]["image"])
                self.assertTrue(dot_path.is_file())
                self.assertTrue(image_path.is_file())
                self.assertGreater(image_path.stat().st_size, 0)
                dot_text = dot_path.read_text(encoding="utf-8")
                self.assertIn('[label="a"]', dot_text)
                self.assertIn('[label="b"]', dot_text)


if __name__ == "__main__":
    unittest.main()
