import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dfa_identify.cli import build_parser, main as cli_main
from dfa_identify.minimum_edit import ENCODING_REVISION

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_nonminimal_five_benchmark as nonminimal_runner
import run_all_small_dfaminer as all_small
from build_minimal_five_benchmark_report import build_parser as report_parser
from run_minimal_five_benchmark import (
    BenchmarkCase,
    DATASET_NAME,
    FACTOR_SIZE_SEARCH_MODE,
    SEARCH_ORDERS,
    _read_matching_result,
    build_parser as runner_parser,
    result_directory,
)


class BfsConfigurationTests(unittest.TestCase):
    def test_cli_rejects_disabling_bfs(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                build_parser().parse_args(["input.dot", "--symmetry-breaking", "none"])
        self.assertEqual(error.exception.code, 2)

    def test_json_defaults_to_permanent_bfs(self):
        payload = {
            "states": [0], "alphabet": ["a"], "start": 0,
            "accepting": [0],
            "transitions": [{"from": 0, "symbol": "a", "to": 0}],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            output = Path(directory) / "result.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            with patch("dfa_identify.cli.find_minimum_edit_decomposition",
                       return_value=None) as solve:
                self.assertEqual(cli_main([str(source), "--output", str(output)]), 0)
            self.assertNotIn("symmetry_breaking", solve.call_args.kwargs)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["symmetry_breaking"], "bfs")
            self.assertEqual(result["encoding_revision"], ENCODING_REVISION)

    def test_removed_json_settings_are_rejected(self):
        for option, value in (("symmetry_breaking", "none"),
                              ("max_edits", 0), ("incremental_sat", True)):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "input.json"
                source.write_text(json.dumps({
                    "states": [0], "alphabet": [], "start": 0,
                    "transitions": [], option: value,
                }), encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "removed configuration options"):
                    cli_main([str(source)])

    def test_cli_rejects_removed_options(self):
        for arguments in (("--max-edits", "0"), ("--incremental-sat",),
                          ("--no-incremental-sat",), ("--symmetry-breaking", "bfs")):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    build_parser().parse_args(["input.dot", *arguments])
                self.assertEqual(error.exception.code, 2)

    def test_runner_has_no_bfs_switch_but_reports_read_historical_modes(self):
        self.assertFalse(hasattr(runner_parser().parse_args([]), "symmetry_modes"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner_parser().parse_args(["--symmetry-modes", "bfs"])
        self.assertEqual(report_parser().parse_args([]).symmetry_modes, ("bfs",))
        self.assertEqual(
            report_parser().parse_args(["--symmetry-modes", "none", "bfs"]).symmetry_modes,
            ["none", "bfs"],
        )

    def test_resume_requires_current_encoding_but_historical_reading_still_works(self):
        case = BenchmarkCase("sample", Path("sample.dot"), 3, 3, 1, None, None)
        record = {"status": "sat", "benchmark": {
            "dataset": DATASET_NAME, "case_id": case.case_id,
            "input_state_count": 3, "optimization_backend": "sat",
            "strategy": "witness", "symmetry_breaking": "bfs",
            "factor_size_search_mode": FACTOR_SIZE_SEARCH_MODE,
            "factor_size_search_order": SEARCH_ORDERS["sat"],
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNotNone(_read_matching_result(path, case, "sat", "witness"))
            self.assertIsNone(_read_matching_result(
                path, case, "sat", "witness", require_current_encoding=True
            ))
            record["benchmark"]["encoding_revision"] = "permanent-bfs-v2"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNotNone(_read_matching_result(path, case, "sat", "witness"))
            self.assertIsNone(_read_matching_result(
                path, case, "sat", "witness", require_current_encoding=True
            ))
            record["benchmark"]["encoding_revision"] = ENCODING_REVISION
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNotNone(_read_matching_result(
                path, case, "sat", "witness", require_current_encoding=True
            ))

    def test_nonminimal_runner_passes_complete_bfs_configuration(self):
        case = BenchmarkCase("sample", Path("sample.dot"), 3, 3, 1, None, None)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(nonminimal_runner, "load_nonminimal_cases",
                              return_value=[case]):
                with patch.object(nonminimal_runner, "_run_one",
                                  return_value={"status": "sat"}) as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(nonminimal_runner.main([
                            "--output-root", directory,
                        ]), 0)
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["symmetry_breaking"], "bfs")
            self.assertEqual(kwargs["strictness_mode"], "finite")
            self.assertEqual(kwargs["factor_order"], "sum")
            self.assertEqual(kwargs["output_directory"], result_directory(
                Path(directory), case, "maxsat", "witness"
            ))

    def test_all_small_separates_historical_reporting_from_resume(self):
        case = all_small.InputCase("minimal", BenchmarkCase(
            "sample", Path("sample.dot"), 3, 3, 1, None, None
        ))
        record = {"status": "sat", "benchmark": {
            "dataset": all_small.DATASET_NAME, "case_id": case.case_id,
            "input_kind": case.kind, "input_state_count": 3,
            "nonempty_requirement": all_small.NONEMPTY_REQUIREMENT,
            "timeout_seconds": 30,
            "factor_size_search_mode": "all-proper-unordered",
            "factor_size_search_order": "edit-budget-then-factor-size",
            "strategy": "witness", "strictness_mode": "finite",
            "optimization_backend": "sat", "symmetry_breaking": "bfs",
            "factor_order": "max",
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNone(all_small.matching_result(path, case, 30, "max"))
            self.assertIsNotNone(all_small.matching_result(
                path, case, 30, "max", require_current_encoding=False
            ))
            record["benchmark"]["encoding_revision"] = "permanent-bfs-v2"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNone(all_small.matching_result(path, case, 30, "max"))
            self.assertIsNotNone(all_small.matching_result(
                path, case, 30, "max", require_current_encoding=False
            ))
            record["benchmark"]["encoding_revision"] = ENCODING_REVISION
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNotNone(all_small.matching_result(path, case, 30, "max"))


if __name__ == "__main__":
    unittest.main()
