import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_all_small_dfaminer import InputCase, write_reports  # noqa: E402
import run_all_small_dfaminer as runner  # noqa: E402
from run_minimal_five_benchmark import BenchmarkCase  # noqa: E402


class AllSmallReportTests(unittest.TestCase):
    def test_timeout_only_retry_preserves_successes_and_resumes_without_repeating(self):
        def make_case(case_id):
            return InputCase("minimal", BenchmarkCase(
                case_id, Path(f"{case_id}.dot"), 3, 3, 1, None, None))

        def saved_attempt(case, status, limit):
            result = {"status": status, "benchmark": {
                "dataset": runner.DATASET_NAME,
                "encoding_revision": runner.ENCODING_REVISION,
                "case_id": case.case_id, "input_kind": case.kind,
                "nonempty_requirement": runner.NONEMPTY_REQUIREMENT,
                "input_dot_sha256": None, "input_state_count": 3,
                "timeout_seconds": limit,
                "factor_size_search_mode": "all-proper-unordered",
                "factor_size_search_order": runner.SEARCH_ORDERS["maxsat"],
                "strategy": "witness", "strictness_mode": "finite",
                "optimization_backend": "maxsat", "symmetry_breaking": "bfs",
                "factor_order": "max", "bfs_parent_redundancy": True,
                "wall_seconds": 5.0 if status == "sat" else limit,
            }}
            if status == "sat":
                result["edit_distance"] = 1
                result["common_accepting_witness"] = ["a"]
                result["strictness_witnesses"] = [[], ["b"]]
                result["factor_sizes"] = [2, 2]
            return result

        solved, short_timeout, long_timeout = map(
            make_case, ("solved", "short-timeout", "long-timeout"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "maxsat"
            originals = {}
            for case, status, limit in (
                (solved, "sat", 4500),
                (short_timeout, "timeout", 4500),
                (long_timeout, "timeout", 5400),
            ):
                path = runner.result_path(output, case)
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(saved_attempt(case, status, limit)), encoding="utf-8")
                originals[case.case_id] = path.read_bytes()
            short_path = runner.result_path(output, short_timeout)
            (short_path.parent / "solver.log").write_text("Original 75-minute attempt", encoding="utf-8")

            def finish_at_new_limit(*args, **kwargs):
                self.assertEqual(args[0], short_timeout.benchmark)
                self.assertEqual(args[7], 5400)
                archive = output / "timeout_retry_history/states_03/minimal/short-timeout"
                self.assertEqual(next(archive.glob("result-*.json")).read_bytes(),
                                 originals["short-timeout"])
                return saved_attempt(short_timeout, "timeout", 5400)

            arguments = [
                "--min-states", "3", "--max-states", "3", "--optimization", "maxsat",
                "--output-root", directory, "--timeout-seconds", "5400",
                "--retry-timeouts-only",
            ]
            with patch.object(runner, "scan_cases", return_value=[solved, short_timeout, long_timeout]):
                with patch.object(runner, "_run_one", side_effect=finish_at_new_limit) as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(runner.main(arguments), 0)
                        self.assertEqual(runner.main(arguments), 0)
                    self.assertEqual(run.call_count, 1)
            for case in (solved, long_timeout):
                self.assertEqual(runner.result_path(output, case).read_bytes(),
                                 originals[case.case_id])
            retry = json.loads(short_path.read_text())["timeout_retry"]
            self.assertEqual(retry["previous_timeout_seconds"], 4500)
            self.assertEqual(retry["requested_timeout_seconds"], 5400)
            self.assertEqual(Path(retry["previous_result_archive"]).read_bytes(),
                             originals["short-timeout"])

    def test_maxsat_runner_configuration_and_output_isolation(self):
        case = InputCase("minimal", BenchmarkCase(
            "sample", Path("sample.dot"), 3, 3, 1, None, None))
        self.assertEqual(runner.build_parser().parse_args([]).optimization, "sat")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "scan_cases", return_value=[case]):
                with patch.object(runner, "_run_one", return_value={"status": "unsat"}) as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(runner.main([
                            "--min-states", "3", "--max-states", "7",
                            "--optimization", "maxsat", "--factor-order", "max",
                            "--bfs-parent-redundancy", "--output-root", directory,
                        ]), 0)
            args = run.call_args.args
            self.assertEqual(args[3:7], ("maxsat", "bfs", "finite", "max"))
            self.assertTrue(run.call_args.kwargs["bfs_parent_redundancy"])
            self.assertEqual(args[1], Path(directory) / "maxsat/runs/states_03/minimal/sample")
            manifest = json.loads((Path(directory) / "maxsat/input_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["configuration"]["optimization"], "maxsat")
            self.assertIsNone(manifest["configuration"]["sat_solver"])

    def test_resume_distinguishes_sat_and_maxsat_searches(self):
        case = InputCase("minimal", BenchmarkCase(
            "sample", Path("sample.dot"), 3, 3, 1, None, None))
        payload = {"status": "unsat", "benchmark": {
            "dataset": runner.DATASET_NAME, "encoding_revision": runner.ENCODING_REVISION,
            "case_id": "sample", "input_kind": "minimal",
            "nonempty_requirement": runner.NONEMPTY_REQUIREMENT,
            "input_dot_sha256": None, "input_state_count": 3,
            "timeout_seconds": 30, "factor_size_search_mode": "all-proper-unordered",
            "factor_size_search_order": runner.SEARCH_ORDERS["maxsat"],
            "strategy": "witness", "strictness_mode": "finite",
            "optimization_backend": "maxsat", "symmetry_breaking": "bfs",
            "factor_order": "max", "bfs_parent_redundancy": True,
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(runner.matching_result(path, case, 30, "max"))
            self.assertIsNotNone(runner.matching_result(path, case, 30, "max", optimization="maxsat"))
            payload["benchmark"]["factor_size_search_order"] = runner.SEARCH_ORDERS["sat"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(runner.matching_result(path, case, 30, "max", optimization="maxsat"))

    def test_cost_distribution_handles_no_results_and_cost_two(self) -> None:
        case = InputCase(
            kind="minimal",
            benchmark=BenchmarkCase(
                case_id="sample",
                input_path=Path("sample.dot"),
                input_state_count=3,
                source_size=3,
                instance_id=1,
                dot_sha256=None,
                sample=None,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            write_reports([case], {}, output_root, 30, "min-cost", 3, 3)
            report = output_root / "reports" / "cost_distribution.csv"
            with report.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [(row["edit_distance"], row["count"]) for row in rows],
                [("0", "0"), ("1", "0"), ("2", "0")],
            )

            result = {
                "status": "sat",
                "edit_distance": 2,
                "factor_sizes": [2, 2],
                "common_accepting_witness": ["a"],
                "strictness_witnesses": [[], ["b"]],
                "visualizations": {
                    "A1": {"image": "A1.png"},
                    "A2": {"image": "A2.png"},
                },
                "benchmark": {"wall_seconds": 1.0},
            }
            write_reports(
                [case], {("minimal", "sample"): result},
                output_root, 30, "min-cost", 3, 3,
            )
            with report.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[2]["count"], "1")
            with (output_root / "reports" / "summary.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                summary = next(csv.DictReader(stream))
            self.assertEqual(summary["cost_2"], "1")
            with (output_root / "reports" / "cost_two_results.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                cost_two = list(csv.DictReader(stream))
            self.assertEqual(len(cost_two), 1)
            self.assertEqual(cost_two[0]["case_id"], "sample")
            self.assertEqual(cost_two[0]["A1_image"], "A1.png")
            self.assertEqual(cost_two[0]["A2_image"], "A2.png")

            write_reports(
                [case], {("minimal", "sample"): result},
                output_root, 30, "none", 3, 3,
            )
            with (output_root / "reports" / "top_five_overall.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            with (output_root / "reports" / "cost_two_results.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            selection = json.loads(
                (output_root / "reports" / "selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selection["rank_by"], "none")
            self.assertEqual(selection["selected_overall"], [])


if __name__ == "__main__":
    unittest.main()
