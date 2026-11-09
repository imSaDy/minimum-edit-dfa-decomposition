"""Protocol barriers, result integrity and scientific comparison safeguards."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from remote import run_paper as p


PLAN = Path(__file__).resolve().parents[1] / "remote/experiments/paper-20261005/verification-variants.json"


def job(variant="T01", states=3, order="max", strictness="finite"):
    return {"variant": {"id": variant, "configuration": {
        "strictness_mode": strictness, "factor_order": order}},
        "case": {"states": states, "case_id": "input", "input_kind": "minimal"},
        "repetition": 1}


def result(j, cost=1, sizes=(2, 2), status="sat"):
    return {"status": status, "edit_distance": cost, "factor_sizes": list(sizes),
            "paper": {"campaign_id": "frozen", "job": j}}


class PaperCampaignTests(unittest.TestCase):
    def test_capacity_requires_headroom_in_addition_to_combined_ceilings(self):
        options = {"workers": 13, "memory_gib_per_test": 16,
                   "minimum_available_ram_gib": 32}
        with patch.object(p, "cpu_slots", return_value=list(range(13))), \
             patch.object(p, "meminfo", return_value={"MemAvailable": 241 * p.GIB}):
            p.validate_capacity(options)
        with patch.object(p, "cpu_slots", return_value=list(range(13))), \
             patch.object(p, "meminfo", return_value={"MemAvailable": 239 * p.GIB}):
            with self.assertRaisesRegex(ValueError, "240.0 GiB"):
                p.validate_capacity(options)

    def test_declared_protocol_is_the_approved_eleven_variant_matrix(self):
        options = p.protocol_options(p.load(PLAN))
        self.assertEqual(options["state_counts"], list(range(3, 11)))
        self.assertEqual(options["timeout_seconds"], 28800)
        self.assertEqual(options["repetitions"], 1)
        self.assertEqual([v["id"] for v in options["variants"]],
                         ["T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T16", "T17"])
        changed = p.load(PLAN)
        changed["variants"][1]["rc2"]["adapt"] = True
        with self.assertRaisesRegex(ValueError, "default RC2"):
            p.protocol_options(changed)

    def test_phase_blocks_finish_state_then_variant_and_exclude_eleven(self):
        options = p.protocol_options(p.load(PLAN))
        options["state_counts"] = [3, 4]
        options["variants"] = options["variants"][:2]
        options["repetitions"] = 2
        manifest = {"immutable": {"options": options, "corpus": [
            {"states": n, "input_kind": "minimal", "case_id": f"n{n}-{i}"}
            for n in (3, 4, 11) for i in (1, 2)]}}
        blocks = list(p.jobs(manifest))
        self.assertEqual([(n, v) for n, v, _ in blocks],
                         [(3, "T01"), (3, "T02"), (4, "T01"), (4, "T02")])
        self.assertTrue(all(len(block) == 4 for _, _, block in blocks))
        self.assertTrue(all(j["case"]["states"] != 11 for _, _, b in blocks for j in b))

    def test_resume_accepts_timeouts_and_memory_limits_but_retries_errors(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "result.json"
            j = job()
            for status in ("sat", "unsat", "timeout", "memory_limit", "interrupted", "error"):
                p.save(path, result(j, status=status))
                self.assertEqual(p.reusable(path, j, "frozen") is not None, status in p.TERMINAL)
            with self.assertRaisesRegex(ValueError, "provenance"):
                p.reusable(path, j, "different")
            changed = copy.deepcopy(j)
            changed["variant"]["configuration"]["factor_order"] = "sum"
            with self.assertRaisesRegex(ValueError, "provenance"):
                p.reusable(path, changed, "frozen")

    def test_semantically_different_infinite_results_are_not_flagged(self):
        finite = result(job(), cost=1)
        infinite = result(job("T07", strictness="infinite"), cost=2, sizes=(2, 3))
        self.assertIsNone(p.disagreement({"f": finite, "i": infinite}))

    def test_finite_cost_or_existence_disagreement_stops_the_campaign(self):
        baseline = result(job(), cost=1)
        for alternative in (result(job("T02"), cost=2),
                            result(job("T02"), status="unsat", cost=None)):
            self.assertIsNotNone(p.disagreement({"a": baseline, "b": alternative}))

    def test_size_ties_are_compared_only_within_same_secondary_objective(self):
        baseline = result(job(), sizes=(3, 3))
        different_order = result(job("T16", order="sum"), sizes=(2, 4))
        self.assertIsNone(p.disagreement({"a": baseline, "b": different_order}))
        same_order = result(job("T02"), sizes=(2, 4))
        self.assertIsNotNone(p.disagreement({"a": baseline, "b": same_order}))

    def test_log_tail_is_bounded_for_long_eight_hour_progress_logs(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "solver.log"
            path.write_bytes(b"x" * 200000 + b"MemoryError")
            tail = p.log_tail(path)
            self.assertEqual(len(tail), 8000)
            self.assertTrue(tail.endswith("MemoryError"))


if __name__ == "__main__":
    unittest.main()
