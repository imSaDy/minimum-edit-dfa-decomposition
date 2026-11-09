"""Controlled protocol, frozen-corpus identity, and four-way aggregation."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compact_bfs_campaign.py"
spec = importlib.util.spec_from_file_location("compact_bfs_campaign", SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def reference():
    base = {"optimization": "sat", "sat_solver": "g3", "factor_sizes": None,
            "factor_order": "max", "strictness": True, "strictness_mode": "finite",
            "product_encoding": "factored", "bfs_parent_redundancy": True}
    return {"campaign_id": "reference", "immutable": {"options": {
        "state_counts": [3], "workers": 13, "timeout_seconds": 28800, "repetitions": 1,
        "memory_gib_per_test": 16, "minimum_available_ram_gib": 32,
        "variants": [{"id": "T01", "configuration": base}, {"id": "T08", "configuration": {
            **base, "bfs_parent_redundancy": False}}]}, "corpus": [
        {"input_kind": "minimal", "case_id": "a", "states": 3, "input_dot_sha256": "abc"},
        {"input_kind": "nonminimal", "case_id": "b", "states": 3, "input_dot_sha256": "def"}]}}


class CompactCampaignTests(unittest.TestCase):
    def test_two_new_scenarios_change_only_bfs_and_converse(self):
        ref = reference()
        plan = c.make_plan(ref)
        self.assertEqual(plan["execution"]["variant_sequence"], ["T18", "T19"])
        for old, new in zip(plan["variants"][:2], plan["variants"][2:]):
            old_config, new_config = dict(old["configuration"]), dict(new["configuration"])
            self.assertEqual(old_config.pop("bfs_encoding"), "classic")
            self.assertEqual(new_config.pop("bfs_encoding"), "compact")
            self.assertEqual(old_config, new_config)
        changed = copy.deepcopy(ref)
        changed["immutable"]["options"]["variants"][0]["configuration"]["sat_solver"] = "m22"
        with self.assertRaisesRegex(ValueError, "matching baseline"):
            c.make_plan(changed)

    def test_comparison_rejects_changed_inputs_limits_and_duplicate_ids(self):
        ref = reference()
        c.assert_same_corpus(ref, copy.deepcopy(ref))
        for alteration in ("hash", "timeout", "duplicate"):
            changed = copy.deepcopy(ref)
            if alteration == "hash":
                changed["immutable"]["corpus"][0]["input_dot_sha256"] = "different"
            elif alteration == "timeout":
                changed["immutable"]["options"]["timeout_seconds"] = 600
            else:
                changed["immutable"]["corpus"].append(changed["immutable"]["corpus"][0])
            with self.assertRaises(ValueError):
                c.assert_same_corpus(ref, changed)

    def test_timeout_denominators_and_joint_cohorts(self):
        rows = {}
        for case in ("a", "b"):
            kind = "minimal" if case == "a" else "nonminimal"
            for variant in c.VARIANTS:
                rows[(kind, case, 1, variant)] = {"status": "sat", "edit_distance": 1,
                    "factor_sizes": [2, 2], "paper": {"wall_seconds": 2 if case == "a" else 100,
                                                       "kernel_peak_rss_bytes": 1024**3}}
        rows[("nonminimal", "b", 1, "T19")]["status"] = "timeout"
        summaries, details = c.summarize(reference(), rows)
        for summary in summaries:
            self.assertEqual(summary["joint_sat_count"], 1)
            self.assertEqual(summary["mean_wall_seconds"], 2)
            self.assertEqual(summary["joint_terminal_count"], 2)
            self.assertEqual(summary["mean_kernel_peak_rss_gib"], 1)
        self.assertEqual(summaries[-1]["completion_percent"], 50)
        self.assertEqual(len(details), 8)
        rows[("minimal", "a", 1, "T18")]["edit_distance"] = 2
        with self.assertRaisesRegex(ValueError, "disagreement"):
            c.summarize(reference(), rows)

    def test_missing_results_produce_pending_counts_and_no_fabricated_means(self):
        summaries, _ = c.summarize(reference(), {})
        self.assertTrue(all(s["pending"] == 2 and s["mean_wall_seconds"] is None for s in summaries))

    def test_server_adapter_is_guarded_and_idempotent(self):
        source = ('argv = ["--factor-order", c["factor_order"], "--sat-solver", c["sat_solver"],]\n'
                  'expected = {"product_encoding": c["product_encoding"],}\n')
        adapted = c.adapt_harness_source(source)
        compile(adapted, "harness.py", "exec")
        self.assertEqual(c.adapt_harness_source(adapted), adapted)
        self.assertIn('"--bfs-encoding"', adapted)
        self.assertIn('"bfs_encoding": c.get', adapted)
        with self.assertRaisesRegex(ValueError, "unexpected worker"):
            c.adapt_harness_source("unrecognized source")


if __name__ == "__main__":
    unittest.main()
