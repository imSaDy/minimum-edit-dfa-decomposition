"""Oversubscribed workers require kernel limits; completed results stay intact."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from remote import dispatch_paper as d
from remote import monitor_paper as monitor
from remote import run_paper as h
from tests.test_paper_dispatch import Clock, FakeHarness, manifest


def override(m, workers=54):
    return {"schema": 1, "campaign_id": m["campaign_id"], "workers": workers,
            "cpu_ids": list(range(workers)), "memory_gib_per_test": 16,
            "minimum_available_ram_gib": 32, "aggregate_memory_max_gib": 208,
            "aggregate_memory_high_gib": 192, "aggregate_swap_max_gib": 0}


class Budget:
    high = 192 * h.GIB
    initial_oom_kills = 0

    def state(self):
        return 0, 0


class PaperResourceTests(unittest.TestCase):
    def test_override_preserves_frozen_limits_and_rejects_wrong_campaign_or_cpus(self):
        m = manifest()
        original = json.dumps(m, sort_keys=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = override(m)
            h.save(root / "runtime-resources.json", value)
            self.assertEqual(d.resource_settings(root, m)["workers"], 54)
            for name, invalid in (("memory_gib_per_test", 12), ("minimum_available_ram_gib", 1),
                                  ("campaign_id", "other"), ("cpu_ids", [0] * 54)):
                h.save(root / "runtime-resources.json", value | {name: invalid})
                with self.assertRaises(ValueError):
                    d.resource_settings(root, m)
        self.assertEqual(json.dumps(m, sort_keys=True), original)

    def test_kernel_ceiling_must_be_real_and_match_all_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            (group / "memory.current").write_text("17")
            (group / "memory.events").write_text("oom 0\noom_kill 0\n")
            for name, value in (("memory.max", 208 * h.GIB), ("memory.high", 192 * h.GIB),
                                ("memory.swap.max", 0)):
                (group / name).write_text(str(value))
            resources = override(manifest())
            self.assertEqual(d.MemoryBudget(resources, h.GIB, group).state(), (17, 0))
            (group / "memory.max").write_text("max")
            with self.assertRaises(ValueError):
                d.MemoryBudget(resources, h.GIB, group)

    def test_unavailable_cpus_and_insufficient_host_ram_fail_closed(self):
        fake = FakeHarness(Clock())
        resources = override(manifest())
        with patch.object(d.os, "sched_getaffinity", return_value={0, 1}, create=True):
            with self.assertRaises(ValueError):
                d.validate_resources(resources, manifest()["immutable"]["options"], fake)
        with patch.object(d.os, "sched_getaffinity", return_value=set(range(64)), create=True), \
             patch.object(d, "MemoryBudget", return_value=Budget()), \
             patch.object(fake, "meminfo", return_value={"MemAvailable": 239 * h.GIB}):
            with self.assertRaises(ValueError):
                d.validate_resources(resources, manifest()["immutable"]["options"], fake)

    def test_all_54_slots_work_and_resume_leaves_completed_bytes_unchanged(self):
        m = manifest()
        m["immutable"]["corpus"] = [{"states": 4, "input_kind": "minimal", "case_id": f"case-{i}"}
                                    for i in range(60)]
        m["immutable"]["total_jobs"] = 120
        clock = Clock()
        fake = FakeHarness(clock)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(d.time, "monotonic", clock.monotonic), \
             patch.object(d.time, "sleep", clock.sleep), \
             patch.object(d, "validate_resources", return_value=Budget()):
            root = Path(directory)
            h.save(root / "runtime-resources.json", override(m))
            self.assertEqual(d.run_dispatch(root, m, fake), 0)
            self.assertEqual(fake.maximum_active, 54)
            self.assertEqual({cpu for _, cpu, _ in fake.starts}, set(range(54)))
            paths = list(root.glob("runs/*/*/*/*/*/result.json"))
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            starts = list(fake.starts)
            self.assertEqual(d.run_dispatch(root, m, fake), 0)
            self.assertEqual(fake.starts, starts)
            self.assertTrue(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
                                for p, sha in hashes.items()))
            progress = h.load(root / "progress.json")
            self.assertEqual(progress["runtime_resources"]["workers"], 54)
            self.assertEqual(monitor.runtime_resources(root, m)["cpu_ids"], list(range(54)))

    def test_aggregate_oom_after_reap_records_interruption_and_can_resume(self):
        m, clock = manifest(), Clock()
        fake = FakeHarness(clock)
        budget = Budget()
        reaped = set()
        original_reap = fake.reap

        def reap(a, blocking=False):
            if a["process"].pid in reaped:
                raise ChildProcessError("already reaped")
            usage = original_reap(a, blocking)
            if usage is not None:
                reaped.add(a["process"].pid)
                budget.state = lambda: (0, 1)
            return usage

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(d.time, "monotonic", clock.monotonic), \
             patch.object(d.time, "sleep", clock.sleep), \
             patch.object(d, "validate_resources", return_value=budget), \
             patch.object(fake, "reap", side_effect=reap):
            root = Path(directory)
            h.save(root / "runtime-resources.json", override(m, 2))
            self.assertEqual(d.run_dispatch(root, m, fake), 130)
            progress = h.load(root / "progress.json")
            self.assertIn("aggregate memory", progress["reason"])
            values = [h.load(p) for p in root.glob("runs/*/*/*/*/*/result.json")]
            self.assertEqual([v["status"] for v in values], ["interrupted", "interrupted"])
        self.assertEqual(len(reaped), 2)

    def test_high_memory_pauses_new_jobs_until_memory_falls(self):
        m, clock = manifest(), Clock()
        fake, budget = FakeHarness(clock), Budget()
        budget.state = lambda: (budget.high if clock.now < .5 else 0, 0)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(d.time, "monotonic", clock.monotonic), \
             patch.object(d.time, "sleep", clock.sleep), \
             patch.object(d, "validate_resources", return_value=budget):
            root = Path(directory)
            h.save(root / "runtime-resources.json", override(m, 2))
            self.assertEqual(d.run_dispatch(root, m, fake), 0)
            self.assertGreaterEqual(fake.starts[0][2], .5)
            self.assertEqual(len(fake.starts), 8)


if __name__ == "__main__":
    unittest.main()
