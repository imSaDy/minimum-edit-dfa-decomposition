"""A slow job must not hold up free CPU slots or cause repeated results."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from remote import dispatch_paper as d
from remote import run_paper as h


PLAN = Path(__file__).resolve().parents[1] / "remote/experiments/paper-20261005/verification-variants.json"


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeHarness:
    GIB = h.GIB
    TERMINAL = h.TERMINAL

    def __init__(self, clock, durations=None):
        self.clock = clock
        self.starts = []
        self.ends = []
        self.durations = durations or {}
        self.maximum_active = 0
        self.running = set()

    def __getattr__(self, name):
        return getattr(h, name)

    def validate_capacity(self, options):
        pass

    def meminfo(self):
        return {"MemAvailable": 400 * h.GIB, "SwapFree": 64 * h.GIB}

    def start_job(self, output, manifest, job, cpu):
        key = (job["case"]["states"], job["variant"]["id"], job["case"]["case_id"])
        self.starts.append((key, cpu, self.clock.now))
        self.running.add(cpu)
        self.maximum_active = max(self.maximum_active, len(self.running))
        folder = output / h.job_relative(job)
        folder.mkdir(parents=True, exist_ok=True)
        return {"job": job, "folder": folder, "process": SimpleNamespace(pid=len(self.starts)),
                "cpu": cpu, "started": self.clock.now, "last_sample": 0.0,
                "last_rss": 0, "peak_rss": 0, "peak_pss": None, "samples": 0,
                "reason": None, "end": self.clock.now + self.durations.get(key, .2)}

    def sample_tree(self, pid):
        return 16 * 1024 ** 2, None

    def reap(self, active, blocking=False):
        if active["reason"] or self.clock.now >= active["end"]:
            self.running.discard(active["cpu"])
            return SimpleNamespace()
        return None

    def kill(self, active):
        active["end"] = self.clock.now

    def finish_job(self, active, usage, manifest):
        j = active["job"]
        key = (j["case"]["states"], j["variant"]["id"], j["case"]["case_id"])
        self.ends.append((key, active["cpu"], self.clock.now))
        return {"status": active["reason"] or "sat", "edit_distance": 0, "factor_sizes": [2, 2],
                "paper": {"campaign_id": manifest["campaign_id"], "job": j,
                          "worker": {"cpu_id": active["cpu"]}, "wall_seconds": self.clock.now - active["started"]}}


def manifest():
    options = h.protocol_options(h.load(PLAN))
    options.update(state_counts=[3, 4], workers=2, variants=options["variants"][:2], timeout_seconds=20)
    corpus = [{"states": n, "input_kind": "minimal", "case_id": f"n{n}-{name}"}
              for n in (3, 4) for name in ("a", "b")]
    return {"campaign_id": "test-campaign", "immutable": {
        "options": options, "cpu_ids": [0, 1], "corpus": corpus, "total_jobs": 8}}


class PaperDispatchTests(unittest.TestCase):
    def test_pending_queue_preserves_state_variant_order_and_skips_saved_jobs(self):
        m = manifest()
        all_jobs = [j for _, _, block in h.jobs(m) for j in block]
        retained = {h.job_relative(all_jobs[0]).as_posix(): {"status": "sat"}}
        queue = list(d.pending_jobs(m, retained, h))
        self.assertEqual(queue, all_jobs[1:])
        self.assertEqual(len({h.job_relative(j).as_posix() for j in queue}), 7)

    def test_free_slot_advances_variants_and_states_while_first_job_is_slow(self):
        clock = Clock()
        fake = FakeHarness(clock, {(3, "T01", "n3-a"): 3.0})
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(d.time, "monotonic", clock.monotonic), \
             patch.object(d.time, "sleep", clock.sleep):
            root = Path(directory)
            m = manifest()
            code = d.run_dispatch(root, m, fake, {"scheduler_id": "test-scheduler"})
            self.assertEqual(code, 0)
            self.assertEqual(len(fake.starts), 8)
            self.assertEqual(fake.maximum_active, 2)
            first_end = next(t for k, _, t in fake.ends if k == (3, "T01", "n3-a"))
            next_variant = next(t for k, _, t in fake.starts if k[:2] == (3, "T02"))
            next_state = next(t for k, _, t in fake.starts if k[0] == 4)
            self.assertLess(next_variant, first_end)
            self.assertLess(next_state, first_end)
            self.assertEqual([k for k, _, _ in fake.starts],
                             [(n, v, f"n{n}-{name}") for n in (3, 4) for v in ("T01", "T02") for name in ("a", "b")])
            original_starts = list(fake.starts)
            self.assertEqual(d.run_dispatch(root, m, fake, {"scheduler_id": "test-scheduler"}), 0)
            self.assertEqual(fake.starts, original_starts)
            self.assertEqual(h.load(root / "progress.json")["completed"], 8)

    def test_timeout_does_not_block_other_jobs_and_is_retained_for_resume(self):
        clock = Clock()
        fake = FakeHarness(clock, {(3, "T01", "n3-a"): 10.0})
        m = manifest()
        m["immutable"]["options"]["timeout_seconds"] = 1
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(d.time, "monotonic", clock.monotonic), \
             patch.object(d.time, "sleep", clock.sleep):
            root = Path(directory)
            self.assertEqual(d.run_dispatch(root, m, fake, {"scheduler_id": "test-scheduler"}), 0)
            values = [h.load(p) for p in root.glob("runs/*/*/*/*/*/result.json")]
            self.assertEqual(sum(v["status"] == "timeout" for v in values), 1)
            self.assertTrue(all(v["paper"]["scheduler_policy"] == d.POLICY for v in values))
            self.assertEqual(len(fake.starts), 8)


if __name__ == "__main__":
    unittest.main()
