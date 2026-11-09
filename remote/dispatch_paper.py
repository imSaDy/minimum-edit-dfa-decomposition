"""Keep fixed CPU slots busy using the frozen solver and an ordered job queue.

Scheduling revisions are saved separately from the immutable algorithm/input
manifest. Completed results from the original dispatcher remain valid.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

POLICY = "ordered-state-variant-queue-v3"


def resource_settings(output, manifest):
    """CPU scheduling can change without changing frozen solver options."""
    options = manifest["immutable"]["options"]
    path = output / "runtime-resources.json"
    if not path.exists():
        return {"workers": options["workers"], "cpu_ids": manifest["immutable"]["cpu_ids"],
                "memory_gib_per_test": options["memory_gib_per_test"],
                "minimum_available_ram_gib": options["minimum_available_ram_gib"]}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != 1 or value.get("campaign_id") != manifest["campaign_id"]:
        raise ValueError("resource override belongs to a different campaign or schema")
    for name in ("workers", "memory_gib_per_test", "minimum_available_ram_gib",
                 "aggregate_memory_max_gib", "aggregate_memory_high_gib"):
        if type(value.get(name)) is not int or value[name] <= 0:
            raise ValueError(f"invalid runtime resource: {name}")
    cpus = value.get("cpu_ids")
    if (not isinstance(cpus, list) or len(cpus) != value["workers"]
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
            or len(set(cpus)) != len(cpus)):
        raise ValueError("each runtime worker needs a distinct CPU slot")
    for name in ("memory_gib_per_test", "minimum_available_ram_gib"):
        if value[name] != options[name]:
            raise ValueError(f"resource override must preserve frozen {name}")
    if (value["aggregate_memory_high_gib"] >= value["aggregate_memory_max_gib"]
            or type(value.get("aggregate_swap_max_gib")) is not int
            or value["aggregate_swap_max_gib"] != 0):
        raise ValueError("aggregate limits require high < max and no swap")
    return value


class MemoryBudget:
    """Verify a real kernel memory ceiling before allowing oversubscription."""
    def __init__(self, resources, gib, group=None):
        if group is None:
            line = next(line for line in Path("/proc/self/cgroup").read_text().splitlines()
                        if line.startswith("0::"))
            group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
        self.group = Path(group)
        self.high = resources["aggregate_memory_high_gib"] * gib
        self.maximum = resources["aggregate_memory_max_gib"] * gib
        expected = {"memory.max": self.maximum, "memory.high": self.high, "memory.swap.max": 0}
        for name, limit in expected.items():
            if (self.group / name).read_text().strip() != str(limit):
                raise ValueError(f"kernel {name} does not match the requested campaign limit")
        self.initial_oom_kills = self.state()[1]

    def state(self):
        events = dict(line.split() for line in (self.group / "memory.events").read_text().splitlines())
        return int((self.group / "memory.current").read_text()), int(events.get("oom_kill", 0))


def validate_resources(resources, options, h):
    if "aggregate_memory_max_gib" not in resources:
        h.validate_capacity(options)
        return None
    if not set(resources["cpu_ids"]).issubset(os.sched_getaffinity(0)):
        raise ValueError("requested CPU slots are not available to this process")
    budget = MemoryBudget(resources, h.GIB)
    current, _ = budget.state()
    required = (resources["aggregate_memory_max_gib"] + resources["minimum_available_ram_gib"]) * h.GIB
    if required > h.meminfo()["MemAvailable"] + current:
        raise ValueError("aggregate memory ceiling plus host reserve exceeds available RAM")
    return budget


def frozen_harness(output):
    path = output / "snapshot/source/remote/run_paper.py"
    spec = importlib.util.spec_from_file_location("frozen_paper_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register_scheduler(output, manifest, h, existing_count, resources):
    source = Path(__file__).resolve()
    info = {"policy": POLICY, "controller_sha256": h.sha(source),
            "campaign_id": manifest["campaign_id"],
            "assignment_order": "state count, then declared variant, repetition, input order",
            "waiting_policy": "advance when a queue is assigned; earlier jobs may still run",
            "cpu_assignment": "reuse each fixed CPU slot as soon as its job ends",
            "runtime_resources": resources}
    identifier = h.digest(info)
    directory = output / "scheduling" / identifier
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "dispatch_paper.py"
    if snapshot.exists():
        if h.sha(snapshot) != info["controller_sha256"]:
            raise ValueError("saved dispatcher source changed")
    else:
        snapshot.write_bytes(source.read_bytes())
    h.save(directory / "runtime-resources.json", resources)
    info.update(scheduler_id=identifier, recorded_at_utc=h.utc(),
                completed_results_at_activation=existing_count,
                source=str(snapshot.relative_to(output)))
    if not (directory / "policy.json").exists():
        h.save(directory / "policy.json", info)
    h.save(output / "scheduler.json", info)
    return info


def pending_jobs(manifest, results, h):
    return deque(job for _, _, block in h.jobs(manifest) for job in block
                 if h.job_relative(job).as_posix() not in results)


def progress(output, manifest, results, active, pending, state, metadata, h, reason=None):
    statuses = Counter(v["status"] for v in results.values())
    batches = Counter((a["job"]["case"]["states"], a["job"]["variant"]["id"])
                      for a in active.values())
    next_job = pending[0] if pending else None
    next_batch = ({"states": next_job["case"]["states"], "variant": next_job["variant"]["id"]}
                  if next_job else None)
    h.save(output / "progress.json", {
        "state": state, "phase": next_batch, "reason": reason,
        "campaign_id": manifest["campaign_id"], "scheduler_policy": POLICY,
        "scheduler_id": metadata["scheduler_id"],
        "runtime_resources": metadata["runtime_resources"],
        "completed": sum(statuses[s] for s in h.TERMINAL),
        "total": manifest["immutable"]["total_jobs"], "statuses": dict(statuses),
        "pending_jobs": len(pending), "active_count": len(active),
        "active_batches": [{"states": n, "variant": variant, "active": count}
                           for (n, variant), count in sorted(batches.items())],
        "updated_at_utc": h.utc()})


def report(output, manifest, results, active, pending, state, metadata, h, reason=None):
    """Retain the original result grid; add scheduling provenance per row."""
    import csv
    h.report(output, manifest, results, state, reason=reason)
    path = output / "results.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or []) + ["scheduler_policy", "scheduler_id", "scheduler_workers",
                                                  "aggregate_memory_max_gib"]
        rows = list(reader)
    temporary = output / ".scheduled-results.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = results.get(Path(row["result"]).parent.as_posix())
            if value:
                paper = value["paper"]
                row["scheduler_policy"] = paper.get("scheduler_policy", "variant-barrier-v1")
                row["scheduler_id"] = paper.get("scheduler_id", "original-frozen-supervisor")
                resource = paper.get("runtime_resources", {})
                row["scheduler_workers"] = resource.get("workers", manifest["immutable"]["options"]["workers"])
                row["aggregate_memory_max_gib"] = resource.get("aggregate_memory_max_gib", "")
            writer.writerow(row)
    temporary.replace(path)
    progress(output, manifest, results, active, pending, state, metadata, h, reason)


def run_dispatch(output, manifest, h, metadata=None):
    options = manifest["immutable"]["options"]
    resources = resource_settings(output, manifest)
    budget = validate_resources(resources, options, h)
    cpus = resources["cpu_ids"]
    results = {}
    for _, _, block in h.jobs(manifest):
        for job in block:
            relative = h.job_relative(job)
            saved = h.reusable(output / relative / "result.json", job, manifest["campaign_id"])
            if saved:
                results[relative.as_posix()] = saved
    metadata = metadata or register_scheduler(output, manifest, h, len(results), resources)
    metadata.setdefault("runtime_resources", resources)
    pending = pending_jobs(manifest, results, h)
    active, stop_reason = {}, None
    index = {}
    for relative, value in results.items():
        case = value["paper"]["job"]["case"]
        index.setdefault((case["input_kind"], case["case_id"]), {})[relative] = value
    prior_problem = h.disagreement(results)
    if prior_problem:
        report(output, manifest, results, active, pending, "invalid", metadata, h, prior_problem)
        return 1
    report(output, manifest, results, active, pending, "running", metadata, h)
    print(f"[scheduler] {POLICY}; {len(results)} retained; {len(pending)} pending; "
          f"{len(cpus)} fixed CPU slots", flush=True)
    last_report = last_progress = time.monotonic()
    last_memory, last_batch = 0.0, None
    with (output / "server-memory.csv").open("a", encoding="utf-8", buffering=1) as memory_log:
        if memory_log.tell() == 0:
            memory_log.write("utc,states,variant,active_jobs,available_ram_bytes,tree_rss_bytes,swap_free_bytes\n")
        try:
            while pending or active:
                memory = h.meminfo()
                group_memory, oom_kills = budget.state() if budget else (0, 0)
                if budget and oom_kills > budget.initial_oom_kills:
                    stop_reason = "aggregate memory ceiling reached; unfinished jobs can be resumed"
                    raise KeyboardInterrupt
                if memory["MemAvailable"] < options["minimum_available_ram_gib"] * h.GIB:
                    stop_reason = "server RAM reserve reached; unfinished jobs can be resumed"
                    raise KeyboardInterrupt
                for cpu in cpus:
                    if budget and budget.state()[0] >= budget.high:
                        break  # Let running jobs finish before adding memory pressure.
                    if cpu not in active and pending:
                        job = pending.popleft()
                        batch = (job["case"]["states"], job["variant"]["id"])
                        if batch != last_batch:
                            print(f"[dispatch] states={batch[0]} variant={batch[1]} "
                                  f"remaining_queue={len(pending)}; earlier jobs may still be active", flush=True)
                            last_batch = batch
                        active[cpu] = h.start_job(output, manifest, job, cpu)
                total_rss = 0
                changed = False
                for cpu, a in list(active.items()):
                    now = time.monotonic()
                    rss = a["last_rss"]
                    if now - a["last_sample"] >= 1:
                        rss, pss = h.sample_tree(a["process"].pid)
                        a["last_rss"], a["last_sample"] = rss, now
                        a["samples"] += 1
                        a["peak_rss"] = max(a["peak_rss"], rss)
                        if pss is not None:
                            a["peak_pss"] = max(a["peak_pss"] or 0, pss)
                    total_rss += rss
                    usage = h.reap(a)
                    if usage is not None:
                        a["reaped_usage"] = usage
                    if budget and budget.state()[1] > budget.initial_oom_kills:
                        stop_reason = "aggregate memory ceiling reached; unfinished jobs can be resumed"
                        raise KeyboardInterrupt
                    if usage is None:
                        if now - a["started"] >= options["timeout_seconds"]:
                            a["reason"] = "timeout"
                            h.kill(a)
                        elif rss > options["memory_gib_per_test"] * h.GIB:
                            a["reason"] = "memory_limit"
                            h.kill(a)
                        continue
                    del active[cpu]
                    value = h.finish_job(a, usage, manifest)
                    value["paper"].update(scheduler_policy=POLICY, scheduler_id=metadata["scheduler_id"],
                                          runtime_resources=resources)
                    h.save(a["folder"] / "result.json", value)
                    relative = h.job_relative(a["job"]).as_posix()
                    results[relative] = value
                    case = a["job"]["case"]
                    group = index.setdefault((case["input_kind"], case["case_id"]), {})
                    group[relative] = value
                    problem = h.disagreement(group)
                    if problem:
                        stop_reason = f"result disagreement: {problem}"
                        raise KeyboardInterrupt
                    variant = a["job"]["variant"]["id"]
                    print(f"[done] {variant} {case['case_id']} states={case['states']} "
                          f"CPU={cpu} status={value['status']} "
                          f"total={len(results)}/{manifest['immutable']['total_jobs']}", flush=True)
                    changed = True
                    if value["status"] == "error":
                        stop_reason = "solver/harness error; inspect the saved solver.log"
                        raise KeyboardInterrupt
                now = time.monotonic()
                if now - last_memory >= 5:
                    # Mixed batches are intentional; preserve the original CSV columns.
                    memory_log.write(f"{h.utc()},mixed,mixed,{len(active)},{memory['MemAvailable']},"
                                     f"{total_rss},{memory['SwapFree']}\n")
                    last_memory = now
                if now - last_report >= 30:
                    report(output, manifest, results, active, pending, "running", metadata, h)
                    last_report = last_progress = now
                elif changed or now - last_progress >= 5:
                    progress(output, manifest, results, active, pending, "running", metadata, h)
                    last_progress = now
                if active or pending:
                    time.sleep(.1)
        except BaseException as error:
            stop_reason = stop_reason or f"interrupted: {type(error).__name__}: {error}"
            for a in active.values():
                if "reaped_usage" not in a:
                    h.kill(a)
            for a in active.values():
                usage = a.get("reaped_usage")
                if usage is None:
                    usage = h.reap(a, blocking=True)
                a["reason"] = "interrupted"
                value = h.finish_job(a, usage, manifest)
                value["paper"].update(scheduler_policy=POLICY, scheduler_id=metadata["scheduler_id"],
                                      runtime_resources=resources)
                h.save(a["folder"] / "result.json", value)
                results[h.job_relative(a["job"]).as_posix()] = value
            active.clear()
            report(output, manifest, results, active, pending, "stopped", metadata, h, stop_reason)
            print(f"[stopped] {stop_reason}", flush=True)
            if not isinstance(error, KeyboardInterrupt):
                raise
            return 130
    report(output, manifest, results, active, pending, "complete", metadata, h)
    print(f"[complete] {len(results)} outcomes saved in {output}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("run the dispatcher on the Linux server")
    output = args.output.resolve()
    h = frozen_harness(output)
    import fcntl
    with (output / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("this campaign is already running")
        manifest = h.prepare(output, h.load(output / "snapshot/protocol.json"))
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupt)
        return run_dispatch(output, manifest, h)


if __name__ == "__main__":
    raise SystemExit(main())
