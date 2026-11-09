"""Frozen, resumable Linux paper campaign with CPU slots and ordered phases.

The parent supervises fresh single-CPU children. Only this campaign's process
groups are terminated. Resource limits do not reserve physical memory.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
TERMINAL = {"sat", "unsat", "timeout", "memory_limit"}
HARNESS_REVISION = "paper-campaign-v1"
THREAD_ENV = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def log_tail(path, limit=8000):
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read(limit).decode("utf-8", errors="replace")


def meminfo():
    return {line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"))}


def cpu_slots(workers):
    allowed = sorted(os.sched_getaffinity(0))
    first, siblings, seen = [], [], set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (int((topology / "physical_package_id").read_text()),
                   int((topology / "core_id").read_text()))
        except (OSError, ValueError):
            key = (0, cpu)
        (siblings if key in seen else first).append(cpu)
        seen.add(key)
    if workers > len(allowed):
        raise ValueError(f"only {len(allowed)} logical CPUs are allowed")
    return (first + siblings)[:workers]


def validate_capacity(options):
    cpu_slots(options["workers"])
    required = (options["workers"] * options["memory_gib_per_test"]
                + options["minimum_available_ram_gib"]) * GIB
    available = meminfo()["MemAvailable"]
    if required > available:
        raise ValueError(f"test ceilings plus RAM reserve require {required / GIB:.1f} GiB; "
                         f"currently only {available / GIB:.1f} GiB is available")


def runtime():
    spec = importlib.util.find_spec("pysolvers")
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "python_binary_sha256": sha(Path(sys.executable).resolve()),
            "packages": sorted((d.metadata["Name"], d.version)
                               for d in importlib.metadata.distributions()),
            "pysolvers_binary_sha256": sha(Path(spec.origin)) if spec and spec.origin else None}


def protocol_options(plan):
    execution = plan["execution"]
    selected = execution["variant_sequence"]
    by_id = {v["id"]: v for v in plan["variants"]}
    options = {key: execution[key] for key in (
        "state_counts", "workers", "timeout_seconds", "repetitions",
        "memory_gib_per_test", "minimum_available_ram_gib")}
    options["case_selection"] = execution.get("case_selection")
    if any(not isinstance(options[k], int) or options[k] <= 0 for k in (
        "workers", "timeout_seconds", "repetitions", "memory_gib_per_test",
        "minimum_available_ram_gib")):
        raise ValueError("resource limits and repetitions must be positive integers")
    states = options["state_counts"]
    if not states or states != sorted(set(states)) or min(states) < 3:
        raise ValueError("state_counts must be unique and ascending, starting at 3 or higher")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate variant ID")
    options["variants"] = [by_id[v] for v in selected]
    rc2_default = {"solver": "g3", "adapt": False, "exhaust": False,
                   "minz": False, "trim": 0, "incr": False, "process": 0, "verbose": 0}
    for variant in options["variants"]:
        c = variant["configuration"]
        if c["factor_sizes"] is not None or c["strictness"] is not True:
            raise ValueError("this campaign requires automatic sizes and encoded strictness")
        if c["optimization"] not in ("sat", "maxsat", "enumeration"):
            raise ValueError("unsupported optimization")
        if c["optimization"] == "maxsat" and (
            variant["maxsat_algorithm"] != "rc2" or variant["rc2"] != rc2_default):
            raise ValueError("only the selected default RC2 profile is enabled")
    return options


def selected_cases(options):
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(PROJECT / "scripts"))
    import run_all_small_dfaminer as scanner
    cases = [c for c in scanner.scan_cases(min(options["state_counts"]), max(options["state_counts"]))
             if c.state_count in options["state_counts"]]
    if options["case_selection"] is not None:
        selected = {(v["input_kind"], v["case_id"]) for v in options["case_selection"]}
        cases = [c for c in cases if (c.kind, c.case_id) in selected]
        if {(c.kind, c.case_id) for c in cases} != selected:
            raise ValueError("some explicitly selected cases are unavailable")
    return cases


def prepare(output, plan):
    """Freeze code, inputs, options and environment; never replace a campaign."""
    options = protocol_options(plan)
    manifest_path = output / "campaign.json"
    if manifest_path.is_file():
        manifest = load(manifest_path)
        immutable = manifest["immutable"]
        if immutable["options"] != options:
            raise ValueError("options differ from the frozen campaign; use a different output directory")
        if manifest["campaign_id"] != digest(immutable):
            raise ValueError("campaign manifest integrity check failed")
        if immutable["harness_revision"] != HARNESS_REVISION:
            raise ValueError("harness revision differs")
        # JSON turns tuples into lists; normalize the runtime record accordingly.
        if immutable["runtime"] != json.loads(json.dumps(runtime())):
            raise ValueError("installed Python/solver environment differs from the frozen campaign")
        for relative, expected in immutable["snapshot_hashes"].items():
            if sha(output / relative) != expected:
                raise ValueError(f"frozen file changed: {relative}")
        if not set(immutable["cpu_ids"]) <= os.sched_getaffinity(0):
            raise ValueError("some frozen CPU slots are unavailable")
        return manifest
    if (output / "snapshot").exists():
        raise ValueError("incomplete snapshot exists without campaign.json; inspect it before restarting")
    cases = selected_cases(options)
    counts = Counter(c.state_count for c in cases)
    if any(not counts[n] for n in options["state_counts"]):
        raise ValueError("at least one requested state count has no inputs")
    staging = output / f".preparation-{os.getpid()}"
    staging.mkdir()
    source = staging / "source"
    files = [p for directory in ("dfa_identify", "scripts", "tests")
             for p in (PROJECT / directory).rglob("*.py") if "__pycache__" not in p.parts]
    files += [p for p in (PROJECT / "examples").glob("*") if p.suffix in (".dot", ".json", ".py")]
    files += [PROJECT / name for name in ("remote/run_paper.py", "remote/start_paper.sh",
              "remote/experiments/paper-20261005/verification-variants.json",
              "pyproject.toml", "poetry.lock", "remote/requirements-lock.txt")]
    for path in files:
        destination = source / path.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    corpus = []
    for case in cases:
        path = case.benchmark.input_path
        relative = f"inputs/{case.kind}/{case.case_id}.dot"
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha(destination) != case.benchmark.dot_sha256:
            raise ValueError(f"input changed while freezing: {path}")
        corpus.append({"case_id": case.case_id, "input_kind": case.kind,
                       "states": case.state_count, "input_dot_sha256": sha(destination),
                       "input": f"snapshot/{relative}", "original_input": str(path),
                       "source_size": case.benchmark.source_size,
                       "instance_id": case.benchmark.instance_id,
                       "metadata": case.benchmark.metadata})
    save(staging / "protocol.json", plan)
    staging.rename(output / "snapshot")
    snapshot_hashes = {p.relative_to(output).as_posix(): sha(p)
                       for p in sorted((output / "snapshot").rglob("*")) if p.is_file()}
    cpus = cpu_slots(options["workers"])
    immutable = {"harness_revision": HARNESS_REVISION, "options": options,
                 "cpu_ids": cpus, "corpus": corpus, "snapshot_hashes": snapshot_hashes,
                 "runtime": runtime(), "deterministic_environment": THREAD_ENV,
                 "created_at_utc": utc(), "total_jobs": len(cases) * len(options["variants"])
                 * options["repetitions"]}
    manifest = {"campaign_id": digest(immutable), "immutable": immutable,
                "hardware": {"platform": platform.platform(), "hostname": platform.node(),
                             "allowed_cpu_ids": sorted(os.sched_getaffinity(0)),
                             "cpuinfo": Path("/proc/cpuinfo").read_text(),
                             "initial_memory": meminfo(),
                             "cpu_selection": "one logical CPU per exposed core first; then siblings"}}
    save(manifest_path, manifest)
    return load(manifest_path)


def jobs(manifest):
    m = manifest["immutable"]
    for n in m["options"]["state_counts"]:
        for variant in m["options"]["variants"]:
            # A phase barrier exists between each yielded block.
            block = []
            for repetition in range(1, m["options"]["repetitions"] + 1):
                for case in m["corpus"]:
                    if case["states"] == n:
                        block.append({"variant": variant, "case": case,
                                      "repetition": repetition})
            yield n, variant["id"], block


def job_relative(job):
    c = job["case"]
    return Path("runs") / f"states_{c['states']:02d}" / job["variant"]["id"] / (
        f"repeat_{job['repetition']:02d}") / c["input_kind"] / c["case_id"]


def reusable(path, job, campaign_id):
    if not path.is_file():
        return None
    payload = load(path)
    if payload.get("paper", {}).get("campaign_id") != campaign_id or payload["paper"].get("job") != job:
        raise ValueError(f"result provenance mismatch: {path}")
    return payload if payload["status"] in TERMINAL else None


def execute_job(output, job_path, cpu):
    """Fresh child entry point. Limits and affinity precede solver imports."""
    import resource
    manifest = load(output / "campaign.json")
    job = load(job_path)
    c = job["variant"]["configuration"]
    source = output / "snapshot/source"
    os.sched_setaffinity(0, {cpu})
    limit = manifest["immutable"]["options"]["memory_gib_per_test"] * GIB
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY and limit > hard:
        raise ValueError("inherited address-space ceiling is below the selected test limit")
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    folder = job_path.parent
    save(folder / "worker.json", {"cpu_id": cpu, "cpu_affinity": sorted(os.sched_getaffinity(0)),
                                 "address_space_limit_bytes": limit, "pid": os.getpid()})
    if sha(output / job["case"]["input"]) != job["case"]["input_dot_sha256"]:
        raise ValueError("frozen input hash mismatch")
    sys.path.insert(0, str(source))
    try:
        from pysat.examples.rc2 import RC2
        import inspect
        for key, expected in (job["variant"].get("rc2") or {}).items():
            if inspect.signature(RC2).parameters[key].default != expected:
                raise ValueError(f"installed RC2 default differs: {key}")
        from dfa_identify.cli import main as solve
        argv = [str(output / job["case"]["input"]), "--strictness", "--optimization", c["optimization"],
                "--product-encoding", c["product_encoding"], "--strictness-mode", c["strictness_mode"],
                "--factor-order", c["factor_order"], "--sat-solver", c["sat_solver"],
                "--bfs-parent-redundancy" if c["bfs_parent_redundancy"] else "--no-bfs-parent-redundancy",
                "--show-progress", "--output", str(folder / "solver-result.json"),
                "--visualize-dir", str(folder / "visualizations")]
        return solve(argv)
    except MemoryError:
        print("[paper-memory-limit] Python allocation reached the address-space ceiling", flush=True)
        return 86


def tree_pids(root):
    found, pending = set(), [root]
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        try:
            pending.extend(int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
        except (OSError, ValueError):
            pass
    return found


def sample_tree(pid):
    rss, pss, available = 0, 0, False
    for child in tree_pids(pid):
        try:
            status = Path(f"/proc/{child}/status").read_text().splitlines()
            rss += sum(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
            for line in Path(f"/proc/{child}/smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    pss += int(line.split()[1]) * 1024
                    available = True
        except (OSError, ValueError):
            pass
    return rss, pss if available else None


def kill(job):
    try:
        os.killpg(job["process"].pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def reap(job, blocking=False):
    pid, status, usage = os.wait4(job["process"].pid, 0 if blocking else os.WNOHANG)
    if not pid:
        return None
    job["process"].returncode = os.waitstatus_to_exitcode(status)
    return usage


def archive_attempt(folder):
    if not any(folder.iterdir()):
        return
    archive = folder / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive.mkdir(parents=True)
    for child in list(folder.iterdir()):
        if child.name != "attempts":
            child.rename(archive / child.name)


def start_job(output, manifest, job, cpu):
    folder = output / job_relative(job)
    folder.mkdir(parents=True, exist_ok=True)
    archive_attempt(folder)
    job_path = folder / "job.json"
    save(job_path, job)
    stream = (folder / "solver.log").open("w", encoding="utf-8")
    command = [sys.executable, "-u", str(output / "snapshot/source/remote/run_paper.py"),
               "--output", str(output), "--execute-job", str(job_path), "--cpu", str(cpu)]
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=output / "snapshot/source", stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True,
                               env=os.environ | THREAD_ENV)
    return {"job": job, "folder": folder, "process": process, "stream": stream,
            "cpu": cpu, "started": started, "started_at_utc": utc(), "peak_rss": 0,
            "peak_pss": None, "samples": 0, "reason": None,
            "last_sample": 0.0, "last_rss": 0}


def finish_job(active, usage, manifest):
    active["stream"].close()
    path = active["folder"] / "solver-result.json"
    code = active["process"].returncode
    reason = active["reason"]
    tail = log_tail(active["folder"] / "solver.log")
    payload = {"status": reason or "error"}
    if reason is None:
        if code == 86 or (code != 0 and any(token in tail for token in (
            "MemoryError", "std::bad_alloc", "[paper-memory-limit]"))):
            payload = {"status": "memory_limit", "failure_reason": "address-space allocation failure"}
        elif code == 0 and path.is_file():
            payload = load(path)
            c = active["job"]["variant"]["configuration"]
            expected = {"optimization_backend": c["optimization"], "strictness_strategy": "witness",
                        "strictness_mode": c["strictness_mode"], "factor_order": c["factor_order"],
                        "product_encoding": c["product_encoding"],
                        "bfs_parent_redundancy": c["bfs_parent_redundancy"], "symmetry_breaking": "bfs",
                        "sat_solver": c["sat_solver"] if c["optimization"] != "maxsat" else None}
            if payload.get("status") not in ("sat", "unsat") or any(
                payload.get(k) != v for k, v in expected.items()):
                payload = {"status": "error", "failure_reason": "solver result configuration mismatch"}
    proof_path = active["folder"] / "worker.json"
    proof = load(proof_path) if proof_path.is_file() else None
    if payload["status"] in ("sat", "unsat") and (
        not proof or proof["cpu_affinity"] != [active["cpu"]]):
        payload = {"status": "error", "failure_reason": "CPU affinity verification failed"}
    payload["paper"] = {"campaign_id": manifest["campaign_id"], "job": active["job"],
                        "started_at_utc": active["started_at_utc"], "finished_at_utc": utc(),
                        "process_returncode": code, "worker": proof,
                        "wall_seconds": round(time.monotonic() - active["started"], 6),
                        "cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime,
                        "kernel_peak_rss_bytes": usage.ru_maxrss * 1024,
                        "sampled_peak_tree_rss_bytes": active["peak_rss"],
                        "sampled_peak_tree_pss_bytes": active["peak_pss"],
                        "memory_samples": active["samples"], "sample_interval_seconds": 1,
                        "resource_scope": "supervised wall includes up to 0.1 s polling latency; wall/CPU/kernel RSS include imports, parsing, optimization, validation and graph export; sampled tree peaks are lower bounds"}
    if payload["status"] not in ("sat", "unsat"):
        payload["solver_output_tail"] = tail
    save(active["folder"] / "result.json", payload)
    return payload


def disagreement(results):
    groups = {}
    for value in results.values():
        if value["status"] not in ("sat", "unsat"):
            continue
        j = value["paper"]["job"]
        if j["variant"]["configuration"]["strictness_mode"] != "finite":
            continue
        c = j["case"]
        groups.setdefault((c["input_kind"], c["case_id"]), []).append(value)
    for key, values in groups.items():
        if len({(v["status"], v.get("edit_distance")) for v in values}) > 1:
            return {"case": key, "reason": "finite-mode optimum/status disagreement",
                    "variants": [v["paper"]["job"]["variant"]["id"] for v in values]}
        by_order = {}
        for v in values:
            if v["status"] == "sat":
                order = v["paper"]["job"]["variant"]["configuration"]["factor_order"]
                by_order.setdefault(order, set()).add(tuple(v["factor_sizes"]))
        if any(len(sizes) > 1 for sizes in by_order.values()):
            return {"case": key, "reason": "same-order secondary optimum disagreement"}
    return None


def report(output, manifest, results, state, phase=None, reason=None):
    import csv
    fields = ["states", "variant", "repetition", "input_kind", "case_id", "status", "edit_distance",
              "factor_sizes", "solve_seconds", "wall_seconds", "cpu_seconds", "kernel_peak_rss_bytes",
              "sampled_peak_tree_rss_bytes", "sampled_peak_tree_pss_bytes", "cpu_id", "result"]
    temporary = output / ".results.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for _, _, block in jobs(manifest):
            for j in block:
                relative = job_relative(j)
                v = results.get(relative.as_posix())
                p = v.get("paper", {}) if v else {}
                row = {"states": j["case"]["states"], "variant": j["variant"]["id"],
                       "repetition": j["repetition"], "input_kind": j["case"]["input_kind"],
                       "case_id": j["case"]["case_id"], "status": v["status"] if v else "pending",
                       "result": str(relative / "result.json")}
                if v:
                    row.update(edit_distance=v.get("edit_distance"), factor_sizes=v.get("factor_sizes"),
                               solve_seconds=v.get("timing", {}).get("solve_seconds"),
                               cpu_seconds=p.get("cpu_user_seconds", 0) + p.get("cpu_system_seconds", 0),
                               cpu_id=(p.get("worker") or {}).get("cpu_id"))
                    row.update({k: p.get(k) for k in fields if k in p})
                writer.writerow(row)
    temporary.replace(output / "results.csv")
    statuses = Counter(v["status"] for v in results.values())
    save(output / "progress.json", {"state": state, "phase": phase, "reason": reason,
         "campaign_id": manifest["campaign_id"], "completed": sum(statuses[s] for s in TERMINAL),
         "total": manifest["immutable"]["total_jobs"], "statuses": dict(statuses), "updated_at_utc": utc()})


def run_campaign(output, manifest):
    options = manifest["immutable"]["options"]
    cpus = manifest["immutable"]["cpu_ids"]
    minimum = options["minimum_available_ram_gib"] * GIB
    validate_capacity(options)
    results, active, stop_reason = {}, {}, None
    for _, _, block in jobs(manifest):
        for j in block:
            relative = job_relative(j)
            saved = reusable(output / relative / "result.json", j, manifest["campaign_id"])
            if saved:
                results[relative.as_posix()] = saved
    prior_problem = disagreement(results)
    if prior_problem:
        report(output, manifest, results, "invalid", reason=prior_problem)
        return 1
    phase = None
    report(output, manifest, results, "running")
    last_report, last_memory = time.monotonic(), 0.0
    with (output / "server-memory.csv").open("a", encoding="utf-8", buffering=1) as memory_log:
        if memory_log.tell() == 0:
            memory_log.write("utc,states,variant,active_jobs,available_ram_bytes,tree_rss_bytes,swap_free_bytes\n")
        try:
            for n, variant_id, block in jobs(manifest):
                phase = {"states": n, "variant": variant_id}
                pending = deque(j for j in block if job_relative(j).as_posix() not in results)
                print(f"[phase] states={n} variant={variant_id} pending={len(pending)}", flush=True)
                while pending or active:
                    memory = meminfo()
                    if memory["MemAvailable"] < minimum:
                        stop_reason = "server RAM reserve reached; unfinished tests can be resumed"
                        raise KeyboardInterrupt
                    for cpu in cpus:
                        if cpu not in active and pending:
                            j = pending.popleft()
                            active[cpu] = start_job(output, manifest, j, cpu)
                    total_rss = 0
                    for cpu, a in list(active.items()):
                        now = time.monotonic()
                        rss = a["last_rss"]
                        if now - a["last_sample"] >= 1:
                            rss, pss = sample_tree(a["process"].pid)
                            a["last_rss"], a["last_sample"] = rss, now
                            a["samples"] += 1
                            a["peak_rss"] = max(a["peak_rss"], rss)
                            if pss is not None:
                                a["peak_pss"] = max(a["peak_pss"] or 0, pss)
                        total_rss += rss
                        usage = reap(a)
                        if usage is None:
                            if time.monotonic() - a["started"] >= options["timeout_seconds"]:
                                a["reason"] = "timeout"
                                kill(a)
                            elif rss > options["memory_gib_per_test"] * GIB:
                                a["reason"] = "memory_limit"
                                kill(a)
                            continue
                        # Remove ownership before any result parsing/reporting can raise.
                        del active[cpu]
                        value = finish_job(a, usage, manifest)
                        results[job_relative(a["job"]).as_posix()] = value
                        print(f"[done] {variant_id} {a['job']['case']['case_id']} "
                              f"CPU={cpu} status={value['status']} total={len(results)}/{manifest['immutable']['total_jobs']}", flush=True)
                        if value["status"] == "error":
                            stop_reason = "solver/harness error; inspect the saved solver.log before resuming"
                            raise KeyboardInterrupt
                    now = time.monotonic()
                    if now - last_memory >= 5:
                        memory_log.write(f"{utc()},{n},{variant_id},{len(active)},{memory['MemAvailable']},"
                                         f"{total_rss},{memory['SwapFree']}\n")
                        last_memory = now
                    if now - last_report >= 30:
                        report(output, manifest, results, "running", phase)
                        last_report = now
                    if active:
                        time.sleep(0.1)
                problem = disagreement(results)
                if problem:
                    report(output, manifest, results, "invalid", phase, problem)
                    print(f"[invalid] {problem}", flush=True)
                    return 1
                report(output, manifest, results, "running", phase)
        except BaseException as error:
            stop_reason = stop_reason or f"interrupted: {type(error).__name__}: {error}"
            # Kill all owned groups before waiting on or writing any individual result.
            for a in active.values():
                kill(a)
            for a in active.values():
                usage = reap(a, blocking=True)
                a["reason"] = "interrupted"
                value = finish_job(a, usage, manifest)
                results[job_relative(a["job"]).as_posix()] = value
            report(output, manifest, results, "stopped", phase, stop_reason)
            print(f"[stopped] {stop_reason}", flush=True)
            if not isinstance(error, KeyboardInterrupt):
                raise
            return 130
    report(output, manifest, results, "complete", phase)
    print(f"[complete] {len(results)} outcomes saved in {output}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT / "remote/experiments/paper-20261005/verification-variants.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true", help="freeze and verify without launching tests")
    parser.add_argument("--dry-run", action="store_true", help="inspect options and corpus without writing files")
    parser.add_argument("--execute-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cpu", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("run this harness on the Linux server")
    output = args.output.resolve()
    if args.execute_job is not None:
        return execute_job(output, args.execute_job.resolve(), args.cpu)
    plan = load(args.protocol)
    options = protocol_options(plan)
    if args.dry_run:
        cases = selected_cases(options)
        print(json.dumps({"options": options, "by_states": dict(Counter(c.state_count for c in cases)),
                          "tests": len(cases) * len(options["variants"]) * options["repetitions"],
                          "cpu_ids": cpu_slots(options["workers"]), "memory": meminfo()}, indent=2))
        return 0
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("this campaign is already running")
        validate_capacity(options)
        manifest = prepare(output, plan)
        print(f"[prepared] {manifest['immutable']['total_jobs']} tests; campaign={manifest['campaign_id']}", flush=True)
        if args.prepare:
            return 0
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupt)
        return run_campaign(output, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
