"""Read-only terminal dashboard for the frozen paper campaign."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PROJECT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def runtime_resources(output, manifest):
    saved = read_json(output / "progress.json")
    return saved.get("runtime_resources") or {
        "workers": manifest["immutable"]["options"]["workers"],
        "cpu_ids": manifest["immutable"]["cpu_ids"]}


def tail(path, size=65536):
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - size))
            return stream.read(size).decode("utf-8", errors="replace")
    except OSError:
        return ""


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def clip(value, width):
    return value if len(value) <= width else value[:max(0, width - 3)] + "..."


def workers(output, previous):
    tick_rate = os.sysconf("SC_CLK_TCK")
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    now = time.monotonic()
    current, samples = {}, {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
            if b"--execute-job" not in command or str(output).encode() not in command:
                continue
            job_path = Path(os.fsdecode(command[command.index(b"--execute-job") + 1]))
            if not job_path.resolve().is_relative_to(output):
                continue
            cpu = int(command[command.index(b"--cpu") + 1])
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] in ("Z", "X"):
                continue
            start_ticks = int(stat[19])
            elapsed = max(.01, uptime - start_ticks / tick_rate)
            cpu_seconds = (int(stat[11]) + int(stat[12])) / tick_rate
            key = (int(entry.name), start_ticks)
            old = previous.get(key)
            cpu_percent = (100 * (cpu_seconds - old[1]) / max(.01, now - old[0])
                           if old else 100 * cpu_seconds / elapsed)
            samples[key] = (now, cpu_seconds)
            status = (entry / "status").read_text().splitlines()
            rss = sum(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
            affinity = next(line.split(":", 1)[1].strip() for line in status
                            if line.startswith("Cpus_allowed_list:"))
            job = read_json(job_path)
            if not job:
                continue
            solver_log = job_path.parent / "solver.log"
            log = tail(solver_log, 8192)
            lines = [line.strip() for line in log.splitlines() if line.strip()]
            budgets = re.findall(r"\[edit-(?:budget|distance) (\d+)", log)
            pairs = re.findall(r"\[factor-pair [^\]]+\] (?:solving|finished|checking fixed repairs for) (\d+)x(\d+)", log)
            # Enumeration reports the current pair after its distance marker.
            if not pairs:
                pairs = re.findall(r"\[edit-distance \d+(?:/\d+)?\].*?(\d+)x(\d+)", log)
            current[cpu] = {"pid": int(entry.name), "job": job, "elapsed": elapsed,
                            "rss": rss, "cpu_percent": cpu_percent, "affinity": affinity,
                            "budget": budgets[-1] if budgets else "-",
                            "pair": "x".join(pairs[-1]) if pairs else "-",
                            "message": lines[-1] if lines else "Starting / loading solver",
                            "log_path": solver_log,
                            "log_age": max(0, time.time() - solver_log.stat().st_mtime),
                            "log_lines": lines[-12:]}
        except (OSError, ValueError, IndexError, StopIteration):
            continue  # Jobs can exit or be replaced during a refresh.
    return current, samples


def render(output, manifest, active, cpu_filter=None):
    m = manifest["immutable"]
    options = m["options"]
    resources = runtime_resources(output, manifest)
    saved = read_json(output / "progress.json")
    log = tail(output / "run.log")
    phases = list(re.finditer(r"\[phase\] states=(\d+) variant=(\w+) pending=(\d+)", log))
    done = list(re.finditer(r"\[done\].*?total=(\d+)/(\d+)", log))
    total = m["total_jobs"]
    finished = saved.get("completed", 0)
    if saved.get("state") == "running" and done:
        finished = max(finished, int(done[-1][1]))
    phase = saved.get("phase") or {}
    pipelined = saved.get("scheduler_policy", "").startswith("ordered-state-variant-queue-")
    batch_finished = None
    if phases and saved.get("state") == "running" and not pipelined:
        last = phases[-1]
        phase = {"states": int(last[1]), "variant": last[2]}
        batch_total = sum(c["states"] == phase["states"] for c in m["corpus"]) * options["repetitions"]
        batch_finished = batch_total - int(last[3]) + len(re.findall(r"\[done\]", log[last.end():]))
    else:
        batch_total = sum(c["states"] == phase.get("states") for c in m["corpus"]) * options["repetitions"]
    variants = {v["id"]: v for v in options["variants"]}
    variant = variants.get(phase.get("variant"), {})
    configuration = variant.get("configuration", {})
    memory = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemAvailable:", "MemTotal:"))}
    try:
        zone = ZoneInfo("Asia/Tehran")
    except ZoneInfoNotFoundError:
        zone = timezone.utc
    clock = datetime.now(zone).strftime("%H:%M:%S %Z")
    lines = [f"Paper campaign | {saved.get('state', 'starting').upper()} | Refreshed {clock}",
             f"Finished: {finished:,} / {total:,} ({100 * finished / total:.1f}%) | "
             f"Active: {len(active)} / {resources['workers']} | Limit: {duration(options['timeout_seconds'])} per test"]
    if pipelined:
        batches = Counter((a['job']['case']['states'], a['job']['variant']['id']) for a in active.values())
        lines.append("Active batches: " + (" | ".join(f"{n} states/{v}: {count}" for (n, v), count in sorted(batches.items())) or "none"))
        queued = saved.get('pending_jobs', 0)
        lines.append(f"Waiting jobs: {queued:,} | Next in queue: " +
                     (f"{phase['states']} states/{phase['variant']}" if phase else "all jobs assigned"))
        lines.append("Scheduling: free CPUs take the next ordered job; variants and state counts can overlap.")
        if cpu_filter is not None and cpu_filter in active:
            configuration = active[cpu_filter]['job']['variant']['configuration']
        else:
            configuration = {}
    else:
        lines.append(f"Current batch: {phase.get('states', '?')} states | {phase.get('variant', '?')} | {variant.get('name', 'Starting')}")
    if batch_finished is not None:
        queued = max(0, batch_total - batch_finished - len(active))
        lines.append(f"Batch finished: {batch_finished}/{batch_total} | Waiting: {queued}")
    if configuration:
        lines.append(f"Options: {configuration['optimization']} / {configuration['sat_solver']} | "
                     f"strictness={configuration['strictness_mode']} | order={configuration['factor_order']} | "
                     f"R={configuration['product_encoding']} | BFS redundancy={configuration['bfs_parent_redundancy']}")
    lines.append(f"Available server RAM: {memory['MemAvailable'] / GIB:.1f} GiB | "
                 f"Active solver RSS: {sum(a['rss'] for a in active.values()) / GIB:.2f} GiB | "
                 f"Per-test ceiling: {options['memory_gib_per_test']} GiB")
    if "aggregate_memory_max_gib" in resources:
        lines.append(f"Campaign memory ceiling: {resources['aggregate_memory_max_gib']} GiB | "
                     f"Pause new jobs at: {resources['aggregate_memory_high_gib']} GiB | "
                     f"Host RAM reserve: {resources['minimum_available_ram_gib']} GiB")
    if saved.get("reason"):
        lines.append(f"Run message: {saved['reason']}")
    columns = max(80, shutil.get_terminal_size((120, 40)).columns)
    lines += ["", "CPU  States Mode  Input                    Elapsed    RAM GiB   CPU%  Budget  Factors"]
    for cpu in resources["cpu_ids"]:
        if cpu_filter is not None and cpu_filter != cpu:
            continue
        a = active.get(cpu)
        if not a:
            lines.append(f"{cpu:>3}  Idle / waiting for next input")
            continue
        case = a["job"]["case"]
        case_name = ("N:" if case["input_kind"] == "nonminimal" else "M:") + case["case_id"]
        lines.append(f"{cpu:>3}  {case['states']:>6} {a['job']['variant']['id']:<4}  {clip(case_name, 24):<24} {duration(a['elapsed'])}  "
                     f"{a['rss'] / GIB:>7.3f}  {a['cpu_percent']:>5.1f}  {a['budget']:>6}  {a['pair']:>7}")
        lines.append(clip(f"     {a['message']} [log age {duration(a['log_age'])}]", columns))
        if cpu_filter is not None:
            lines += [f"     Input kind: {case['input_kind']} | Parsed states: {case['states']} | PID: {a['pid']}",
                      f"     Affinity: {a['affinity']} | Remaining limit: {duration(options['timeout_seconds'] - a['elapsed'])}",
                      f"     Full input ID: {case['case_id']}", f"     Log: {a['log_path']}", "", "Recent solver messages:"]
            lines += a["log_lines"]
    statuses = saved.get("statuses", {})
    if statuses:
        try:
            age = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(saved["updated_at_utc"])).total_seconds())
            lines.append(f"\nSaved outcome counts ({int(age)}s ago): " + ", ".join(f"{k}={v:,}" for k, v in statuses.items()))
        except (ValueError, KeyError):
            pass
    lines += ["", "M = minimal, N = nonminimal. Budget is the current search cost, not a proven optimum.",
              "CPU% uses recent samples (lifetime average on the first refresh). RAM is current solver RSS.",
              "An unchanged solver step can still be busy; CPU% and elapsed time show its activity.",
              "Ctrl+C closes this monitor; the tests continue."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "remote/results/paper-final-3-10")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--cpu", type=int, help="show full details for one CPU slot")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("refresh interval must be at least one second")
    if sys.platform != "linux":
        parser.error("run this monitor on the Linux server")
    output = args.output.resolve()
    manifest = read_json(output / "campaign.json")
    if not manifest:
        parser.error(f"campaign manifest not found: {output}")
    resources = runtime_resources(output, manifest)
    if args.cpu is not None and args.cpu not in resources["cpu_ids"]:
        parser.error("that CPU slot is not assigned to this campaign")
    # Keep the monitor off the CPUs assigned to the tests where possible.
    spare_cpus = os.sched_getaffinity(0) - set(resources["cpu_ids"])
    if spare_cpus:
        os.sched_setaffinity(0, spare_cpus)
    previous = {}
    try:
        while True:
            active, previous = workers(output, previous)
            view = render(output, manifest, active, args.cpu)
            if sys.stdout.isatty() and not args.once:
                print("\033[H\033[2J", end="")
            print(view, flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor closed. Tests continue in the background.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
