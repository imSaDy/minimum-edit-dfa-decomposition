"""Prepare either the curated subset or the full DFAMiner benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dfa_identify.dot_io import dfa_from_dot, dfa_to_dot


ALL_SIZES = tuple(range(4, 17))
ALL_INSTANCE_IDS = tuple(range(1, 101))
DEFAULT_SELECTION = {
    4: (26, 36, 70, 87),
    5: (32, 61, 74, 97),
    6: (40, 45, 60, 85),
    7: (8, 30, 39, 46),
    8: (2, 36, 63, 81),
    9: (10, 38, 48, 53),
    10: (8, 15, 61, 91),
    11: (19, 23, 25, 84),
    12: (8, 38, 59, 70),
}

_ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()
_ACTIVE_PROCESSES_LOCK = threading.Lock()
_CANCELLED = threading.Event()


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _terminate_all_active_processes() -> None:
    with _ACTIVE_PROCESSES_LOCK:
        processes = tuple(_ACTIVE_PROCESSES)
    for process in processes:
        _terminate_process_tree(process)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_statistics(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().split()
        if len(header) != 2:
            raise ValueError(f"invalid Abbadingo header in {path}")
        declared_count, alphabet_size = map(int, header)
        positive = 0
        negative = 0
        lengths = []
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            membership = int(fields[0])
            length = int(fields[1])
            if len(fields[2:]) != length:
                raise ValueError(f"invalid word length in {path}: {line!r}")
            if membership == 1:
                positive += 1
            elif membership == 0:
                negative += 1
            else:
                raise ValueError(f"unexpected sample label {membership} in {path}")
            lengths.append(length)
    if positive + negative != declared_count:
        raise ValueError(f"sample count does not match header in {path}")
    return {
        "sample_count": declared_count,
        "positive_count": positive,
        "negative_count": negative,
        "alphabet_size": alphabet_size,
        "minimum_word_length": min(lengths),
        "maximum_word_length": max(lengths),
    }


def _prepare_one(
    upstream: Path,
    output: Path,
    source_size: int,
    instance_id: int,
    timeout_seconds: int,
    resume: bool,
) -> dict[str, Any]:
    source_name = f"res{source_size}-{instance_id}.txt"
    source = (
        upstream
        / "DFA_Miner_dataset"
        / "dataset"
        / str(source_size)
        / source_name
    )
    if not source.is_file():
        raise FileNotFoundError(source)

    sample_directory = output / "samples" / f"states_{source_size:02d}"
    dot_directory = output / "dot" / f"states_{source_size:02d}"
    sample_directory.mkdir(parents=True, exist_ok=True)
    dot_directory.mkdir(parents=True, exist_ok=True)
    copied_sample = sample_directory / source_name
    shutil.copy2(source, copied_sample)

    output_name = f"dfa-{source_size:02d}-{instance_id:03d}.dot"
    dot_path = dot_directory / output_name
    dfa = None
    elapsed = None
    if resume and dot_path.is_file():
        try:
            dfa = dfa_from_dot(dot_path)
        except (OSError, TypeError, ValueError):
            # A previously interrupted write is regenerated below.  The old
            # file is only replaced after DFAMiner produces a valid DFA.
            dfa = None

    if dfa is None:
        if _CANCELLED.is_set():
            raise RuntimeError("benchmark preparation was cancelled")
        with tempfile.TemporaryDirectory(prefix="dfaminer-") as temporary:
            raw_dot = Path(temporary) / output_name
            command = [
                sys.executable,
                str(upstream / "dfaminer.py"),
                "--file",
                str(copied_sample.resolve()),
                "--out",
                str(raw_dot),
                "--output-format",
                "dot",
                "--upper",
                "20",
                "--verify",
            ]
            started = time.perf_counter()
            popen_options: dict[str, Any] = {}
            if os.name == "nt":
                popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_options["start_new_session"] = True
            process = subprocess.Popen(
                command,
                cwd=upstream,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **popen_options,
            )
            with _ACTIVE_PROCESSES_LOCK:
                _ACTIVE_PROCESSES.add(process)
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as error:
                _CANCELLED.set()
                _terminate_all_active_processes()
                raise RuntimeError(
                    f"DFAMiner timed out after {timeout_seconds} seconds for "
                    f"{source_name}"
                ) from error
            finally:
                with _ACTIVE_PROCESSES_LOCK:
                    _ACTIVE_PROCESSES.discard(process)
            elapsed = time.perf_counter() - started
            if process.returncode != 0 or not raw_dot.is_file():
                details = (stderr or stdout)[-4000:]
                raise RuntimeError(
                    f"DFAMiner failed for {source_name} "
                    f"({process.returncode}):\n{details}"
                )
            dfa = dfa_from_dot(raw_dot)

    states = tuple(dfa.states())
    alphabet = tuple(sorted(dfa.inputs))
    if alphabet != ("a", "b"):
        raise ValueError(f"unexpected alphabet for {source_name}: {alphabet!r}")
    if len(states) > 20:
        raise ValueError(f"DFA exceeds 20 states for {source_name}")
    if elapsed is not None:
        dot_path.write_text(dfa_to_dot(dfa), encoding="utf-8")
    reparsed = dfa_from_dot(dot_path)
    if reparsed != dfa:
        raise ValueError(f"DOT round trip changed the DFA for {source_name}")

    statistics = _sample_statistics(copied_sample)
    return {
        "id": f"dfaminer-{source_size:02d}-{instance_id:03d}",
        "source_size": source_size,
        "instance_id": instance_id,
        "learned_state_count": len(states),
        "accepting_state_count": sum(bool(dfa._label(state)) for state in states),
        "alphabet": list(alphabet),
        **statistics,
        "dfaminer_seconds": round(elapsed, 4) if elapsed is not None else None,
        "sample": copied_sample.relative_to(output).as_posix(),
        "dot": dot_path.relative_to(output).as_posix(),
        "sample_sha256": _sha256(copied_sample),
        "dot_sha256": _sha256(dot_path),
    }


def _upstream_commit(upstream: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks") / "dfaminer"
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="stop if one DFAMiner instance exceeds this duration",
    )
    parser.add_argument(
        "--selection",
        choices=("curated", "all"),
        default="curated",
        help=(
            "prepare the original four-per-size subset or all 100 raw "
            "instances in every source-size category"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse valid normalized DOT files that already exist",
    )
    args = parser.parse_args()
    _CANCELLED.clear()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")

    upstream = args.upstream.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_sizes = (
        tuple(DEFAULT_SELECTION) if args.selection == "curated" else ALL_SIZES
    )
    instance_ids_by_source_size = {
        source_size: (
            DEFAULT_SELECTION[source_size]
            if args.selection == "curated"
            else ALL_INSTANCE_IDS
        )
        for source_size in source_sizes
    }
    jobs = [
        (source_size, instance_id)
        for source_size in source_sizes
        for instance_id in instance_ids_by_source_size[source_size]
    ]
    records = []
    executor = ThreadPoolExecutor(max_workers=args.jobs)
    futures = {
        executor.submit(
            _prepare_one,
            upstream,
            output,
            source_size,
            instance_id,
            args.timeout_seconds,
            args.resume,
        ): (source_size, instance_id)
        for source_size, instance_id in jobs
    }
    try:
        for future in as_completed(futures):
            source_size, instance_id = futures[future]
            record = future.result()
            records.append(record)
            print(
                f"prepared N={source_size} id={instance_id}: "
                f"{record['learned_state_count']} states",
                flush=True,
            )
    except BaseException:
        _CANCELLED.set()
        _terminate_all_active_processes()
        for pending in futures:
            pending.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    records.sort(key=lambda item: (item["source_size"], item["instance_id"]))
    license_source = upstream / "LICENSE.txt"
    if license_source.is_file():
        shutil.copy2(license_source, output / "UPSTREAM_LICENSE.txt")
    learned_size_distribution: dict[str, int] = {}
    exact_source_size_count = 0
    for record in records:
        learned_size = str(record["learned_state_count"])
        learned_size_distribution[learned_size] = (
            learned_size_distribution.get(learned_size, 0) + 1
        )
        if record["learned_state_count"] == record["source_size"]:
            exact_source_size_count += 1

    manifest = {
        "source": "https://github.com/liyong31/DFAMiner",
        "upstream_commit": _upstream_commit(upstream),
        "selection": {
            "mode": args.selection,
            "source_sizes": list(source_sizes),
            "instance_ids_by_source_size": {
                str(size): list(instance_ids_by_source_size[size])
                for size in source_sizes
            },
            "criteria": (
                "four most label-balanced instances whose learned minimal DFA "
                "has the source category's state count"
                if args.selection == "curated"
                else "all 100 upstream raw sample files in every source category"
            ),
            "instances_per_size": 4 if args.selection == "curated" else 100,
            "total_instances": len(records),
        },
        "curated_subset": {
            "instance_ids_by_source_size": {
                str(size): list(DEFAULT_SELECTION[size])
                for size in DEFAULT_SELECTION
            }
        },
        "symbol_mapping": {"0": "a", "1": "b"},
        "summary": {
            "exact_source_size_count": exact_source_size_count,
            "different_from_source_size_count": (
                len(records) - exact_source_size_count
            ),
            "learned_state_count_distribution": dict(
                sorted(
                    learned_size_distribution.items(),
                    key=lambda item: int(item[0]),
                )
            ),
        },
        "instances": records,
    }
    temporary_manifest = output / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(output / "manifest.json")
    print(f"wrote {len(records)} instances to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
