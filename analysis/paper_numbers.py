"""Recompute the numbers reported in Section 6 of the paper from results/per-run-results.csv.

Usage:  python analysis/paper_numbers.py [path/to/per-run-results.csv]

Only the standard library is used. Numbers that need the per-attempt logs of the
raw result files (time per budget and size) are not recomputed here.
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TIME_LIMIT = 8 * 3600
NAMES = {
    "T01": "default", "T02": "MaxSAT", "T03": "enumeration", "T04": "CaDiCaL",
    "T05": "MapleCM", "T06": "MiniSat", "T07": "strongly strict",
    "T08": "no redundant parent clauses", "T09": "direct product closure",
    "T16": "sum order", "T17": "product order", "T18": "compact BFS",
    "T19": "compact BFS, no redundant parent clauses", "T20": "lazy strictness",
}


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%"


def load(path: Path):
    runs = defaultdict(dict)  # variant -> case_id -> row
    states = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["input_kind"] != "minimal":  # the paper uses the 693 minimal instances
                continue
            runs[r["variant"]][r["case_id"]] = r
            states[r["case_id"]] = int(r["states"])
    return runs, states


def solved(rows):
    return {c for c, r in rows.items() if r["status"] == "sat"}


def seconds(r, clip=0.01):
    return max(float(r["solve_seconds"]), clip)


def slower_share(runs, variant, base="T01"):
    joint = solved(runs[variant]) & solved(runs[base])
    slower = sum(seconds(runs[variant][c]) > seconds(runs[base][c]) for c in joint)
    faster = sum(seconds(runs[variant][c]) < seconds(runs[base][c]) for c in joint)
    return pct(slower, len(joint)), pct(faster, len(joint))


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "per-run-results.csv"
    runs, states = load(path)
    n_all = len(states)
    by_n = defaultdict(set)
    for c, n in states.items():
        by_n[n].add(c)
    print(f"Instances: {n_all}  " + ", ".join(f"n={n}: {len(by_n[n])}" for n in sorted(by_n)))

    print("\nRQ1 (default configuration)")
    d = runs["T01"]
    print("  solved:", pct(len(solved(d)), n_all), "| at 10 states:", pct(len(solved(d) & by_n[10]), len(by_n[10])))
    med = {}
    for n in sorted(by_n):
        times = [float(d[c]["solve_seconds"]) if d[c]["status"] == "sat" else TIME_LIMIT for c in by_n[n]]
        med[n] = statistics.median(times)
    print("  median running time (timeouts count as 8 h):", ", ".join(f"n={n}: {med[n]:.3g} s" for n in med))

    print("\nRQ2 (optimal costs)")
    cost = {}
    for v, rows in runs.items():
        if v == "T07":
            continue
        for c in solved(rows):
            k = int(rows[c]["edit_distance"])
            if cost.setdefault(c, k) != k:
                print("  disagreement on", c)
    dist = defaultdict(int)
    for k in cost.values():
        dist[k] += 1
    print("  " + ", ".join(f"cost {k}: {pct(m, n_all)}" for k, m in sorted(dist.items())), f"| unknown: {pct(n_all - len(cost), n_all)}")
    strong = runs["T07"]
    known = [c for c in solved(strong) if c in cost]
    same = sum(int(strong[c]["edit_distance"]) == cost[c] for c in known)
    print("  strongly strict: same cost on", pct(same, len(known)), "of the instances it solves;",
          "timeouts on", pct(len([c for c in cost if strong[c]["status"] != "sat"]), len(cost)), "of the instances with a known cost")

    print("\nRQ3 (configurations)")
    for v in sorted(runs, key=lambda x: int(x[1:])):
        s = solved(runs[v])
        line = f"  {NAMES.get(v, v):42s} solved {pct(len(s), n_all):>6s}  at 10 states {pct(len(s & by_n[10]), len(by_n[10])):>6s}"
        if v != "T01":
            sl, fa = slower_share(runs, v)
            line += f"  slower than default on {sl:>6s}, faster on {fa:>6s} of the jointly solved"
        print(line)
    lazy = runs["T20"]
    ls = solved(lazy)
    others = set.intersection(*(solved(runs[v]) for v in runs if v != "T20"))
    print("  lazy strictness fails on", pct(len(others - ls), n_all), "of the instances that every other configuration solves")
    print("  lazy: first decomposition not strict in", pct(sum(int(lazy[c]["blocked_models"] or 0) > 0 for c in ls), len(ls)),
          "of the solved runs; more than a million excluded pairs in", pct(sum(int(lazy[c]["blocked_models"] or 0) > 10**6 for c in ls), len(ls)))
    joint = solved(runs["T01"]) & solved(runs["T17"])
    bigger = sum(max(int(runs["T17"][c]["factor_1"]), int(runs["T17"][c]["factor_2"]))
                 > max(int(runs["T01"][c]["factor_1"]), int(runs["T01"][c]["factor_2"])) for c in joint)
    print("  product order returns a larger larger factor than the max order on", pct(bigger, len(joint)), "of the jointly solved")
    j10 = sorted(solved(runs["T01"]) & solved(runs["T09"]) & by_n[10])
    mem = [float(runs["T09"][c]["sampled_peak_tree_rss_mib"]) / float(runs["T01"][c]["sampled_peak_tree_rss_mib"]) for c in j10]
    print(f"  direct closure, 10 states: median peak-memory ratio {statistics.median(mem):.2f} over {len(j10)} jointly solved instances")


if __name__ == "__main__":
    main()
