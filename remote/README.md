# Experiment harness

These files ran the experiments of the paper on a Linux server with 64
logical CPUs (Intel Xeon Gold 6230, 2.1 GHz) and 247 GiB of memory. Up to 54
runs were executed in parallel, each pinned to its own logical CPU. They are
kept unchanged so that the experimental setup can be inspected.

| File | Role |
| --- | --- |
| `run_paper.py` | Prepares a campaign: freezes the code, the inputs, the options and the software environment, and records their hashes in `campaign.json`. Runs each job in a fresh process with the time and memory limits. |
| `dispatch_paper.py` | Keeps the CPU slots busy with the queued jobs of a campaign. |
| `monitor_paper.py`, `monitor_paper.sh` | Read-only progress display. |
| `start_paper.sh` | Starts the reference campaign. |
| `setup_server.sh` | Creates the Python environment on the server from `requirements-lock.txt`. |
| `experiments/paper-20261005/verification-variants.json` | The protocol of the reference campaign: its configurations, limits and inputs. `VERIFY-VARIANTS.md` describes it. |
| `../scripts/compact_bfs_campaign.py`, `../scripts/start_compact_bfs_campaign.sh` | Prepare and start the supplement campaign (compact BFS) on the same inputs. |
| `../scripts/blocking_campaign.py`, `../scripts/start_blocking_campaign.sh` | Prepare and start the blocking campaign (lazy strictness) on the same inputs. |
| `../scripts/run_all_small_dfaminer.py`, `../scripts/run_minimal_five_benchmark.py`, `../scripts/run_nonminimal_five_benchmark.py` | Locate the inputs for `run_paper.py` and run a single job. |
| `../scripts/build_minimal_five_benchmark_report.py` | Reads result files; used by the tests. |
| `../scripts/prepare_dfaminer_subset.py` | Computes the benchmark instances with DFAMiner (see `benchmark/README.md`). |

The harness reads the inputs from the benchmark directory layout of the
original workspace, which is not part of this repository. To reproduce
individual runs, use the command-line tool with the instances in
`benchmark/instances/` and the options in `results/configurations.csv`.
