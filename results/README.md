# Results

- `per-run-results.csv`: one row per run, 14 configurations × 723 inputs. The
  paper uses the 693 rows per configuration with `input_kind = minimal`. The 30
  rows with `input_kind = nonminimal` belong to an earlier benchmark design and
  are not used in the paper.
- `configurations.csv`: the options of each configuration (`variant` T01–T20),
  both as stored by the experiment harness and as command-line options.

## Main columns of `per-run-results.csv`

| Column | Meaning |
| --- | --- |
| `variant` | Configuration, see `configurations.csv`. |
| `case_id`, `states`, `input_dot_sha256` | The instance, its number of states and the hash of its DOT file (`benchmark/instances.csv`). |
| `status` | `sat` if the run finished within the time limit, `timeout` otherwise. |
| `edit_distance` | Optimal edit cost found by the run. |
| `factor_1`, `factor_2` | Numbers of states of the returned factors. |
| `solve_seconds` | Running time of the solver in seconds. |
| `sampled_peak_tree_rss_mib` | Peak memory of the run in MiB, sampled once per second. |
| `kernel_peak_worker_rss_mib` | Peak memory reported by the kernel. Its values drift upward during a campaign, apparently because they include memory of the supervising process, so use `sampled_peak_tree_rss_mib` to compare configurations. |
| `blocked_models` | Lazy strictness only: number of excluded non-strict pairs of factors. |
| `base_cnf_variables`, `base_cnf_clauses`, `base_cnf_literals` | Size of the formula for the returned factor sizes. |
| `campaign`, `campaign_id` | Server campaign of the run (see the main README). |

Each run had a time limit of 8 hours and a memory limit of 16 GiB, and ran
once. The columns `result_file` and `result_sha256` refer to the raw result
file of each run, which is not part of this repository.

`analysis/paper_numbers.py` recomputes the numbers of the paper from this file.
