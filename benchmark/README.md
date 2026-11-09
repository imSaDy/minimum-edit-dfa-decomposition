# Benchmark

`instances/` contains the 693 DFAs used in the paper, grouped by their number
of states:

| States | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Instances | 18 | 98 | 93 | 95 | 100 | 90 | 104 | 95 |

## Origin

The instances are derived from the DFAMiner benchmark
(<https://github.com/liyong31/DFAMiner>). Each sample of that benchmark is
generated from a random target DFA over a binary alphabet. Its positive
examples are random words that the target DFA accepts, and its negative
examples are random words that it rejects. The benchmark contains 100 samples
for each number of states of the target DFA from 4 to 16.

For every sample whose target DFA has 4 to 12 states, we computed a minimum
separating DFA with DFAMiner, that is, a DFA with as few states as possible
that accepts all positive examples and rejects all negative examples. The
instances are the 693 minimum separating DFAs that have 3 to 10 states.
`scripts/prepare_dfaminer_subset.py` performs this computation from a DFAMiner
checkout.

## Files

- `instances/states_NN/dfa-SS-III.dot`: an instance with `NN` states, computed
  from sample `III` of the target size `SS`. A minimum separating DFA can have
  fewer states than its target DFA, so `NN` and `SS` may differ.
- `instances.csv`: the case name, the number of states and the SHA-256 hash
  of each instance. The hashes match the column `input_dot_sha256` in
  `results/per-run-results.csv`.

The DOT files use the symbols `a` and `b` in place of the symbols `0` and `1`
of the DFAMiner samples. The DFAMiner license is in `DFAMINER_LICENSE.txt`.
