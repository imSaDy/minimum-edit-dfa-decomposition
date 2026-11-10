# Minimum-Edit Decomposition of Deterministic Finite Automata

This repository contains the implementation, the benchmark and the experimental
results of the paper

> Sadegh Samadi, Mahboubeh Samadi, Hossein Hojjat.
> *Minimum-Edit Decomposition of Deterministic Finite Automata.*
> Submitted to FSEN 2027.

Given a DFA `A`, the tool finds an edited DFA `A_edt`, obtained from `A` with
the minimum number of edits (redirected transitions and flipped acceptance
flags), together with two factors `A1` and `A2` that form a strict
decomposition of `A_edt`:

```text
L(A_edt) = L(A1) ∩ L(A2) ≠ ∅,   L(A_edt) ⊊ L(A1),   L(A_edt) ⊊ L(A2),
2 ≤ |A1| ≤ |A2| < |A|.
```

The problem is encoded in SAT. Three procedures search for an optimal
solution: a linear search over the edit budget, MaxSAT with RC2, and an
enumeration of edited DFAs. Every solution is checked independently of the
encoding before it is reported.

## Repository layout

| Path | Contents |
| --- | --- |
| `dfa_identify/` | The solver. `minimum_edit.py` contains the encoding and the three procedures, `cli.py` the command-line interface, `dot_io.py` DOT input and output. |
| `tests/` | Unit tests. Run them with `poetry run python -m unittest discover -s tests`. |
| `examples/` | The running example of the paper (`trailing_b_mod_3.dot`). |
| `benchmark/` | The 693 benchmark instances used in the paper, with a manifest of their SHA-256 hashes. |
| `results/` | One row per run of the experiments and the exact configuration of each run (see `results/README.md`). |
| `analysis/` | `paper_numbers.py` recomputes the numbers reported in Sect. 6 of the paper from `results/`. |
| `remote/`, `scripts/` | The experiment harness exactly as it ran on the server (see `remote/README.md`). |
| `LICENSES/`, `THIRD_PARTY_NOTICES.md` | Licenses of the code and data this repository builds on. |

## Installation

Python 3.11 or later is required. With [Poetry](https://python-poetry.org/):

```sh
poetry install
```

The experiments used PySAT 1.9.dev15 (CaDiCaL 3.0 is available in PySAT as
`cadical300`). The exact package versions of the server are listed in
`remote/requirements-lock.txt`. Graphviz is needed only for the optional PNG
drawings of the automata.

## Usage

Solve one instance with the default configuration of the paper:

```sh
poetry run minimum-edit-dfa examples/trailing_b_mod_3.dot \
  --optimization sat --sat-solver g3 --factor-order max \
  --product-encoding factored --bfs-encoding classic \
  --bfs-parent-redundancy --strictness --strictness-mode finite \
  -o result.json
```

The defaults of the command-line tool differ from the default configuration of
the paper, so give the options explicitly. `results/configurations.csv` lists
the options of all configurations. Run
`poetry run minimum-edit-dfa --help` for a description of each option.

## Configurations

The paper reports 11 configurations: the default configuration and ten
configurations that each change one choice (MaxSAT, enumeration, sum order,
product order, lazy strictness, strongly strict, CaDiCaL, MapleCM, MiniSat,
direct product closure). The results in this repository contain three further
configurations that the paper does not report: no redundant parent clauses,
compact BFS, and compact BFS without redundant parent clauses.

## Reproducing the paper

- **Numbers in Sect. 6:** `python analysis/paper_numbers.py` prints the shares
  of solved instances, the optimal costs and the running-time comparisons
  reported in the paper. It also prints the same numbers for the three
  additional configurations and a peak-memory comparison of the two closure
  encodings, which are not in the paper. It needs only the standard library.
- **Single runs:** run the tool on an instance from `benchmark/instances/`
  with the options of a configuration from `results/configurations.csv`. Each
  run of the paper had a time limit of 8 hours and a memory limit of 16 GiB.
- **Full campaign:** the experiments ran on a Linux server with 54 parallel
  workers, each pinned to one logical CPU. `remote/README.md` describes the
  harness.

## Code versions

The experiments ran in three campaigns. The server recorded the SHA-256 hash
of the solver file `dfa_identify/minimum_edit.py` in each campaign:

| Campaign | Configurations | SHA-256 of `minimum_edit.py` |
| --- | --- | --- |
| reference | default, MaxSAT, enumeration, CaDiCaL, MapleCM, MiniSat, strongly strict, no redundant parent clauses, direct product closure, sum order, product order | `a2d06d18a19b…` |
| supplement | compact BFS, with and without redundant parent clauses | `c70ceb0cc037…` |
| blocking | lazy strictness | `e7e4406af4dc…` |

The file in this repository is the version of the blocking campaign
(`e7e4406af4dc…`). It differs from the supplement version only in the code for
lazy strictness and its counters. The versions of the reference and supplement
campaigns are not included. The `campaign_id` column of
`results/per-run-results.csv` identifies the campaign of each run.

The hashes above were computed on the file with Windows line endings (CRLF),
as it was deployed to the server. Git stores the file with Unix line endings
(LF), for which the SHA-256 hash is `a537f423f289…`. Converting the line
endings to CRLF gives `e7e4406af4dc…` again.

## License

The code in this repository is released under the MIT License (`LICENSE`).
The modules `active.py`, `concept_class_restrictions.py`, `decomposed.py`,
`encoding.py`, `graphs.py` and `identify.py` in `dfa_identify/` come unchanged
from [dfa-identify](https://github.com/mvcisback/dfa-identify) by Marcell
Vazquez-Chanlatte et al. (MIT License). The benchmark instances are derived
from the samples of the [DFAMiner](https://github.com/liyong31/DFAMiner)
benchmark (MIT License). See `THIRD_PARTY_NOTICES.md`.

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`).
