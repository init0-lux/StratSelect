# StratSelect

StratSelect reproduces the case study “Adaptive Execution Strategy Selection for Filtered Vector Similarity Queries in PostgreSQL.” It compares ANN plus post-filtering with exact filter-first search and selects an execution strategy from estimated filter selectivity.

```text
Query → selectivity → cost model → exact or ANN → top-k
```

## Why

pgvector's HNSW index fetches the `ef_search` nearest vectors (40 by default) and applies the `WHERE` clause afterwards. With selectivity `s`, about `40 × s` candidates survive, so a strict filter silently returns too few rows: at 1% selectivity, recall@10 is about 0.05. StratSelect scans the matching rows exactly when the filter is strict and uses ANN with a widened `ef_search` when it is not, keeping recall@10 at 1.00 at every tested selectivity.

## Quick start

All commands run in Docker:

```bash
docker compose build
docker compose run --rm benchmark                                   # 30 tests
docker compose run --rm benchmark python3 stratselect_tui.py        # interactive app
docker compose run --rm benchmark python3 live_benchmark.py         # live recall sweep
docker compose run --rm benchmark python3 demo.py --tour            # plain-text walkthrough
docker compose run --rm benchmark python3 benchmark.py              # reference table + plots
```

| Entry point | What it does |
| --- | --- |
| `stratselect_tui.py` | Interactive app. Pick a category and max price (or a preset); it shows selectivity, both costs, the chosen plan and recall vs exact ground truth. The **Sweep** tab (`s`) recomputes recall at every selectivity point; the **Tests** tab (`t`) runs the suite with a description per module. |
| `live_benchmark.py` | Recomputes recall@10 for default ANN (`ef_search=40`), exact and adaptive over 100 seeded queries per point. Work is reported in vectors compared. |
| `demo.py` | The same single-query walkthrough as plain terminal output; `--tour` runs four preset filters with no input. |
| `benchmark.py` | Writes the case-study reference table to `results/results.json` and two plots under `results/plots/`. |

Live run (recall@10):

| Selectivity | Default (ef=40) | Exact | Adaptive | Plan |
| --- | --- | --- | --- | --- |
| 0.1% | 0.006 | 1.000 | 1.000 | exact |
| 1% | 0.050 | 1.000 | 1.000 | exact |
| 5% | 0.229 | 1.000 | 1.000 | exact |
| 20% | 0.789 | 1.000 | 1.000 | ANN, ef=150 |
| 100% | 1.000 | 1.000 | 1.000 | ANN, ef=30 |

Default recall tracks `ef × s / k`.

## Method

The deterministic benchmark uses `N=50,000`, dimension `64`, `k=10`, `100 queries/point`, seed `42`, and selectivity points from `0.1%` through `100%`. It uses Euclidean distance consistently for exact search, ground truth, ANN simulation, and recall.

The cost model preserves the source formulas:

```text
exact_cost = N × selectivity
ef_search  = ceil(3 × k / selectivity)
ann_cost   = 5 × ef_search
```

`ef_search = 3k/s` leaves about `3k` candidates after filtering, enough for top-k. The calibration multiplier of `5` accounts for an ANN step costing more than one exact comparison; it moves the crossover (about 5.5% here) and does not affect recall.

## Limits

- The ANN is simulated as a perfect ANN (true top-`ef_search`, then filter), so all recall loss comes from post-filtering. Real HNSW would do slightly worse.
- The Python benchmark is a deterministic reproduction of the case-study evaluation. Its latency values in `results/results.json` are calibrated reference measurements and are not hardware-independent PostgreSQL execution latency. The 100% latency values are approximate values read from the supplied case-study figure; the six rows through 50% are the exact PRD reference values.
- The data is synthetic, and filters are independent of the embeddings.

## PostgreSQL

The SQL demonstration is in `db/`: apply `schema.sql`, `indexes.sql`, and `adaptive_search.sql` to a PostgreSQL instance with pgvector (`docker compose up postgres` starts one). `adaptive_search()` counts matching rows, compares the two costs, and either scans exactly or sets `hnsw.ef_search` before using the HNSW index.

The benchmark does not require PostgreSQL to reproduce the reference output.
