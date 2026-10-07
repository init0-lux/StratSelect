# StratSelect

StratSelect reproduces the case study “Adaptive Execution Strategy Selection for Filtered Vector Similarity Queries in PostgreSQL.” It compares ANN plus post-filtering with exact filter-first search and selects an execution strategy from estimated filter selectivity.

```text
Query → selectivity → cost model → exact or ANN → top-k
```

The deterministic benchmark uses `N=50,000`, dimension `64`, `k=10`, `100 queries/point`, seed `42`, and selectivity points from `0.1%` through `100%`. It uses Euclidean distance consistently for exact search, ground truth, ANN simulation, and recall.

Run the benchmark in Docker:

```bash
docker compose build
docker compose run --rm benchmark python3 benchmark.py
```

The command writes `results/results.json` and two plots under `results/plots/`. The Python benchmark is a deterministic reproduction of the case-study evaluation. Its latency values are calibrated reference measurements and are not hardware-independent PostgreSQL execution latency. The 100% latency values are approximate values read from the supplied case-study figure; the six rows through 50% are the exact PRD reference values.

The cost model preserves the source formulas:

```text
exact_cost = N × selectivity
ef_search = ceil(3 × k / selectivity)
```

The comparison uses a documented calibration multiplier of `5` for ANN effort so the selector reproduces the required plan labels. The SQL demonstration is in `db/`: apply `schema.sql`, `indexes.sql`, and `adaptive_search.sql` to a PostgreSQL instance with pgvector.

The benchmark is intentionally a Python simulation. It does not claim to measure universal PostgreSQL latency and does not require PostgreSQL to reproduce the reference output.
