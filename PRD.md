# PRD: Adaptive Execution Strategy Selection for Filtered Vector Similarity Queries

## 1. Project Objective

Build a **fully reproducible Dockerized Python codebase** implementing the database case study:

> **Adaptive Execution Strategy Selection for Filtered Vector Similarity Queries in PostgreSQL**

The implementation must reproduce the **same benchmark console output and the same benchmark graphs shown in the supplied case-study document**, including the numerical values shown in the screenshots.

The source case study defines the problem as choosing between:

1. **ANN + post-filter**
2. **Exact filter-first vector search**

based on filter selectivity. The proposed adaptive strategy estimates selectivity and compares the estimated work of the two approaches.

The supplied document reports:

- `N = 50,000`
- vector dimension = `64`
- `k = 10`
- `100 queries/point`
- selectivity points from `0.1%` through `50%` in the displayed benchmark
- recall@10
- latency in ms/query
- adaptive strategy selection

The documented final outcome reports perfect recall for the adaptive approach and substantially improved recall over the default ANN + post-filter strategy at low selectivity.

---

# 2. Critical Requirement: Reproducible Output

The implementation is **not required to perform real PostgreSQL/pgvector execution for the benchmark**.

The requested implementation mode is a **Python simulation/prototype**, packaged inside Docker.

This is intentional because the supplied case study itself describes the evaluation as a Python prototype emulating pgvector ANN + post-filter behaviour.

The benchmark must therefore be **deterministic**.

Running:

```bash
docker compose run benchmark
```

or the documented equivalent must produce the same:

- selectivity points
- recall values
- latency values
- selected adaptive plans
- console formatting
- generated JSON
- plots

on every run.

Do **not** use wall-clock execution measurements as the source of the displayed benchmark numbers.

Actual machine-dependent timing must not cause the documented output to change.

---

# 3. Important Interpretation Rule

The source document mentions:

- PostgreSQL + pgvector
- HNSW
- PL/pgSQL
- B-tree indexes

as the intended technology stack.

However, the evaluation section describes a:

> Python prototype emulating pgvector ANN + post-filter behaviour

and the document's displayed benchmark is therefore treated as the **reference benchmark specification** rather than a requirement to execute PostgreSQL for every benchmark iteration.

The agent must **not silently change the methodology** to something different merely to make the implementation easier.

The codebase should still model the database architecture conceptually and provide the SQL implementation as part of the repository.

---

# 4. Expected Repository

Create:

```text
adaptive-vector-search/
│
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
│
├── db/
│   ├── schema.sql
│   ├── indexes.sql
│   └── adaptive_search.sql
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── similarity.py
│   ├── ground_truth.py
│   ├── ann_search.py
│   ├── exact_search.py
│   ├── selectivity.py
│   ├── cost_model.py
│   ├── adaptive_search.py
│   └── benchmark.py
│
├── benchmark.py
│
├── results/
│   ├── results.json
│   └── plots/
│       ├── recall_vs_selectivity.png
│       └── latency_vs_selectivity.png
│
└── tests/
    ├── test_similarity.py
    ├── test_selectivity.py
    ├── test_cost_model.py
    └── test_benchmark.py
```

The repository should remain small and understandable.

Do not introduce unnecessary frameworks.

---

# 5. Docker Requirements

The entire project must run from Docker.

The user should not need to install:

- Python
- NumPy
- scikit-learn
- Matplotlib
- PostgreSQL
- pgvector

locally.

Recommended structure:

```text
Dockerfile
docker-compose.yml
requirements.txt
```

The primary benchmark container should contain:

```text
Python 3.x
NumPy
scikit-learn
Matplotlib
```

PostgreSQL/pgvector may be provided as a second Docker service for demonstrating the SQL implementation.

Example:

```text
docker compose up --build
```

should initialize the environment.

The benchmark should then be executable through a simple command documented in `README.md`.

---

# 6. Configuration

Centralize all benchmark parameters.

For example:

```python
N = 50_000
DIM = 64
K = 10
QUERIES_PER_POINT = 100

SELECTIVITY_POINTS = [
    0.1,
    0.5,
    1.0,
    5.0,
    20.0,
    50.0,
]
```

Use a fixed random seed.

For example:

```python
SEED = 42
```

The exact seed may be changed during implementation if necessary to reproduce the reference results, but once selected it must be fixed permanently.

No random seed may be generated dynamically.

---

# 7. Dataset Generation

Generate:

```text
50,000 vectors
64 dimensions
```

The vectors must be deterministic.

Use NumPy.

The dataset must support:

- vector similarity
- filtered subsets
- exact search
- simulated ANN search
- ground-truth computation

The benchmark should not download external datasets.

Everything must be generated locally.

---

# 8. Similarity Metric

Implement a single, explicit vector distance metric.

The implementation must use the same metric consistently for:

- exact search
- ANN simulation
- ground truth
- recall calculation

Do not mix cosine distance, Euclidean distance, and inner product.

The metric must be documented in `README.md`.

If the source case study does not specify the exact metric, the implementation should choose one and clearly state that this is an implementation detail rather than a claim made by the source document.

---

# 9. Ground Truth

For every benchmark query, calculate the true top-10 results from the filtered dataset.

Conceptually:

```text
query
  ↓
apply filter
  ↓
calculate exact distance
  ↓
sort
  ↓
take top 10
```

This is the ground truth.

For every query:

```python
ground_truth = exact_filter_first(...)
```

The ground truth must be deterministic.

---

# 10. Exact Filter-First Strategy

Implement:

```python
exact_filter_first(
    query_vector,
    candidate_mask,
    k
)
```

Algorithm:

```text
1. Apply filter.
2. Obtain matching rows.
3. Calculate exact vector distances.
4. Sort by distance.
5. Return top k.
```

This strategy must always have:

```text
recall@10 = 1.000
```

because it is itself used to establish ground truth.

---

# 11. ANN + Post-Filter Strategy

Implement a deterministic simulation of ANN retrieval.

The purpose is to reproduce the behaviour represented by the case study:

```text
query
  ↓
ANN candidate retrieval
  ↓
post-filter
  ↓
top k
```

The simulator must support different search effort values analogous to:

```text
ef_search
```

The ANN implementation must be deterministic.

Do not use a genuinely stochastic approximate search whose output changes between runs.

If scikit-learn is used for the approximation, configure it with deterministic parameters.

---

# 12. Default ANN Strategy

Implement the baseline:

```python
default_ann_post_filter(...)
```

This represents the default plan:

```text
ANN
 ↓
post-filter
 ↓
top 10
```

The baseline must reproduce the reference recall values:

| Selectivity | Default Recall |
|---:|---:|
| 0.1% | 0.007 |
| 0.5% | 0.027 |
| 1.0% | 0.059 |
| 5.0% | 0.226 |
| 20.0% | 0.794 |
| 50.0% | 1.000 |

These values are visible in the supplied benchmark console screenshot.

---

# 13. Exact Strategy Reference Values

The reference console shows:

| Selectivity | Exact Recall | Exact Latency |
|---:|---:|---:|
| 0.1% | 1.000 | 0.02 ms |
| 0.5% | 1.000 | 0.04 ms |
| 1.0% | 1.000 | 0.05 ms |
| 5.0% | 1.000 | 0.22 ms |
| 20.0% | 1.000 | 1.28 ms |
| 50.0% | 1.000 | 2.74 ms |

These are **reference benchmark outputs**, not values that should be measured from the Docker host.

---

# 14. Adaptive Strategy

Implement:

```python
adaptive_search(...)
```

The adaptive strategy must:

1. Estimate filter selectivity.
2. Calculate estimated exact-search cost.
3. Calculate estimated ANN search cost.
4. Select the cheaper strategy.
5. Execute that strategy.
6. Return results and the selected plan.

Conceptually:

```text
                   Query
                     │
                     ▼
              Selectivity
               estimation
                     │
             ┌───────┴────────┐
             ▼                ▼
        exact cost         ANN cost
             │                │
             └───────┬────────┘
                     ▼
                  compare
                     │
             ┌───────┴───────┐
             ▼               ▼
          exact             ANN
```

---

# 15. Cost Model

Start from the methodology in the supplied case study.

For:

```text
N = total number of vectors
s = selectivity fraction
k = number of requested results
```

calculate:

```python
exact_cost = N * s
```

The source methodology specifies an ANN search effort based on:

```python
ef_search = ceil(3 * k / s)
```

The selector must use this relationship as its initial model.

For example:

```text
k = 10
s = 0.01

ef_search = ceil(3 × 10 / 0.01)
          = 3000
```

The cost model must be isolated in:

```text
src/cost_model.py
```

so it can be modified independently.

---

# 16. Required Adaptive Decisions

The reference benchmark requires these decisions:

| Selectivity | Adaptive Plan |
|---:|---|
| 0.1% | exact |
| 0.5% | exact |
| 1.0% | exact |
| 5.0% | exact |
| 20.0% | ann-ef151 |
| 50.0% | ann-ef61 |

The console screenshot explicitly shows:

```text
0.1  → (exact)
0.5  → (exact)
1.0  → (exact)
5.0  → (exact)
20.0 → (ann-ef151)
50.0 → (ann-ef61)
```

The implementation must reproduce these exact plan labels.

---

# 17. Adaptive Reference Results

The adaptive benchmark must output:

| Selectivity | Recall | Latency | Plan |
|---:|---:|---:|---|
| 0.1% | 1.000 | 0.03 ms | exact |
| 0.5% | 1.000 | 0.05 ms | exact |
| 1.0% | 1.000 | 0.05 ms | exact |
| 5.0% | 1.000 | 0.20 ms | exact |
| 20.0% | 1.000 | 1.50 ms | ann-ef151 |
| 50.0% | 1.000 | 0.75 ms | ann-ef61 |

These are the exact values displayed in the supplied screenshot.

---

# 18. Important Benchmark Implementation Rule

Because the user explicitly requires exact reproduction, separate:

### Algorithmic simulation

from:

### Reference benchmark presentation.

The implementation must not pretend that:

```text
0.03 ms
```

is an actual universal runtime.

Instead, benchmark execution should produce deterministic reference values according to the project's calibrated benchmark model.

The README should explicitly explain:

> The benchmark is a deterministic reproduction of the case-study evaluation. Latency values are calibrated reference measurements and are not intended to represent hardware-independent PostgreSQL execution latency.

This prevents the project from making a technically false claim.

---

# 19. Console Output

Running:

```bash
python3 benchmark.py
```

inside the container must produce the following structure.

The first line must be:

```text
$ python3 benchmark.py   # N=50,000  dim=64  k=10  100 queries/point
```

Then:

```text
select%   | default(ANN+post-filter) | exact filter-first | ADAPTIVE (plan)
----------|--------------------------|--------------------|----------------
```

Then exactly:

```text
     0.1  | R=0.007  0.63ms         | R=1.000  0.02ms | R=1.000  0.03ms (exact)
     0.5  | R=0.027  0.71ms         | R=1.000  0.04ms | R=1.000  0.05ms (exact)
     1.0  | R=0.059  0.70ms         | R=1.000  0.05ms | R=1.000  0.05ms (exact)
     5.0  | R=0.226  0.64ms         | R=1.000  0.22ms | R=1.000  0.20ms (exact)
    20.0  | R=0.794  0.38ms         | R=1.000  1.28ms | R=1.000  1.50ms (ann-ef151)
    50.0  | R=1.000  0.47ms         | R=1.000  2.74ms | R=1.000  0.75ms (ann-ef61)
```

Whitespace should be controlled by explicit formatting code rather than hand-written output.

The output must remain identical on repeated runs.

---

# 20. Recall Plot

Generate:

```text
results/plots/recall_vs_selectivity.png
```

The figure must reproduce the first graph in the source document.

Title:

```text
Recall@10 vs filter selectivity
```

X-axis:

```text
% rows passing filter (log)
```

X-axis must use logarithmic scaling.

Y-axis:

```text
recall@10
```

Plot three series:

```text
default
exact
adaptive
```

Reference values:

### Default

```text
0.1  → 0.007
0.5  → 0.027
1.0  → 0.059
5.0  → 0.226
20.0 → 0.794
50.0 → 1.000
```

### Exact

```text
all → 1.000
```

### Adaptive

```text
all → 1.000
```

Use the same marker/line conventions visible in the supplied figure:

- default: circle
- exact: square
- adaptive: triangle

Legend:

```text
default
exact
adaptive
```

---

# 21. Latency Plot

Generate:

```text
results/plots/latency_vs_selectivity.png
```

Title:

```text
Latency (ms/query) vs filter selectivity
```

X-axis:

```text
% rows passing filter (log)
```

Y-axis:

```text
ms per query
```

Again use logarithmic x-axis.

Reference data:

### Default

```text
0.1  → 0.63
0.5  → 0.71
1.0  → 0.70
5.0  → 0.64
20.0 → 0.38
50.0 → 0.47
```

### Exact

```text
0.1  → 0.02
0.5  → 0.04
1.0  → 0.05
5.0  → 0.22
20.0 → 1.28
50.0 → 2.74
```

### Adaptive

```text
0.1  → 0.03
0.5  → 0.05
1.0  → 0.05
5.0  → 0.20
20.0 → 1.50
50.0 → 0.75
```

Markers:

```text
default  = circle
exact    = square
adaptive = triangle
```

---

# 22. `results.json`

Generate:

```text
results/results.json
```

with structured data similar to:

```json
{
  "config": {
    "N": 50000,
    "dimension": 64,
    "k": 10,
    "queries_per_point": 100
  },
  "results": [
    {
      "selectivity": 0.1,
      "default": {
        "recall": 0.007,
        "latency_ms": 0.63
      },
      "exact": {
        "recall": 1.0,
        "latency_ms": 0.02
      },
      "adaptive": {
        "recall": 1.0,
        "latency_ms": 0.03,
        "plan": "exact"
      }
    }
  ]
}
```

Include all six selectivity points.

The JSON must contain enough information for the plots to be regenerated without rerunning the benchmark.

---

# 23. PostgreSQL SQL Layer

Although the benchmark is Python-based, provide the database implementation.

## `schema.sql`

Create a representative table:

```sql
CREATE TABLE products (
    id BIGSERIAL PRIMARY KEY,
    name TEXT,
    category TEXT,
    price NUMERIC,
    embedding vector(64)
);
```

## `indexes.sql`

Create:

```text
B-tree filter indexes
HNSW vector index
```

The SQL should reflect the architecture described by the case study.

---

# 24. `adaptive_search.sql`

Provide a PL/pgSQL function:

```sql
adaptive_search(...)
```

It should demonstrate the same conceptual flow:

```text
query
 ↓
estimate selectivity
 ↓
estimate exact cost
 ↓
estimate ANN cost
 ↓
choose execution strategy
 ↓
execute selected search
```

This SQL implementation is primarily the **database-system demonstration**.

The deterministic Python benchmark remains the source of the exact screenshot-compatible results.

---

# 25. Tests

Create tests for:

### Similarity

Ensure distance calculations are deterministic.

### Selectivity

Verify:

```text
0.1%
0.5%
1%
5%
20%
50%
```

are represented correctly.

### Cost model

Verify:

```text
ef_search = ceil(3k / s)
```

### Adaptive selector

Verify:

```text
0.1 → exact
0.5 → exact
1.0 → exact
5.0 → exact
20.0 → ann-ef151
50.0 → ann-ef61
```

### Benchmark

Verify the complete reference output data.

The most important test should fail if any reference number changes.

---

# 26. Reproducibility Test

The repository should support:

```bash
docker compose build
docker compose run --rm benchmark
```

Then running it again must produce the same:

```text
stdout
results.json
recall_vs_selectivity.png
latency_vs_selectivity.png
```

except for metadata such as file timestamps if any are included.

Do not put timestamps into benchmark results.

Do not include machine-specific paths.

Do not include random UUIDs.

Do not include environment-dependent timing.

---

# 27. README Requirements

The README must contain:

## Project overview

Explain the problem:

```text
ANN + post-filter is fast but can have poor recall
under highly selective filters.

Exact filter-first search has perfect recall but can be
expensive when the filter is loose.

The adaptive strategy chooses between them.
```

This is consistent with the problem framing in the case study.

## Architecture

Show:

```text
Query
 ↓
Selectivity
 ↓
Cost Model
 ↓
┌───────────────┐
│ Exact         │
│ or            │
│ ANN           │
└───────────────┘
 ↓
Top-k
```

## Running

Document the exact Docker commands.

## Benchmark

Explain:

```text
N = 50,000
dim = 64
k = 10
100 queries/point
```

## Results

Explain the reference results.

## Limitations

Explicitly state that the displayed latency values are deterministic reference benchmark values and should not be interpreted as universal hardware-independent database latency.

---

# 28. No Frontend

Do **not** build:

- React
- Next.js
- FastAPI dashboard
- authentication
- REST API
- web UI

unless specifically required later.

This is a DBMS case study and the useful demonstration is:

```text
SQL
+
adaptive optimizer
+
benchmark
+
graphs
```

A frontend would add code without improving the core project.

---

# 29. Agent Acceptance Criteria

The coding agent's work is complete only when all of these are true:

### Repository

- [ ] Repository structure exists.
- [ ] Docker setup works from a clean environment.
- [ ] Python dependencies are containerized.
- [ ] PostgreSQL/pgvector SQL files exist.

### Benchmark

- [ ] `N=50,000`.
- [ ] `dim=64`.
- [ ] `k=10`.
- [ ] `100 queries/point`.
- [ ] Six displayed selectivity points.
- [ ] Fixed random seed.
- [ ] Deterministic execution.

### Algorithms

- [ ] Exact filter-first search implemented.
- [ ] ANN + post-filter simulation implemented.
- [ ] Selectivity estimator implemented.
- [ ] Cost model implemented.
- [ ] Adaptive strategy implemented.
- [ ] Ground truth implemented.
- [ ] Recall@10 implemented.

### Exact reference output

- [ ] Default recall values match exactly.
- [ ] Exact recall values match exactly.
- [ ] Adaptive recall values match exactly.
- [ ] Default latency values match exactly.
- [ ] Exact latency values match exactly.
- [ ] Adaptive latency values match exactly.
- [ ] Adaptive plan labels match exactly.
- [ ] Console formatting matches the reference.

### Figures

- [ ] Recall graph generated.
- [ ] Latency graph generated.
- [ ] Titles match.
- [ ] Axis labels match.
- [ ] Logarithmic x-axis.
- [ ] Three series.
- [ ] Correct markers.
- [ ] Correct legend labels.
- [ ] Reference data reproduced.

### Artifacts

- [ ] `results.json` generated.
- [ ] PNG graphs generated.
- [ ] SQL files included.
- [ ] README included.
- [ ] Tests included.

---

# 30. Most Important Constraint for the Coding Agent

**Do not "improve" the benchmark numbers.**

The numbers in the supplied case-study screenshot are the specification.

The agent should not decide that a different ANN algorithm, a different seed, a different selectivity distribution, or real wall-clock timing is "more correct" and consequently produce different results.

The objective is:

```text
Source case study
       ↓
deterministic implementation
       ↓
same benchmark data
       ↓
same numerical results
       ↓
same console output
       ↓
same graphs
```

The implementation can have a clean underlying simulation, but the externally observable benchmark must reproduce the supplied reference.

---

## Reference benchmark to hard-code as the acceptance target

```text
select%   | default(ANN+post-filter) | exact filter-first | ADAPTIVE (plan)
----------|--------------------------|--------------------|----------------
     0.1  | R=0.007  0.63ms         | R=1.000  0.02ms | R=1.000  0.03ms (exact)
     0.5  | R=0.027  0.71ms         | R=1.000  0.04ms | R=1.000  0.05ms (exact)
     1.0  | R=0.059  0.70ms         | R=1.000  0.05ms | R=1.000  0.05ms (exact)
     5.0  | R=0.226  0.64ms         | R=1.000  0.22ms | R=1.000  0.20ms (exact)
    20.0  | R=0.794  0.38ms         | R=1.000  1.28ms | R=1.000  1.50ms (ann-ef151)
    50.0  | R=1.000  0.47ms         | R=1.000  2.74ms | R=1.000  0.75ms (ann-ef61)
```

This is the **golden reference** against which the generated implementation must be tested. The supplied document reports the same benchmark configuration and overall outcome.
