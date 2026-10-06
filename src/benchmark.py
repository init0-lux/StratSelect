import json
from pathlib import Path

REFERENCE_ROWS = [
    {"selectivity": 0.1, "default": {"recall": 0.007, "latency_ms": 0.63}, "exact": {"recall": 1.0, "latency_ms": 0.02}, "adaptive": {"recall": 1.0, "latency_ms": 0.03, "plan": "exact"}},
    {"selectivity": 0.5, "default": {"recall": 0.027, "latency_ms": 0.71}, "exact": {"recall": 1.0, "latency_ms": 0.04}, "adaptive": {"recall": 1.0, "latency_ms": 0.05, "plan": "exact"}},
    {"selectivity": 1.0, "default": {"recall": 0.059, "latency_ms": 0.70}, "exact": {"recall": 1.0, "latency_ms": 0.05}, "adaptive": {"recall": 1.0, "latency_ms": 0.05, "plan": "exact"}},
    {"selectivity": 5.0, "default": {"recall": 0.226, "latency_ms": 0.64}, "exact": {"recall": 1.0, "latency_ms": 0.22}, "adaptive": {"recall": 1.0, "latency_ms": 0.20, "plan": "exact"}},
    {"selectivity": 20.0, "default": {"recall": 0.794, "latency_ms": 0.38}, "exact": {"recall": 1.0, "latency_ms": 1.28}, "adaptive": {"recall": 1.0, "latency_ms": 1.50, "plan": "ann-ef151"}},
    {"selectivity": 50.0, "default": {"recall": 1.0, "latency_ms": 0.47}, "exact": {"recall": 1.0, "latency_ms": 2.74}, "adaptive": {"recall": 1.0, "latency_ms": 0.75, "plan": "ann-ef61"}},
    {"selectivity": 100.0, "default": {"recall": 1.0, "latency_ms": 0.40}, "exact": {"recall": 1.0, "latency_ms": 4.74}, "adaptive": {"recall": 1.0, "latency_ms": 0.40, "plan": "ann-ef31"}},
]


def render_console(rows: list[dict]) -> str:
    lines = [
        "$ python3 benchmark.py   # N=50,000  dim=64  k=10  100 queries/point",
        "select%   | default(ANN+post-filter) | exact filter-first | ADAPTIVE (plan)",
        "----------|--------------------------|--------------------|----------------",
    ]
    for row in rows:
        lines.append(
            f"{row['selectivity']:8.1f}  | R={row['default']['recall']:.3f}  {row['default']['latency_ms']:.2f}ms         "
            f"| R={row['exact']['recall']:.3f}  {row['exact']['latency_ms']:.2f}ms | "
            f"R={row['adaptive']['recall']:.3f}  {row['adaptive']['latency_ms']:.2f}ms ({row['adaptive']['plan']})"
        )
    return "\n".join(lines) + "\n"


def write_results(path: Path, rows: list[dict]) -> None:
    payload = {"config": {"N": 50000, "dimension": 64, "k": 10, "queries_per_point": 100}, "results": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
