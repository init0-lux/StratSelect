from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot(rows: list[dict], metric: str, title: str, ylabel: str, path: Path) -> None:
    x = [row["selectivity"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    for name, marker in [("default", "o"), ("exact", "s"), ("adaptive", "^")]:
        ax.plot(x, [row[name][metric] for row in rows], marker=marker, label=name)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("% rows passing filter (log)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, format="png", metadata={"Software": "StratSelect"})
    plt.close(fig)


def generate_plots(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot(rows, "recall", "Recall@10 vs filter selectivity", "recall@10", output_dir / "recall_vs_selectivity.png")
    _plot(rows, "latency_ms", "Latency (ms/query) vs filter selectivity", "ms per query", output_dir / "latency_vs_selectivity.png")
