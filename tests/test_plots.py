from pathlib import Path

from src.benchmark import REFERENCE_ROWS
from src.plots import generate_plots


def test_generate_plots_writes_two_nonempty_deterministic_pngs(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_plots(REFERENCE_ROWS, first)
    generate_plots(REFERENCE_ROWS, second)

    for name in ["recall_vs_selectivity.png", "latency_vs_selectivity.png"]:
        first_bytes = (first / name).read_bytes()
        second_bytes = (second / name).read_bytes()
        assert len(first_bytes) > 1_000
        assert first_bytes == second_bytes
        assert first_bytes.startswith(b"\x89PNG")
