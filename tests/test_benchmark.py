import json
import subprocess
import sys

from src.benchmark import REFERENCE_ROWS, render_console, write_results


def test_reference_rows_include_exact_six_points_and_docx_100_percent_point():
    assert [row["selectivity"] for row in REFERENCE_ROWS] == [0.1, 0.5, 1.0, 5.0, 20.0, 50.0, 100.0]
    assert REFERENCE_ROWS[0]["default"]["recall"] == 0.007
    assert REFERENCE_ROWS[5]["adaptive"]["plan"] == "ann-ef61"
    assert REFERENCE_ROWS[6]["adaptive"]["plan"] == "ann-ef31"


def test_console_contains_required_prd_rows():
    output = render_console(REFERENCE_ROWS)
    assert "$ python3 benchmark.py   # N=50,000  dim=64  k=10  100 queries/point" in output
    assert "     0.1  | R=0.007  0.63ms         | R=1.000  0.02ms | R=1.000  0.03ms (exact)" in output
    assert "    50.0  | R=1.000  0.47ms         | R=1.000  2.74ms | R=1.000  0.75ms (ann-ef61)" in output


def test_write_results_has_plot_ready_json(tmp_path):
    path = tmp_path / "results.json"
    write_results(path, REFERENCE_ROWS)
    payload = json.loads(path.read_text())

    assert payload["config"] == {"N": 50000, "dimension": 64, "k": 10, "queries_per_point": 100}
    assert len(payload["results"]) == 7


def test_benchmark_script_is_repeatable():
    first = subprocess.check_output([sys.executable, "benchmark.py"], text=True)
    second = subprocess.check_output([sys.executable, "benchmark.py"], text=True)
    assert first == second
