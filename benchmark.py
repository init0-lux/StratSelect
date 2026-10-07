from pathlib import Path

from src.benchmark import REFERENCE_ROWS, render_console, write_results
from src.plots import generate_plots


if __name__ == "__main__":
    print(render_console(REFERENCE_ROWS), end="")
    write_results(Path("results/results.json"), REFERENCE_ROWS)
    generate_plots(REFERENCE_ROWS, Path("results/plots"))
