"""Interactive StratSelect app: build a filter, watch the planner pick a strategy.

    python3 stratselect_tui.py
"""

import re
import subprocess
import sys

import numpy as np
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from src.ann_search import ann_post_filter
from src.config import DIM, K, N, QUERIES_PER_POINT, SEED, SELECTIVITY_POINTS
from src.cost_model import CALIBRATION_MULTIPLIER, ann_estimated_cost, ef_search, exact_cost
from src.dataset import generate_dataset
from src.exact_search import exact_filter_first
from src.ground_truth import ground_truth

DEFAULT_EF = 40  # pgvector's default hnsw.ef_search
CATEGORIES = ("any", "shoes", "books", "home", "tech")
PRESETS = [
    ("Needle: shoes < 300", "shoes", 300),
    ("Crossover: tech < 2100", "tech", 2100),
    ("Broad: any < 2100", "any", 2100),
    ("No filter", "any", 10_001),
]
TEST_EXPLAIN = {
    "test_similarity": "Euclidean distance + exact search are correct",
    "test_selectivity": "selectivity estimate from the filter mask",
    "test_cost_model": "exact_cost = N·s,  ef_search = ⌈3k/s⌉",
    "test_ann_search": "ANN + post-filter loses rows under strict filters",
    "test_adaptive": "planner picks exact / ANN at the right selectivities",
    "test_benchmark": "benchmark table + results.json shape",
    "test_plots": "plots render reproducibly",
    "test_sql_artifacts": "pgvector schema, HNSW index and SQL function exist",
}
ACCENT, GOOD, WARN, BAD = "bright_cyan", "bright_green", "bright_yellow", "bright_red"


def bar(fraction, width=24, color=ACCENT):
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    return Text("█" * filled, style=color) + Text("░" * (width - filled), style="grey46")


def recall_color(r):
    return GOOD if r >= 0.99 else WARN if r >= 0.5 else BAD


def recall(result, truth):
    return len(set(result) & set(truth)) / len(truth) if truth else 1.0


def stages(states):
    icons = {"done": ("✔", GOOD), "running": ("●", ACCENT), "failed": ("✘", BAD), "pending": ("○", "grey70")}
    table = Table.grid(padding=(0, 1))
    for stage, (state, note) in states.items():
        icon, color = icons[state]
        table.add_row(Text(icon, style=color), Text(stage, style="bold"), Text(note, style="grey82"))
    return table


class StratSelectApp(App):
    TITLE = "StratSelect"
    SUB_TITLE = "Adaptive execution strategy for filtered vector search"
    CSS = """
    Screen { background: #1e2230; color: #f2f4f8; }
    Header { background: #2f6fde; color: #ffffff; text-style: bold; }
    Footer { background: #2a3042; }
    #sidebar { width: 38; background: #262b3b; border-right: solid #5b8def; padding: 0 1; }
    #sidebar Label { margin-top: 1; color: #9ecbff; text-style: bold; }
    #sidebar Select > SelectCurrent, #sidebar Input { background: #323a50; color: #ffffff; border: tall #5b8def; }
    #sidebar Button { width: 100%; margin-top: 1; }
    #run { background: #2f6fde; color: #ffffff; text-style: bold; }
    .preset { margin-top: 0 !important; background: #3a4560; color: #ffffff; border: tall #5b6b8f; }
    .preset:hover, .preset:focus { background: #4b5b80; }
    TabbedContent Tabs { background: #262b3b; }
    Tab { color: #c9d1e0; }
    Tab.-active { color: #ffffff; text-style: bold; }
    TabPane { padding: 0 1; }
    .hint { color: #d7dce6; padding: 1 0; }
    #sweep-progress, #tests-progress { margin: 1 0; }
    DataTable { height: 1fr; background: #1e2230; }
    DataTable > .datatable--header { background: #2f6fde; color: #ffffff; text-style: bold; }
    """
    BINDINGS = [
        Binding("r", "run_query", "Run query"),
        Binding("s", "sweep", "Sweep"),
        Binding("t", "tests", "Tests"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.data = generate_dataset(N, DIM, SEED)
        self.rng = np.random.default_rng(SEED + 7)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("category")
                yield Select([(c, c) for c in CATEGORIES], value="shoes", allow_blank=False, id="category")
                yield Label("max price (100 – 10001)")
                yield Input("300", id="price", type="integer")
                yield Button("Run query  [r]", id="run", variant="primary")
                yield Label("presets")
                for i, (name, _, _) in enumerate(PRESETS):
                    yield Button(name, id=f"preset-{i}", classes="preset")
            with TabbedContent(id="center"):
                with TabPane("Query", id="query"):
                    yield VerticalScroll(Static("Pick a filter on the left and press [b]r[/b].", id="query-view", classes="hint"))
                with TabPane("Sweep", id="sweep"):
                    yield Static(f"Press [b]s[/b] to recompute recall@{K} over {QUERIES_PER_POINT} queries at every selectivity point.", id="sweep-summary", classes="hint")
                    yield ProgressBar(id="sweep-progress", show_eta=False)
                    yield DataTable(id="sweep-table", zebra_stripes=True)
                with TabPane("Tests", id="tests"):
                    yield Static("Press [b]t[/b] to run the test suite.", id="tests-summary", classes="hint")
                    yield ProgressBar(id="tests-progress", show_eta=False)
                    yield DataTable(id="tests-table", zebra_stripes=True)
        yield Footer()

    def on_mount(self):
        self.sub_title = f"{self.SUB_TITLE} · N={N:,} dim={DIM} k={K} seed={SEED}"
        sweep = self.query_one("#sweep-table", DataTable)
        sweep.add_columns("selectivity", "default recall (ef=40)", "exact recall", "exact work", "adaptive recall", "plan")
        tests = self.query_one("#tests-table", DataTable)
        tests.add_columns("", "module", "what it checks", "test")
        for widget in ("#sweep-progress", "#tests-progress", "#sweep-table", "#tests-table"):
            self.query_one(widget).display = False

    # ---------- single query ----------

    @on(Button.Pressed, "#run")
    @on(Input.Submitted, "#price")
    def action_run_query(self):
        category = self.query_one("#category", Select).value
        try:
            price = max(101, min(10_001, int(self.query_one("#price", Input).value)))
        except ValueError:
            price = 10_001
        self.query_one("#center", TabbedContent).active = "query"
        self.run_query(category, price)

    @on(Button.Pressed, ".preset")
    def preset(self, event):
        _, category, price = PRESETS[int(event.button.id.split("-")[1])]
        self.query_one("#category", Select).value = category
        self.query_one("#price", Input).value = str(price)
        self.action_run_query()

    def show_query(self, renderable):
        widget = self.query_one("#query-view", Static)
        widget.remove_class("hint")
        widget.update(renderable)

    @work(thread=True, exclusive=True, group="query")
    def run_query(self, category, price):
        import time

        mask = self.data.prices < price
        if category != "any":
            mask &= self.data.categories == category
        where = f"price < {price}" + ("" if category == "any" else f" AND category = '{category}'")
        sql = Text.assemble(("SELECT … WHERE ", "grey82"), (where, "bold"), (f" ORDER BY embedding <-> $q LIMIT {K}", "grey82"))
        states = {
            "Estimate selectivity": ("running", "counting rows that pass the filter…"),
            "Cost model": ("pending", ""),
            "Plan decision": ("pending", ""),
            "Execute + ground truth": ("pending", ""),
        }
        panels = []

        def render():
            self.call_from_thread(self.show_query, Group(sql, Text(""), stages(states), Text(""), *panels))
            time.sleep(0.35)

        render()
        matching = int(mask.sum())
        if matching == 0:
            states["Estimate selectivity"] = ("failed", "0 rows match: nothing to search")
            render()
            return
        s = matching / N
        states["Estimate selectivity"] = ("done", f"{matching:,} / {N:,} rows  →  s = {s:.2%}")
        states["Cost model"] = ("running", "pricing both strategies…")
        render()

        exact, ef, ann = exact_cost(N, s), ef_search(K, s), ann_estimated_cost(K, s)
        plan = "exact" if exact <= ann else "ann"
        top = max(exact, ann)
        costs = Table.grid(padding=(0, 1))
        costs.add_row(Text("exact filter-first", style="bold"), bar(exact / top, color=GOOD if plan == "exact" else "grey70"), Text(f"{exact:>9,.0f}  N·s"))
        costs.add_row(Text("ANN + post-filter", style="bold"), bar(ann / top, color=GOOD if plan == "ann" else "grey70"), Text(f"{ann:>9,.0f}  {CALIBRATION_MULTIPLIER}·⌈3k/s⌉  (ef={ef:,})"))
        panels.append(Panel(Group(costs, Text(""), Text(f"ef = 3k/s → ~{3 * K} candidates survive the filter on average: enough for top-{K}.", style="grey82")),
                            title="Cost model", title_align="left", border_style=ACCENT))
        states["Cost model"] = ("done", f"exact {exact:,.0f}  vs  ANN {ann:,.0f}")
        states["Plan decision"] = ("running", "")
        render()

        decision = (Text.assemble(("EXACT", f"bold {GOOD}"), " · brute-force the matching rows: the filter is strict enough that this is cheaper")
                    if plan == "exact" else
                    Text.assemble((f"ANN  ef_search={ef}", f"bold {GOOD}"), " · graph search, with ef widened so enough rows survive the filter"))
        panels.append(Panel(decision, title="Plan", title_align="left", border_style=GOOD))
        states["Plan decision"] = ("done", "exact" if plan == "exact" else f"ann (ef={ef})")
        states["Execute + ground truth"] = ("running", "running default pgvector, adaptive and exact ground truth…")
        render()

        query = self.rng.standard_normal(DIM)
        truth = ground_truth(query, self.data.vectors, mask, K)
        default = ann_post_filter(query, self.data.vectors, mask, K, DEFAULT_EF)
        adaptive = (exact_filter_first(query, self.data.vectors, mask, K) if plan == "exact"
                    else ann_post_filter(query, self.data.vectors, mask, K, ef))
        results = Table.grid(padding=(0, 1))
        for name, rows in ((f"default pgvector (ef={DEFAULT_EF})", default), (f"adaptive ({plan})", adaptive)):
            r = recall(rows, truth)
            results.add_row(Text(name, style="bold"), bar(r, color=recall_color(r)), Text(f"recall@{K} = {r:.2f}", style=recall_color(r)), Text(f"{len(rows)}/{K} rows returned", style="grey82"))
        panels.append(Panel(Group(results, Text(""), Text(f"Default fetches the {DEFAULT_EF} nearest vectors, then filters: expected survivors = {DEFAULT_EF}·s = {DEFAULT_EF * s:.1f}", style="grey82")),
                            title="Result vs exact ground truth", title_align="left", border_style=ACCENT))
        states["Execute + ground truth"] = ("done", "compared against brute-force top-k")
        render()

    # ---------- sweep ----------

    def action_sweep(self):
        self.query_one("#center", TabbedContent).active = "sweep"
        self.run_sweep()

    @work(thread=True, exclusive=True, group="sweep")
    def run_sweep(self):
        table = self.query_one("#sweep-table", DataTable)
        progress = self.query_one("#sweep-progress", ProgressBar)
        summary = self.query_one("#sweep-summary", Static)

        def start():
            table.clear()
            table.display = progress.display = True
            progress.update(total=len(SELECTIVITY_POINTS) * QUERIES_PER_POINT, progress=0)
            summary.update(f"Computing live from the seeded dataset · {QUERIES_PER_POINT} queries per point…")

        self.call_from_thread(start)
        rng = np.random.default_rng(SEED + 1)
        vectors = self.data.vectors
        for percent in SELECTIVITY_POINTS:
            s = percent / 100
            ef = ef_search(K, s)
            plan = "exact" if exact_cost(N, s) <= ann_estimated_cost(K, s) else "ann"
            totals = {"default": 0.0, "exact": 0.0, "adaptive": 0.0}
            for _ in range(QUERIES_PER_POINT):
                query = rng.standard_normal(DIM)
                mask = np.zeros(N, dtype=bool)
                mask[rng.choice(N, size=max(1, round(N * s)), replace=False)] = True
                truth = ground_truth(query, vectors, mask, K)
                exact_rows = exact_filter_first(query, vectors, mask, K)
                totals["default"] += recall(ann_post_filter(query, vectors, mask, K, DEFAULT_EF), truth)
                totals["exact"] += recall(exact_rows, truth)
                totals["adaptive"] += recall(exact_rows if plan == "exact" else ann_post_filter(query, vectors, mask, K, ef), truth)
                self.call_from_thread(progress.advance, 1)
            r = {name: total / QUERIES_PER_POINT for name, total in totals.items()}
            row = (
                f"{percent:g}%",
                Text(f"{r['default']:.3f}", style=recall_color(r["default"])),
                Text(f"{r['exact']:.3f}", style=recall_color(r["exact"])),
                f"{max(1, round(N * s)):,}",
                Text(f"{r['adaptive']:.3f}", style=recall_color(r["adaptive"])),
                "exact" if plan == "exact" else f"ann ef={ef}",
            )
            self.call_from_thread(table.add_row, *row)
        self.call_from_thread(summary.update, f"Done · every number above was computed on this run (seed {SEED}). Default recall ≈ min(1, ef·s/k).")

    # ---------- tests ----------

    def action_tests(self):
        self.query_one("#center", TabbedContent).active = "tests"
        self.run_tests()

    @work(thread=True, exclusive=True, group="tests")
    def run_tests(self):
        table = self.query_one("#tests-table", DataTable)
        progress = self.query_one("#tests-progress", ProgressBar)
        summary = self.query_one("#tests-summary", Static)
        collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings"], capture_output=True, text=True).stdout
        total = sum(1 for line in collected.splitlines() if "::" in line)

        def start():
            table.clear()
            table.display = progress.display = True
            progress.update(total=total or None, progress=0)
            summary.update(f"Running {total} tests…")

        self.call_from_thread(start)
        proc = subprocess.Popen([sys.executable, "-m", "pytest", "-v", "-p", "no:warnings", "--no-header", "--color=no"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        passed = failed = 0
        for line in proc.stdout:
            match = re.match(r"tests/(\w+)\.py::(\S+) (PASSED|FAILED|ERROR)", line)
            if not match:
                continue
            module, test, status = match.groups()
            ok = status == "PASSED"
            passed, failed = passed + ok, failed + (not ok)
            self.call_from_thread(table.add_row, Text("✔" if ok else "✘", style=GOOD if ok else BAD), module.removeprefix("test_"), TEST_EXPLAIN.get(module, ""), test.removeprefix("test_").replace("_", " "))
            self.call_from_thread(progress.advance, 1)
        proc.wait()
        color = GOOD if failed == 0 else BAD
        self.call_from_thread(summary.update, Text(f"{passed} passed, {failed} failed", style=f"bold {color}"))


if __name__ == "__main__":
    StratSelectApp().run()
