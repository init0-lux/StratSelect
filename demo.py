"""Interactive walkthrough: type a filter, watch the planner pick a strategy.

    python3 demo.py          # interactive
    python3 demo.py --tour   # scripted tour, no input needed
"""

import sys
import time

import numpy as np

from src.ann_search import ann_post_filter
from src.cost_model import CALIBRATION_MULTIPLIER, ann_estimated_cost, ef_search, exact_cost
from src.config import DIM, K, N, SEED
from src.dataset import generate_dataset
from src.exact_search import exact_filter_first
from src.ground_truth import ground_truth

DEFAULT_EF = 40  # pgvector's default hnsw.ef_search
CATEGORIES = ("shoes", "books", "home", "tech")
TOUR = [("shoes", 300), ("tech", 2100), ("any", 2100), ("any", 10_001)]

BOLD, DIM_, GREEN, RED, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def step(n: int, title: str) -> None:
    print(f"\n{BOLD}[{n}] {title}{RESET}")
    time.sleep(0.3)


def note(text: str) -> None:
    print(f"    {DIM_}{text}{RESET}")


def colour_recall(r: float) -> str:
    colour = GREEN if r >= 0.99 else YELLOW if r >= 0.5 else RED
    return f"{colour}{r:.2f}{RESET}"


def recall(result: list[int], truth: list[int]) -> float:
    return len(set(result) & set(truth)) / len(truth) if truth else 1.0


def run_query(data, category: str, max_price: int, rng) -> None:
    vectors = data.vectors
    mask = data.prices < max_price
    if category != "any":
        mask &= data.categories == category
    matching = int(mask.sum())
    where = f"price < {max_price}" + ("" if category == "any" else f" AND category = '{category}'")
    print(f"\n{BOLD}Query:{RESET} SELECT ... WHERE {where} ORDER BY embedding <-> $q LIMIT {K}")

    step(1, "Estimate selectivity")
    if matching == 0:
        print("    0 rows match the filter, so there is nothing to search.")
        return
    s = matching / N
    print(f"    {matching:,} of {N:,} rows pass the filter  ->  selectivity = {s:.2%}")

    step(2, "Cost model")
    exact = exact_cost(N, s)
    ef = ef_search(K, s)
    ann = ann_estimated_cost(K, s)
    print(f"    exact filter-first : scan the matching rows     = {exact:>10,.0f} vectors")
    print(f"    ANN + post-filter  : ef_search = 3k/s = {ef:<6,}  x{CALIBRATION_MULTIPLIER} = {ann:>10,.0f} units")
    note(f"ef = 3k/s means about {3 * K} candidates survive the filter on average: enough for top-{K}.")

    step(3, "Plan decision")
    plan = "exact" if exact <= ann else "ann"
    if plan == "exact":
        print(f"    {GREEN}EXACT{RESET}: exact is cheaper ({exact:,.0f} <= {ann:,.0f}), so it brute-forces the matching rows.")
    else:
        print(f"    {GREEN}ANN, ef_search={ef}{RESET}: ANN is cheaper ({ann:,.0f} < {exact:,.0f}), with ef widened for the filter.")

    step(4, "Execute and compare with ground truth")
    query = rng.standard_normal(DIM)
    truth = ground_truth(query, vectors, mask, K)
    default = ann_post_filter(query, vectors, mask, K, DEFAULT_EF)
    if plan == "exact":
        adaptive = exact_filter_first(query, vectors, mask, K)
    else:
        adaptive = ann_post_filter(query, vectors, mask, K, ef)
    print(f"    default pgvector (ef={DEFAULT_EF}) : {len(default):>2}/{K} results returned   recall@{K} = {colour_recall(recall(default, truth))}")
    note(f"it fetched the {DEFAULT_EF} nearest vectors and kept the ones that passed the filter; expected survivors = {DEFAULT_EF * s:.1f}")
    print(f"    adaptive ({plan:<5})          : {len(adaptive):>2}/{K} results returned   recall@{K} = {colour_recall(recall(adaptive, truth))}")


def ask() -> tuple[str, int] | None:
    raw = input(f"\n{BOLD}category{RESET} {CATEGORIES + ('any',)} [q to quit]: ").strip().lower()
    if raw in ("q", "quit", "exit"):
        return None
    if raw not in CATEGORIES + ("any",):
        print("    unknown category; using 'any'")
        raw = "any"
    price = input(f"{BOLD}max price{RESET} (100-10001, e.g. 300 = strict, 10001 = no price filter): ").strip()
    try:
        return raw, max(101, min(10_001, int(price)))
    except ValueError:
        print("    not a number; using 10001")
        return raw, 10_001


def main() -> None:
    print(f"{BOLD}StratSelect: adaptive strategy for filtered vector search{RESET}")
    note(f"Generating {N:,} products: {DIM}-dim embeddings, a category from {len(CATEGORIES)}, price 100-10000 (seed {SEED}) ...")
    data = generate_dataset(N, DIM, SEED)
    rng = np.random.default_rng(SEED + 7)
    if "--tour" in sys.argv:
        for category, price in TOUR:
            run_query(data, category, price, rng)
            print("\n" + "-" * 72)
        return
    try:
        while (choice := ask()) is not None:
            run_query(data, *choice, rng)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
