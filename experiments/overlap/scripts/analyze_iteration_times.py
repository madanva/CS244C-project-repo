"""
Analyze Phase 3 iteration-time results across NCCL configs.

Reads:
  results/iteration_times_auto.txt
  results/iteration_times_simple.txt
  results/iteration_times_ll128.txt

Each file should contain one iteration time in milliseconds per line.
Prints summary stats per config so you can see which NCCL setting
minimizes end-to-end iteration time (not just bandwidth).
"""

from __future__ import annotations

import math
from pathlib import Path


RESULTS_DIR = Path(__file__).parent / "results"
CONFIGS = ["auto", "simple", "ll128"]


def load_times(path: Path) -> list[float]:
    if not path.is_file():
        return []
    vals: list[float] = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(float(line))
        except ValueError:
            continue
    return vals


def summarize(times: list[float]) -> dict[str, float]:
    if not times:
        return {}
    times_sorted = sorted(times)
    n = len(times_sorted)

    def pct(p: float) -> float:
        if n == 0:
            return math.nan
        idx = int(max(0, min(n - 1, round(p * (n - 1)))))
        return times_sorted[idx]

    mean = sum(times_sorted) / n
    return {
        "n": n,
        "mean": mean,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "min": times_sorted[0],
        "max": times_sorted[-1],
    }


def main() -> None:
    print(f"Reading iteration times from {RESULTS_DIR}")
    summaries: dict[str, dict[str, float]] = {}

    for cfg in CONFIGS:
        path = RESULTS_DIR / f"iteration_times_{cfg}.txt"
        times = load_times(path)
        if not times:
            print(f"- {cfg}: no data at {path}")
            continue
        summaries[cfg] = summarize(times)

    if not summaries:
        print("No iteration-time files found; run run_modal.py first.")
        return

    print("\nSummary (ms per iteration):")
    header = f"{'config':<8} {'n':>4} {'mean':>8} {'p50':>8} {'p90':>8} {'p95':>8} {'min':>8} {'max':>8}"
    print(header)
    print("-" * len(header))
    for cfg in CONFIGS:
        s = summaries.get(cfg)
        if not s:
            continue
        print(
            f"{cfg:<8} "
            f"{int(s['n']):>4} "
            f"{s['mean']:>8.3f} "
            f"{s['p50']:>8.3f} "
            f"{s['p90']:>8.3f} "
            f"{s['p95']:>8.3f} "
            f"{s['min']:>8.3f} "
            f"{s['max']:>8.3f}"
        )

    print(
        "\nInterpretation:\n"
        "- Lower mean/p95 = better end-to-end iteration time.\n"
        "- Compare AUTO vs Simple vs LL128: the config with highest communication\n"
        "  bandwidth from Phase 1 may not minimize iteration time here.\n"
        "- This gap is what we want to surface for the workload-aware tuner\n"
        "  (e.g., maybe Simple or LL128 gives slightly worse bandwidth but\n"
        "  better iteration latency under your compute pattern)."
    )


if __name__ == "__main__":
    main()

