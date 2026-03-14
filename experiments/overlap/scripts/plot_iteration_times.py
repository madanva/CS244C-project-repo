"""
Plot Phase 3 iteration-time results across NCCL configs.

Reads:
  results/iteration_times_auto.txt
  results/iteration_times_simple.txt
  results/iteration_times_ll128.txt

Produces (in results/):
  - iteration_times_bar_mean.png      # mean + p95 as error bars
  - iteration_times_boxplot.png       # box plots per config
  - iteration_times_cdf.png           # empirical CDF per config
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).parent / "results"
CONFIGS = ["auto", "simple", "ll128"]
LABELS = {"auto": "AUTO", "simple": "Simple", "ll128": "LL128"}
COLORS = {"auto": "tab:blue", "simple": "tab:orange", "ll128": "tab:green"}


def load_times(path: Path) -> list[float]:
    vals: list[float] = []
    if not path.is_file():
        return vals
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(float(line))
        except ValueError:
            continue
    return vals


def compute_stats(times: list[float]) -> tuple[float, float]:
    if not times:
        return float("nan"), float("nan")
    times_sorted = sorted(times)
    n = len(times_sorted)
    mean = sum(times_sorted) / n
    p95 = times_sorted[int(max(0, min(n - 1, round(0.95 * (n - 1)))))]
    return mean, p95


def plot_bar_mean(stats: dict[str, tuple[float, float]]) -> None:
    cfgs = [c for c in CONFIGS if c in stats]
    means = [stats[c][0] for c in cfgs]
    p95s = [stats[c][1] for c in cfgs]
    errs = [p95s[i] - means[i] for i in range(len(cfgs))]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        [LABELS[c] for c in cfgs],
        means,
        yerr=errs,
        color=[COLORS[c] for c in cfgs],
        capsize=4,
    )
    ax.set_ylabel("Iteration time (ms)")
    ax.set_title("Mean iteration time (error bar = p95)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "iteration_times_bar_mean.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_box(times_by_cfg: dict[str, list[float]]) -> None:
    cfgs = [c for c in CONFIGS if c in times_by_cfg]
    data = [times_by_cfg[c] for c in cfgs]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(
        data,
        labels=[LABELS[c] for c in cfgs],
        patch_artist=True,
        showfliers=True,
    )
    for patch, c in zip(bp["boxes"], [COLORS[cfg] for cfg in cfgs]):
        patch.set_facecolor(c)
    ax.set_ylabel("Iteration time (ms)")
    ax.set_title("Iteration time distribution by NCCL config")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "iteration_times_boxplot.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_cdf(times_by_cfg: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for cfg in CONFIGS:
        times = times_by_cfg.get(cfg)
        if not times:
            continue
        xs = sorted(times)
        n = len(xs)
        ys = [i / (n - 1) if n > 1 else 1.0 for i in range(n)]
        ax.plot(xs, ys, label=LABELS[cfg], color=COLORS[cfg])
    ax.set_xlabel("Iteration time (ms)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("CDF of iteration times")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "iteration_times_cdf.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    times_by_cfg: dict[str, list[float]] = {}
    stats: dict[str, tuple[float, float]] = {}

    for cfg in CONFIGS:
        path = RESULTS_DIR / f"iteration_times_{cfg}.txt"
        times = load_times(path)
        if not times:
            print(f"{cfg}: no data at {path}")
            continue
        times_by_cfg[cfg] = times
        stats[cfg] = compute_stats(times)

    if not times_by_cfg:
        print("No iteration time data found; run run_modal.py first.")
        return

    plot_bar_mean(stats)
    plot_box(times_by_cfg)
    plot_cdf(times_by_cfg)
    print("Saved plots to results/:")
    print("  - iteration_times_bar_mean.png")
    print("  - iteration_times_boxplot.png")
    print("  - iteration_times_cdf.png")


if __name__ == "__main__":
    main()

