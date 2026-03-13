"""
Phase 4 analysis: message-size transition boundary sweep.

Reads phase4_sweep_summary.json and per-size iteration_times / bandit logs
from results/.

Produces:
  - results/phase4_landscape.png           (iter time vs msg size, all configs)
  - results/phase4_auto_gap.png            (% gap: AUTO vs best fixed config)
  - results/phase4_bandit_picks.png        (UCB1 preferred arm at each size)
  - results/phase4_transitions.png         (zoomed view of transition regions)
  - Prints summary table to stdout
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

CONFIG_COLORS = {
    "auto":        "#1f77b4",
    "tree_simple": "#2ca02c",
    "tree_ll128":  "#ff7f0e",
    "ring_simple": "#9467bd",
    "ring_ll128":  "#8c564b",
}

CONFIG_LABELS = {
    "auto":        "AUTO",
    "tree_simple": "Tree+Simple",
    "tree_ll128":  "Tree+LL128",
    "ring_simple": "Ring+Simple",
    "ring_ll128":  "Ring+LL128",
}

SIZE_ORDER = ["32KB", "64KB", "256KB", "1MB", "2MB", "4MB", "16MB", "64MB", "256MB"]
SIZE_BYTES = {
    "32KB": 32768, "64KB": 65536, "256KB": 262144,
    "1MB": 1048576, "2MB": 2097152, "4MB": 4194304,
    "16MB": 16777216, "64MB": 67108864, "256MB": 268435456,
}


def load_summary():
    path = RESULTS_DIR / "phase4_sweep_summary.json"
    if not path.exists():
        print(f"No sweep summary found at {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def load_times(tag):
    path = RESULTS_DIR / f"iteration_times_{tag}.txt"
    if not path.exists():
        return []
    return [float(l.strip()) for l in path.read_text().strip().split("\n")
            if l.strip()]


# ---------------------------------------------------------------------------
# 1. Landscape: iteration time vs message size for all configs
# ---------------------------------------------------------------------------
def plot_landscape(summary):
    sizes = [s for s in SIZE_ORDER if s in summary]
    if not sizes:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    x_bytes = [SIZE_BYTES[s] for s in sizes]

    for cfg in ["auto", "tree_simple", "tree_ll128", "ring_simple", "ring_ll128"]:
        means = [summary[s].get(cfg, 0) for s in sizes]
        ax.plot(x_bytes, means, marker="o", linewidth=2,
                color=CONFIG_COLORS[cfg], label=CONFIG_LABELS[cfg])

    # UCB1 converged
    ucb_means = [summary[s].get("ucb1_converged", 0) for s in sizes]
    ax.plot(x_bytes, ucb_means, marker="s", linewidth=2, linestyle="--",
            color="#d62728", label="UCB1 Bandit (converged)")

    ax.set_xscale("log")
    ax.set_xlabel("AllReduce Message Size (bytes)")
    ax.set_ylabel("Mean Iteration Time (ms)")
    ax.set_title("Phase 4: Tuning Landscape Across Message Sizes (8x A100)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # Label x-axis with human sizes
    ax.set_xticks(x_bytes)
    ax.set_xticklabels(sizes, rotation=30, ha="right")

    # Mark known transition zones
    for boundary, label in [(65536, "LL->LL128"), (1048576, "LL128->Simple"),
                             (8388608, "tree->ring")]:
        ax.axvline(boundary, color="gray", linestyle=":", alpha=0.5)
        ax.text(boundary, ax.get_ylim()[1] * 0.98, label,
                fontsize=7, ha="center", va="top", color="gray")

    fig.tight_layout()
    out = RESULTS_DIR / "phase4_landscape.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 2. AUTO gap: % difference between AUTO and best fixed config
# ---------------------------------------------------------------------------
def plot_auto_gap(summary):
    sizes = [s for s in SIZE_ORDER if s in summary]
    if not sizes:
        return

    gaps = []
    best_labels = []
    for s in sizes:
        data = summary[s]
        auto = data.get("auto", 0)
        configs = {c: data.get(c, 999) for c in CONFIG_COLORS if c != "auto"}
        best_label = min(configs, key=configs.get)
        best_val = configs[best_label]
        gap = (auto - best_val) / auto * 100 if auto > 0 else 0
        gaps.append(gap)
        best_labels.append(best_label)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sizes))
    colors = [CONFIG_COLORS[bl] for bl in best_labels]
    bars = ax.bar(x, gaps, color=colors, alpha=0.85, edgecolor="black",
                   linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=30, ha="right")
    ax.set_ylabel("AUTO Overhead vs Best Config (%)")
    ax.set_title("Phase 4: Where AUTO Is Suboptimal (8x A100)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    # Label bars with best config name
    for bar, bl, gap in zip(bars, best_labels, gaps):
        if gap > 0.2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{CONFIG_LABELS[bl]}\n{gap:.1f}%",
                    ha="center", va="bottom", fontsize=7)

    # Legend for colors
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CONFIG_COLORS[c], alpha=0.85)
               for c in ["tree_simple", "tree_ll128", "ring_simple", "ring_ll128"]]
    labels = [CONFIG_LABELS[c]
              for c in ["tree_simple", "tree_ll128", "ring_simple", "ring_ll128"]]
    ax.legend(handles, labels, title="Best config", fontsize=8, loc="upper left")

    fig.tight_layout()
    out = RESULTS_DIR / "phase4_auto_gap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 3. UCB1 bandit preferred arm at each size
# ---------------------------------------------------------------------------
def plot_bandit_picks(summary):
    sizes = [s for s in SIZE_ORDER if s in summary]
    if not sizes:
        return

    arm_names = list(CONFIG_COLORS.keys())
    arm_idx_map = {name: i for i, name in enumerate(arm_names)}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [2, 1]},
                                    sharex=True)

    x = np.arange(len(sizes))

    # Top: per-size UCB1 converged time vs AUTO
    auto_means = [summary[s].get("auto", 0) for s in sizes]
    ucb_means = [summary[s].get("ucb1_converged", 0) for s in sizes]
    ax1.bar(x - 0.2, auto_means, 0.35, label="AUTO", color="#1f77b4",
            alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.bar(x + 0.2, ucb_means, 0.35, label="UCB1 (converged)", color="#d62728",
            alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Mean Iteration Time (ms)")
    ax1.set_title("UCB1 Bandit vs AUTO at Each Message Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Bottom: which arm the bandit preferred
    pref_arms = [summary[s].get("ucb1_preferred_arm", "auto") for s in sizes]
    pref_indices = [arm_idx_map.get(a, 0) for a in pref_arms]
    pref_colors = [CONFIG_COLORS.get(a, "#333") for a in pref_arms]
    ax2.scatter(x, pref_indices, c=pref_colors, s=120, edgecolors="black",
                linewidths=0.5, zorder=3)
    ax2.set_yticks(range(len(arm_names)))
    ax2.set_yticklabels([CONFIG_LABELS[a] for a in arm_names], fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes, rotation=30, ha="right")
    ax2.set_xlabel("AllReduce Message Size")
    ax2.set_ylabel("Preferred Arm")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "phase4_bandit_picks.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 4. Box plots at transition sizes
# ---------------------------------------------------------------------------
def plot_transitions(summary):
    transition_sizes = ["64KB", "1MB", "2MB", "16MB"]
    transition_sizes = [s for s in transition_sizes if s in summary]
    if not transition_sizes:
        return

    n_sizes = len(transition_sizes)
    fig, axes = plt.subplots(1, n_sizes, figsize=(4 * n_sizes, 6), sharey=False)
    if n_sizes == 1:
        axes = [axes]

    for ax, size_label in zip(axes, transition_sizes):
        box_data = []
        box_labels = []
        box_colors = []
        for cfg in ["auto", "tree_simple", "tree_ll128",
                     "ring_simple", "ring_ll128"]:
            times = load_times(f"{size_label}_{cfg}")
            if times:
                box_data.append(times)
                box_labels.append(CONFIG_LABELS[cfg])
                box_colors.append(CONFIG_COLORS[cfg])

        if not box_data:
            ax.set_title(f"{size_label} — no data")
            continue

        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(f"{size_label}", fontsize=11)
        ax.set_ylabel("Iteration Time (ms)")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Transition Boundaries: Distribution Comparison (8x A100)",
                 fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / "phase4_transitions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(summary):
    sizes = [s for s in SIZE_ORDER if s in summary]
    if not sizes:
        print("No data.", file=sys.stderr)
        return

    header = (f"{'Size':<8s} {'AUTO':>9s} {'Tree+S':>9s} {'Tree+L':>9s} "
              f"{'Ring+S':>9s} {'Ring+L':>9s} {'UCB1':>9s} "
              f"{'Best':>12s} {'Gap':>7s}")
    print(f"\n{header}")
    print("-" * 90)

    for s in sizes:
        d = summary[s]
        auto = d.get("auto", 0)
        ts = d.get("tree_simple", 0)
        tl = d.get("tree_ll128", 0)
        rs = d.get("ring_simple", 0)
        rl = d.get("ring_ll128", 0)
        ucb = d.get("ucb1_converged", 0)

        configs = {"auto": auto, "tree_simple": ts, "tree_ll128": tl,
                   "ring_simple": rs, "ring_ll128": rl}
        best_label = min(configs, key=configs.get)
        best_val = configs[best_label]
        gap = (auto - best_val) / auto * 100 if auto > 0 else 0

        print(f"{s:<8s} {auto:>9.3f} {ts:>9.3f} {tl:>9.3f} "
              f"{rs:>9.3f} {rl:>9.3f} {ucb:>9.3f} "
              f"{CONFIG_LABELS[best_label]:>12s} {gap:>+6.1f}%")

    # Show where AUTO is worst
    print(f"\n--- Largest AUTO overhead ---")
    gaps = []
    for s in sizes:
        d = summary[s]
        auto = d.get("auto", 0)
        configs = {c: d.get(c, 999) for c in CONFIG_COLORS if c != "auto"}
        best_val = min(configs.values())
        gap = (auto - best_val) / auto * 100 if auto > 0 else 0
        gaps.append((s, gap))
    for s, gap in sorted(gaps, key=lambda x: -x[1])[:3]:
        print(f"  {s}: AUTO is {gap:.1f}% slower than best config")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    summary = load_summary()
    print_summary(summary)
    plot_landscape(summary)
    plot_auto_gap(summary)
    plot_bandit_picks(summary)
    plot_transitions(summary)


if __name__ == "__main__":
    main()
