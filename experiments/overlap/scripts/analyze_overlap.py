"""
Analysis & plotting for the overlap-aware NCCL tuning experiment.

Generates paper-quality figures:
  1. seq_vs_overlap.png     — Side-by-side: sequential vs overlap config rankings
  2. winner_flip.png        — Heatmap: which config wins at each size, seq vs overlap
  3. auto_gap_comparison.png — AUTO gap (%) in sequential vs overlap
  4. compute_heatmap.png    — Heatmap: best config across (msg_size × compute_intensity)
  5. compute_sweep_lines.png — Line plots: iteration time vs compute intensity per config
  6. sm_contention.png      — Shows how protocol ranking changes with compute load
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


RESULTS_DIR = Path(__file__).parent / "results" / "overlap_experiment"
RESULTS_FILE = RESULTS_DIR / "overlap_experiment_results.json"

CONFIGS = ["auto", "tree_simple", "tree_ll128", "ring_simple", "ring_ll128"]
CONFIG_COLORS = {
    "auto": "#333333",
    "tree_simple": "#e74c3c",
    "tree_ll128": "#3498db",
    "ring_simple": "#2ecc71",
    "ring_ll128": "#f39c12",
}
CONFIG_SHORT = {
    "auto": "AUTO",
    "tree_simple": "Tree+Simple",
    "tree_ll128": "Tree+LL128",
    "ring_simple": "Ring+Simple",
    "ring_ll128": "Ring+LL128",
}

SIZE_ORDER = ["32KB", "64KB", "256KB", "1MB", "2MB", "4MB", "16MB", "64MB", "256MB"]
SIZE_BYTES = {
    "32KB": 32768, "64KB": 65536, "256KB": 262144, "1MB": 1048576,
    "2MB": 2097152, "4MB": 4194304, "16MB": 16777216, "64MB": 67108864,
    "256MB": 268435456,
}


def load_results():
    if not RESULTS_FILE.exists():
        print(f"Error: {RESULTS_FILE} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(RESULTS_FILE.read_text())


def plot_seq_vs_overlap(results):
    """Figure 1: Side-by-side bar charts comparing sequential vs overlap."""
    seq = results["sequential"]
    ovl = results["overlap"]

    sizes = [s for s in SIZE_ORDER if s in seq and s in ovl]
    x = np.arange(len(sizes))
    width = 0.08

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for i, cfg in enumerate(CONFIGS):
        vals_seq = [seq[s].get(cfg, 0) for s in sizes]
        vals_ovl = [ovl[s].get(cfg, 0) for s in sizes]
        ax1.bar(x + i * width, vals_seq, width, label=CONFIG_SHORT[cfg],
                color=CONFIG_COLORS[cfg], alpha=0.85)
        ax2.bar(x + i * width, vals_ovl, width, label=CONFIG_SHORT[cfg],
                color=CONFIG_COLORS[cfg], alpha=0.85)

    for ax, title in [(ax1, "Sequential (no overlap)"),
                       (ax2, "Overlap (compute + allreduce concurrent)")]:
        ax.set_xlabel("Message Size", fontsize=12)
        ax.set_ylabel("Iteration Time (ms)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(sizes, rotation=45, ha="right")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "seq_vs_overlap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_winner_flip(results):
    """Figure 2: Which config wins at each size — sequential vs overlap."""
    seq = results["sequential"]
    ovl = results["overlap"]

    sizes = [s for s in SIZE_ORDER if s in seq and s in ovl]

    fig, ax = plt.subplots(figsize=(12, 4))

    cfg_to_idx = {c: i for i, c in enumerate(CONFIGS)}
    cmap = mcolors.ListedColormap([CONFIG_COLORS[c] for c in CONFIGS])

    seq_winners = []
    ovl_winners = []
    for s in sizes:
        s_best = min(seq[s], key=seq[s].get)
        o_best = min(ovl[s], key=ovl[s].get)
        seq_winners.append(cfg_to_idx[s_best])
        ovl_winners.append(cfg_to_idx[o_best])

    data = np.array([seq_winners, ovl_winners])
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=len(CONFIGS) - 1)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Sequential", "Overlap"], fontsize=12)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes, fontsize=11)
    ax.set_xlabel("Message Size", fontsize=12)
    ax.set_title("Best Config: Sequential vs Overlap", fontsize=14, fontweight="bold")

    # Add text labels
    for row in range(2):
        for col in range(len(sizes)):
            cfg_name = CONFIGS[data[row, col]]
            ax.text(col, row, CONFIG_SHORT[cfg_name], ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    # Mark flips
    for col in range(len(sizes)):
        if seq_winners[col] != ovl_winners[col]:
            ax.annotate("FLIP", xy=(col, 1.35), fontsize=9, ha="center",
                        color="red", fontweight="bold")

    plt.tight_layout()
    out = RESULTS_DIR / "winner_flip.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_auto_gap_comparison(results):
    """Figure 3: AUTO gap (%) in sequential vs overlap mode."""
    seq = results["sequential"]
    ovl = results["overlap"]

    sizes = [s for s in SIZE_ORDER if s in seq and s in ovl]

    seq_gaps = []
    ovl_gaps = []
    for s in sizes:
        s_auto = seq[s].get("auto", 0)
        s_best = min(seq[s].values())
        seq_gaps.append((s_auto - s_best) / s_auto * 100 if s_auto > 0 else 0)

        o_auto = ovl[s].get("auto", 0)
        o_best = min(ovl[s].values())
        ovl_gaps.append((o_auto - o_best) / o_auto * 100 if o_auto > 0 else 0)

    x = np.arange(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, seq_gaps, width, label="Sequential",
                   color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width / 2, ovl_gaps, width, label="Overlap",
                   color="#e74c3c", alpha=0.85)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.2:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Message Size", fontsize=12)
    ax.set_ylabel("AUTO Overhead vs Best Config (%)", fontsize=12)
    ax.set_title("NCCL AUTO Suboptimality: Sequential vs Overlap",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = RESULTS_DIR / "auto_gap_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_compute_heatmap(results):
    """Figure 4: Heatmap of best config across (msg_size × compute_intensity)."""
    sweep = results.get("compute_sweep", {})
    if not sweep:
        print("  Skipping compute heatmap (no sweep data)")
        return

    sizes = list(sweep.keys())
    cmul_labels = list(next(iter(sweep.values())).keys())

    cfg_to_idx = {c: i for i, c in enumerate(CONFIGS)}
    cmap = mcolors.ListedColormap([CONFIG_COLORS[c] for c in CONFIGS])

    data = np.zeros((len(cmul_labels), len(sizes)), dtype=int)
    gap_data = np.zeros((len(cmul_labels), len(sizes)))

    for col, size in enumerate(sizes):
        for row, cmul in enumerate(cmul_labels):
            cmul_data = sweep[size].get(cmul, {})
            if cmul_data:
                best = min(cmul_data, key=cmul_data.get)
                auto_t = cmul_data.get("auto", 0)
                best_t = cmul_data[best]
                data[row, col] = cfg_to_idx[best]
                gap_data[row, col] = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: winner heatmap
    im1 = ax1.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=len(CONFIGS) - 1)
    for row in range(len(cmul_labels)):
        for col in range(len(sizes)):
            cfg_name = CONFIGS[data[row, col]]
            ax1.text(col, row, CONFIG_SHORT[cfg_name], ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5))

    ax1.set_xticks(range(len(sizes)))
    ax1.set_xticklabels(sizes)
    ax1.set_yticks(range(len(cmul_labels)))
    ax1.set_yticklabels(cmul_labels)
    ax1.set_xlabel("Message Size", fontsize=12)
    ax1.set_ylabel("Compute Intensity", fontsize=12)
    ax1.set_title("Best Config (msg_size × compute)", fontsize=13, fontweight="bold")

    # Right: AUTO gap heatmap
    im2 = ax2.imshow(gap_data, cmap="YlOrRd", aspect="auto", vmin=0)
    for row in range(len(cmul_labels)):
        for col in range(len(sizes)):
            ax2.text(col, row, f"{gap_data[row, col]:.1f}%", ha="center",
                     va="center", fontsize=9, fontweight="bold")

    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels(sizes)
    ax2.set_yticks(range(len(cmul_labels)))
    ax2.set_yticklabels(cmul_labels)
    ax2.set_xlabel("Message Size", fontsize=12)
    ax2.set_ylabel("Compute Intensity", fontsize=12)
    ax2.set_title("AUTO Gap % (msg_size × compute)", fontsize=13, fontweight="bold")
    plt.colorbar(im2, ax=ax2, label="AUTO overhead (%)")

    plt.tight_layout()
    out = RESULTS_DIR / "compute_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_compute_sweep_lines(results):
    """Figure 5: Line plots showing iteration time vs compute intensity."""
    sweep = results.get("compute_sweep", {})
    if not sweep:
        print("  Skipping compute sweep lines (no data)")
        return

    sizes = list(sweep.keys())
    cmul_labels = list(next(iter(sweep.values())).keys())

    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 5), sharey=False)
    if len(sizes) == 1:
        axes = [axes]

    for ax, size in zip(axes, sizes):
        for cfg in CONFIGS:
            vals = []
            for cmul in cmul_labels:
                cmul_data = sweep[size].get(cmul, {})
                vals.append(cmul_data.get(cfg, 0))
            ax.plot(range(len(cmul_labels)), vals, "o-",
                    label=CONFIG_SHORT[cfg], color=CONFIG_COLORS[cfg],
                    linewidth=2, markersize=6)

        ax.set_xticks(range(len(cmul_labels)))
        ax.set_xticklabels(cmul_labels, rotation=45, ha="right")
        ax.set_xlabel("Compute Intensity", fontsize=11)
        ax.set_ylabel("Iteration Time (ms)", fontsize=11)
        ax.set_title(f"AllReduce = {size}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Protocol Performance Under Increasing Compute Load (Overlap Mode)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "compute_sweep_lines.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_sm_contention(results):
    """Figure 6: Normalized performance — how much each protocol degrades
    relative to its own sequential baseline as compute load increases."""
    sweep = results.get("compute_sweep", {})
    seq = results.get("sequential", {})
    if not sweep:
        print("  Skipping SM contention plot (no data)")
        return

    sizes = list(sweep.keys())
    cmul_labels = list(next(iter(sweep.values())).keys())

    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 5), sharey=True)
    if len(sizes) == 1:
        axes = [axes]

    for ax, size in zip(axes, sizes):
        seq_times = seq.get(size, {})
        for cfg in CONFIGS:
            if cfg == "auto":
                continue
            seq_base = seq_times.get(cfg, 1)
            slowdowns = []
            for cmul in cmul_labels:
                cmul_data = sweep[size].get(cmul, {})
                ovl_t = cmul_data.get(cfg, 0)
                slowdowns.append(ovl_t / seq_base if seq_base > 0 else 1)
            ax.plot(range(len(cmul_labels)), slowdowns, "o-",
                    label=CONFIG_SHORT[cfg], color=CONFIG_COLORS[cfg],
                    linewidth=2, markersize=6)

        ax.set_xticks(range(len(cmul_labels)))
        ax.set_xticklabels(cmul_labels, rotation=45, ha="right")
        ax.set_xlabel("Compute Intensity", fontsize=11)
        ax.set_ylabel("Slowdown vs Sequential Baseline", fontsize=11)
        ax.set_title(f"AllReduce = {size}", fontsize=13, fontweight="bold")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Protocol SM Contention: Slowdown Under Compute Load",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "sm_contention.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def main():
    print("Loading results...")
    results = load_results()

    print("Generating plots...")
    plot_seq_vs_overlap(results)
    plot_winner_flip(results)
    plot_auto_gap_comparison(results)
    plot_compute_heatmap(results)
    plot_compute_sweep_lines(results)
    plot_sm_contention(results)
    print("\nDone!")


if __name__ == "__main__":
    main()
