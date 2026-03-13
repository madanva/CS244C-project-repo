"""
Analysis & plotting for the CTA/channel count sweep experiment.

Generates paper-quality figures:
  1. cta_overlap_curves.png     — Iteration time vs CTA count (overlap) per size
  2. cta_seq_vs_overlap.png     — Optimal CTA under sequential vs overlap
  3. cta_pareto.png             — Pareto frontier: comm bandwidth vs compute throughput
  4. cta_heatmap.png            — Best config across (CTA count × message size)
  5. cta_compute_interaction.png — CTA × compute intensity at 4MB
  6. cta_auto_gap.png           — AUTO gap amplified by CTA constraint
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


RESULTS_DIR = Path(__file__).parent / "results" / "channel_experiment"
RESULTS_FILE = RESULTS_DIR / "channel_experiment_results.json"

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

CTA_ORDER = ["1", "2", "4", "8", "12", "16", "24", "32"]


def load_results():
    if not RESULTS_FILE.exists():
        print(f"Error: {RESULTS_FILE} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(RESULTS_FILE.read_text())


def plot_cta_overlap_curves(results):
    """Figure 1: Iteration time vs CTA count under overlap, per message size."""
    sweep = results["overlap_cta_sweep"]
    sizes = list(sweep.keys())

    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 5), sharey=False)
    if len(sizes) == 1:
        axes = [axes]

    for ax, size in zip(axes, sizes):
        ctas_present = [c for c in CTA_ORDER if c in sweep[size]]
        x = list(range(len(ctas_present)))

        for cfg in CONFIGS:
            vals = [sweep[size][c].get(cfg, 0) for c in ctas_present]
            ax.plot(x, vals, "o-", label=CONFIG_SHORT[cfg],
                    color=CONFIG_COLORS[cfg], linewidth=2, markersize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(ctas_present)
        ax.set_xlabel("NCCL_MAX_CTAS (SM count)", fontsize=11)
        ax.set_ylabel("Iteration Time (ms)", fontsize=11)
        ax.set_title(f"AllReduce = {size}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Effect of CTA Count on Iteration Time Under Overlap",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "cta_overlap_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_cta_seq_vs_overlap(results):
    """Figure 2: Optimal CTA count — sequential vs overlap."""
    ovl = results["overlap_cta_sweep"]
    seq = results["sequential_cta_sweep"]
    sizes = list(ovl.keys())

    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 5), sharey=False)
    if len(sizes) == 1:
        axes = [axes]

    for ax, size in zip(axes, sizes):
        ctas_present = [c for c in CTA_ORDER if c in ovl.get(size, {})]
        x = list(range(len(ctas_present)))

        # Best config time at each CTA, overlap vs sequential
        ovl_best = [min(ovl[size][c].values()) for c in ctas_present]
        seq_best = [min(seq.get(size, {}).get(c, {"x": 999}).values())
                    for c in ctas_present]

        ax.plot(x, seq_best, "s--", label="Sequential (best config)",
                color="#3498db", linewidth=2, markersize=8)
        ax.plot(x, ovl_best, "o-", label="Overlap (best config)",
                color="#e74c3c", linewidth=2, markersize=8)

        # Mark optimal point for each
        ovl_opt_idx = ovl_best.index(min(ovl_best))
        seq_opt_idx = seq_best.index(min(seq_best))
        ax.annotate(f"OVL opt: {ctas_present[ovl_opt_idx]} CTAs",
                    xy=(ovl_opt_idx, ovl_best[ovl_opt_idx]),
                    xytext=(ovl_opt_idx + 0.5, ovl_best[ovl_opt_idx] - 0.1),
                    fontsize=9, color="#e74c3c", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c"))
        ax.annotate(f"SEQ opt: {ctas_present[seq_opt_idx]} CTAs",
                    xy=(seq_opt_idx, seq_best[seq_opt_idx]),
                    xytext=(seq_opt_idx + 0.5, seq_best[seq_opt_idx] + 0.1),
                    fontsize=9, color="#3498db", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#3498db"))

        ax.set_xticks(x)
        ax.set_xticklabels(ctas_present)
        ax.set_xlabel("NCCL_MAX_CTAS", fontsize=11)
        ax.set_ylabel("Best Iteration Time (ms)", fontsize=11)
        ax.set_title(f"AllReduce = {size}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Optimal CTA Count: Sequential vs Overlap",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "cta_seq_vs_overlap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_cta_heatmap(results):
    """Figure 4: Heatmap of best config across (CTA count × message size)."""
    sweep = results["overlap_cta_sweep"]
    sizes = list(sweep.keys())
    ctas_present = [c for c in CTA_ORDER if any(c in sweep[s] for s in sizes)]

    cfg_to_idx = {c: i for i, c in enumerate(CONFIGS)}
    cmap = mcolors.ListedColormap([CONFIG_COLORS[c] for c in CONFIGS])

    data = np.zeros((len(ctas_present), len(sizes)), dtype=int)
    gap_data = np.zeros((len(ctas_present), len(sizes)))

    for col, size in enumerate(sizes):
        for row, cta in enumerate(ctas_present):
            cta_data = sweep[size].get(cta, {})
            if cta_data:
                best = min(cta_data, key=cta_data.get)
                auto_t = cta_data.get("auto", 0)
                best_t = cta_data[best]
                data[row, col] = cfg_to_idx[best]
                gap_data[row, col] = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=len(CONFIGS) - 1)
    for row in range(len(ctas_present)):
        for col in range(len(sizes)):
            cfg_name = CONFIGS[data[row, col]]
            ax1.text(col, row, CONFIG_SHORT[cfg_name], ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5))

    ax1.set_xticks(range(len(sizes)))
    ax1.set_xticklabels(sizes)
    ax1.set_yticks(range(len(ctas_present)))
    ax1.set_yticklabels([f"{c} CTAs" for c in ctas_present])
    ax1.set_xlabel("Message Size", fontsize=12)
    ax1.set_ylabel("NCCL_MAX_CTAS", fontsize=12)
    ax1.set_title("Best Config (CTAs × Size, Overlap)", fontsize=13, fontweight="bold")

    im2 = ax2.imshow(gap_data, cmap="YlOrRd", aspect="auto", vmin=0)
    for row in range(len(ctas_present)):
        for col in range(len(sizes)):
            ax2.text(col, row, f"{gap_data[row, col]:.1f}%", ha="center",
                     va="center", fontsize=9, fontweight="bold")

    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels(sizes)
    ax2.set_yticks(range(len(ctas_present)))
    ax2.set_yticklabels([f"{c} CTAs" for c in ctas_present])
    ax2.set_xlabel("Message Size", fontsize=12)
    ax2.set_ylabel("NCCL_MAX_CTAS", fontsize=12)
    ax2.set_title("AUTO Gap % (CTAs × Size, Overlap)", fontsize=13, fontweight="bold")
    plt.colorbar(im2, ax=ax2, label="AUTO overhead (%)")

    plt.tight_layout()
    out = RESULTS_DIR / "cta_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_cta_compute_interaction(results):
    """Figure 5: CTA × compute intensity at 4MB under overlap."""
    interaction = results.get("cta_compute_interaction", {})
    if not interaction:
        print("  Skipping CTA×compute interaction (no data)")
        return

    cmul_labels = list(interaction.keys())

    fig, axes = plt.subplots(1, len(cmul_labels), figsize=(6 * len(cmul_labels), 5),
                             sharey=False)
    if len(cmul_labels) == 1:
        axes = [axes]

    for ax, cmul_label in zip(axes, cmul_labels):
        ctas_present = [c for c in CTA_ORDER if c in interaction[cmul_label]]
        x = list(range(len(ctas_present)))

        for cfg in CONFIGS:
            vals = [interaction[cmul_label][c].get(cfg, 0) for c in ctas_present]
            ax.plot(x, vals, "o-", label=CONFIG_SHORT[cfg],
                    color=CONFIG_COLORS[cfg], linewidth=2, markersize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(ctas_present)
        ax.set_xlabel("NCCL_MAX_CTAS", fontsize=11)
        ax.set_ylabel("Iteration Time (ms)", fontsize=11)
        ax.set_title(f"Compute: {cmul_label}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("CTA Count × Compute Intensity at 4MB (Overlap)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "cta_compute_interaction.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_cta_auto_gap(results):
    """Figure 6: AUTO gap across CTA counts — does constraining CTAs amplify gap?"""
    sweep = results["overlap_cta_sweep"]
    sizes = list(sweep.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = []
    bar_data = {s: [] for s in sizes}

    ctas_present = [c for c in CTA_ORDER if any(c in sweep[s] for s in sizes)]

    width = 0.18
    x = np.arange(len(ctas_present))

    for i, size in enumerate(sizes):
        gaps = []
        for cta in ctas_present:
            cta_data = sweep[size].get(cta, {})
            auto_t = cta_data.get("auto", 0)
            best_t = min(cta_data.values()) if cta_data else 0
            gap = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0
            gaps.append(gap)
        ax.bar(x + i * width, gaps, width, label=size,
               alpha=0.85)

    ax.set_xticks(x + width * (len(sizes) - 1) / 2)
    ax.set_xticklabels([f"{c} CTAs" for c in ctas_present])
    ax.set_xlabel("NCCL_MAX_CTAS", fontsize=12)
    ax.set_ylabel("AUTO Gap vs Best Config (%)", fontsize=12)
    ax.set_title("Does CTA Constraint Amplify AUTO Suboptimality? (Overlap)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, title="Message Size")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "cta_auto_gap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def main():
    print("Loading results...")
    results = load_results()

    print("Generating plots...")
    plot_cta_overlap_curves(results)
    plot_cta_seq_vs_overlap(results)
    plot_cta_heatmap(results)
    plot_cta_compute_interaction(results)
    plot_cta_auto_gap(results)
    print("\nDone!")


if __name__ == "__main__":
    main()
