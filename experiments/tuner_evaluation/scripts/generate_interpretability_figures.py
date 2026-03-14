"""
Generate interpretability figures for the node sweep analysis.

Produces:
  1. Gap heatmap (node count × message size)
  2. AUTO selection comparison (what AUTO picked vs what's best)
  3. Statistical analysis with error bars and confidence intervals
  4. Bandwidth utilization analysis

Usage:
    python3 generate_interpretability_figures.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
NODE_SWEEP_DIR = RESULTS_DIR / "node_sweep"
SC_DIR = RESULTS_DIR / "same_cluster_comparison"
FIGURES_DIR = RESULTS_DIR / "paper_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["1x8", "2x4", "4x2"]
SIZES = ["256KB", "1MB", "4MB", "16MB", "64MB", "256MB"]
SIZE_BYTES = {
    "256KB": 256 * 1024,
    "1MB": 1024 * 1024,
    "4MB": 4 * 1024 * 1024,
    "16MB": 16 * 1024 * 1024,
    "64MB": 64 * 1024 * 1024,
    "256MB": 256 * 1024 * 1024,
}
ALGOS = ["auto", "tree_simple", "tree_ll", "tree_ll128",
         "ring_simple", "ring_ll", "ring_ll128"]
ALGO_LABELS = {
    "auto": "AUTO", "tree_simple": "Tree+S", "tree_ll": "Tree+LL",
    "tree_ll128": "Tree+LL128", "ring_simple": "Ring+S",
    "ring_ll": "Ring+LL", "ring_ll128": "Ring+LL128",
}
MODES = ["sequential", "overlap"]


def load_results():
    """Load results.json for each config."""
    data = {}
    for cfg in CONFIGS:
        f = NODE_SWEEP_DIR / cfg / "results.json"
        if f.is_file():
            data[cfg] = json.loads(f.read_text())
    return data


def load_auto_selections():
    """Load auto_selections.json for each config."""
    sels = {}
    for cfg in CONFIGS:
        f = NODE_SWEEP_DIR / cfg / "auto_selections.json"
        if f.is_file():
            sels[cfg] = json.loads(f.read_text())
    return sels


def load_raw_timings():
    """Load raw timing files (nanoseconds) for statistical analysis."""
    raw = {}  # {cfg: {mode: {size: {algo: [times_ms]}}}}

    for cfg in CONFIGS:
        raw[cfg] = {}
        for mode in MODES:
            raw[cfg][mode] = {}
            for size in SIZES:
                raw[cfg][mode][size] = {}
                for algo in ALGOS:
                    times = _find_timing_file(cfg, mode, size, algo)
                    if times is not None:
                        raw[cfg][mode][size][algo] = times
    return raw


def _find_timing_file(cfg, mode, size, algo):
    """Find and read a timing file, trying different naming patterns."""
    patterns = [
        NODE_SWEEP_DIR / cfg / f"times_ns_{cfg}_{mode}_{size}_{algo}.txt",
        SC_DIR / f"times_sc_{mode}_{size}_{algo}.txt",  # 2x4 fallback
    ]
    for p in patterns:
        if p.is_file():
            try:
                text = p.read_text().strip()
                if not text:
                    continue
                values = [float(x) for x in text.split("\n") if x.strip()]
                # Files are already in ms despite "times_ns" naming
                return values
            except (ValueError, IOError):
                continue
    return None


def parse_auto_algo(log_line):
    """Extract algorithm and protocol from NCCL log line."""
    algo = "?"
    proto = "?"
    if "Algo RING" in log_line:
        algo = "Ring"
    elif "Algo TREE" in log_line:
        algo = "Tree"
    if "proto SIMPLE" in log_line:
        proto = "Simple"
    elif "proto LL128" in log_line:
        proto = "LL128"
    elif "proto LL " in log_line:
        proto = "LL"
    return f"{algo}+{proto}"


def find_best_algo(size_data):
    """Find the best non-auto algorithm for a size."""
    best_algo = None
    best_t = float("inf")
    for algo in ALGOS:
        if algo == "auto":
            continue
        t = size_data.get(algo, float("inf"))
        if isinstance(t, (int, float)) and t > 0 and t < best_t:
            best_t = t
            best_algo = algo
    return best_algo, best_t


# ─────────────────────────────────────────────────────────────────
# Figure 1: Gap Heatmap
# ─────────────────────────────────────────────────────────────────
def fig_gap_heatmap(data):
    """2D heatmap: gap % as function of (node config × message size)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("NCCL AUTO Suboptimality: Topology × Message Size",
                 fontsize=15, fontweight="bold", y=1.02)

    for ax_idx, mode in enumerate(MODES):
        ax = axes[ax_idx]
        matrix = np.full((len(CONFIGS), len(SIZES)), np.nan)

        for i, cfg in enumerate(CONFIGS):
            if cfg not in data:
                continue
            for j, size in enumerate(SIZES):
                sd = data[cfg].get(mode, {}).get(size, {})
                auto_t = sd.get("auto", 0)
                _, best_t = find_best_algo(sd)
                if auto_t > 0 and best_t < float("inf"):
                    gap = (auto_t - best_t) / best_t * 100
                    matrix[i, j] = max(gap, 0)  # clip negatives to 0

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=70, interpolation="nearest")

        # Annotate cells
        for i in range(len(CONFIGS)):
            for j in range(len(SIZES)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val > 40 else "black"
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=color)

        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels(SIZES, fontsize=10)
        ax.set_yticks(range(len(CONFIGS)))
        ax.set_yticklabels(["1×8\n(1 node)", "2×4\n(2 nodes)", "4×2\n(4 nodes)"],
                           fontsize=10)
        ax.set_xlabel("Message Size", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Topology", fontsize=11)
        mode_label = "Sequential" if mode == "sequential" else "Overlap"
        ax.set_title(f"{mode_label} Mode", fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("AUTO Gap (% slower than optimal)", fontsize=11)

    out = FIGURES_DIR / "fig_gap_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 2: AUTO Selection Comparison
# ─────────────────────────────────────────────────────────────────
def fig_auto_selection(data, auto_sels):
    """Show what AUTO picked vs what was actually optimal at each topology."""
    fig, axes = plt.subplots(len(CONFIGS), 1, figsize=(12, 8), sharex=True)
    fig.suptitle("NCCL AUTO Algorithm Choice vs Optimal\n(Sequential Mode)",
                 fontsize=15, fontweight="bold", y=1.02)

    # Color mapping for algo families
    algo_colors = {
        "Ring+Simple": "#22c55e", "Ring+LL": "#14b8a6", "Ring+LL128": "#06b6d4",
        "Tree+Simple": "#ef4444", "Tree+LL": "#f97316", "Tree+LL128": "#fbbf24",
        "AUTO": "#6b7280",
    }

    for row, cfg in enumerate(CONFIGS):
        ax = axes[row]
        auto_choices = []
        best_choices = []
        gap_values = []

        for size in SIZES:
            # What AUTO chose
            key = f"sequential_{size}"
            sel_line = auto_sels.get(cfg, {}).get(key, [""])[0]
            auto_algo = parse_auto_algo(sel_line)
            auto_choices.append(auto_algo)

            # What was actually best
            sd = data[cfg].get("sequential", {}).get(size, {})
            best_algo, best_t = find_best_algo(sd)
            auto_t = sd.get("auto", 0)

            if best_algo:
                best_label = ALGO_LABELS.get(best_algo, best_algo)
                # Map to same format as auto
                ba_map = {
                    "tree_simple": "Tree+Simple", "tree_ll": "Tree+LL",
                    "tree_ll128": "Tree+LL128", "ring_simple": "Ring+Simple",
                    "ring_ll": "Ring+LL", "ring_ll128": "Ring+LL128",
                }
                best_choices.append(ba_map.get(best_algo, best_algo))
            else:
                best_choices.append("?")

            if auto_t > 0 and best_t < float("inf"):
                gap_values.append(max((auto_t - best_t) / best_t * 100, 0))
            else:
                gap_values.append(0)

        x = np.arange(len(SIZES))
        bar_width = 0.35

        # Draw bars for gap
        bars = ax.bar(x, gap_values, 0.7, color=[
            "#ef4444" if g > 20 else "#f59e0b" if g > 10 else "#22c55e"
            for g in gap_values
        ], alpha=0.7, edgecolor="white")

        # Annotate each bar with AUTO choice and best choice
        for i, (ac, bc, gv) in enumerate(zip(auto_choices, best_choices, gap_values)):
            match = ac == bc
            y_pos = max(gv + 2, 5)
            ax.text(i, y_pos, f"AUTO: {ac}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                    color="#6b7280")
            ax.text(i, y_pos + 8, f"Best: {bc}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                    color="#22c55e" if match else "#ef4444")
            if match:
                ax.text(i, gv + 0.5, "✓", ha="center", va="bottom",
                        fontsize=10, color="#22c55e")

        node_labels = {"1x8": "1×8 (single node)", "2x4": "2×4 (2 nodes)",
                       "4x2": "4×2 (4 nodes)"}
        ax.set_ylabel("Gap (%)", fontsize=10)
        ax.set_title(node_labels[cfg], fontsize=12, fontweight="bold", loc="left")
        ax.set_ylim(0, max(max(gap_values) * 1.5, 20))
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0, color="gray", linewidth=0.5)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(SIZES, fontsize=10)
    axes[-1].set_xlabel("Message Size", fontsize=11)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_auto_selection_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 3: Statistical Analysis with Error Bars
# ─────────────────────────────────────────────────────────────────
def fig_error_bars(data, raw):
    """Show AUTO vs best with error bars (95% CI) at key sizes."""
    key_sizes = ["4MB", "16MB", "64MB", "256MB"]

    fig, axes = plt.subplots(1, len(key_sizes), figsize=(18, 5), sharey=False)
    fig.suptitle("AUTO vs Optimal: Statistical Significance (Sequential, 95% CI)",
                 fontsize=14, fontweight="bold", y=1.02)

    for col, size in enumerate(key_sizes):
        ax = axes[col]
        x = np.arange(len(CONFIGS))
        width = 0.35

        auto_medians = []
        auto_cis = []
        best_medians = []
        best_cis = []
        best_names = []

        for cfg in CONFIGS:
            sd = data.get(cfg, {}).get("sequential", {}).get(size, {})
            best_algo, _ = find_best_algo(sd)

            # AUTO raw data
            auto_raw = raw.get(cfg, {}).get("sequential", {}).get(size, {}).get("auto", [])
            best_raw = raw.get(cfg, {}).get("sequential", {}).get(size, {}).get(best_algo, []) if best_algo else []

            if auto_raw:
                auto_med = np.median(auto_raw)
                auto_ci = 1.96 * np.std(auto_raw) / np.sqrt(len(auto_raw))
            else:
                auto_med = sd.get("auto", 0)
                auto_ci = 0

            if best_raw:
                best_med = np.median(best_raw)
                best_ci = 1.96 * np.std(best_raw) / np.sqrt(len(best_raw))
            elif best_algo:
                best_med = sd.get(best_algo, 0)
                best_ci = 0
            else:
                best_med = 0
                best_ci = 0

            auto_medians.append(auto_med)
            auto_cis.append(auto_ci)
            best_medians.append(best_med)
            best_cis.append(best_ci)
            best_names.append(ALGO_LABELS.get(best_algo, "?"))

        ax.bar(x - width/2, auto_medians, width, yerr=auto_cis,
               color="#6b7280", alpha=0.8, label="AUTO", edgecolor="white",
               capsize=4, error_kw={"linewidth": 1.5})
        ax.bar(x + width/2, best_medians, width, yerr=best_cis,
               color="#22c55e", alpha=0.8, label="Best", edgecolor="white",
               capsize=4, error_kw={"linewidth": 1.5})

        # Annotate gap % and best algo name
        for i in range(len(CONFIGS)):
            if best_medians[i] > 0:
                gap = (auto_medians[i] - best_medians[i]) / best_medians[i] * 100
                y_max = max(auto_medians[i], best_medians[i])
                ax.text(i, y_max * 1.05, f"+{gap:.0f}%",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color="#ef4444" if gap > 15 else "#f59e0b" if gap > 5 else "#22c55e")
                ax.text(i + width/2, best_medians[i] * 0.5, best_names[i],
                        ha="center", va="center", fontsize=7, color="white",
                        fontweight="bold", rotation=90)

        ax.set_title(size, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["1×8", "2×4", "4×2"], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.set_ylabel("Median Latency (ms)", fontsize=11)
            ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    out = FIGURES_DIR / "fig_error_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 4: Bandwidth Utilization
# ─────────────────────────────────────────────────────────────────
def fig_bandwidth(data):
    """Show achieved bandwidth (GB/s) for AUTO vs best across topologies."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Achieved AllReduce Bandwidth: AUTO vs Optimal",
                 fontsize=15, fontweight="bold", y=1.02)

    for ax_idx, mode in enumerate(MODES):
        ax = axes[ax_idx]

        for cfg_idx, cfg in enumerate(CONFIGS):
            auto_bw = []
            best_bw = []
            size_labels = []

            for size in SIZES:
                sd = data.get(cfg, {}).get(mode, {}).get(size, {})
                auto_t = sd.get("auto", 0)
                _, best_t = find_best_algo(sd)
                nbytes = SIZE_BYTES[size]

                if auto_t > 0:
                    # AllReduce effective bandwidth: 2*(N-1)/N * size / time
                    # For 8 GPUs: factor = 2*7/8 = 1.75
                    factor = 2 * 7 / 8
                    auto_bw.append(factor * nbytes / (auto_t * 1e-3) / 1e9)  # GB/s
                else:
                    auto_bw.append(0)

                if best_t < float("inf") and best_t > 0:
                    factor = 2 * 7 / 8
                    best_bw.append(factor * nbytes / (best_t * 1e-3) / 1e9)
                else:
                    best_bw.append(0)

            x = np.arange(len(SIZES))
            style = {"1x8": ("o", "-"), "2x4": ("s", "--"), "4x2": ("D", ":")}
            marker, ls = style[cfg]
            colors = {"1x8": "#3b82f6", "2x4": "#f59e0b", "4x2": "#ef4444"}

            ax.plot(x, auto_bw, marker=marker, linestyle=ls,
                    color=colors[cfg], alpha=0.4, linewidth=1.5,
                    markersize=6)
            ax.plot(x, best_bw, marker=marker, linestyle=ls,
                    color=colors[cfg], alpha=0.9, linewidth=2.5,
                    markersize=8, label=f"{cfg} best" if ax_idx == 0 else None)

            # Shade the gap
            ax.fill_between(x, auto_bw, best_bw, alpha=0.1, color=colors[cfg])

        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels(SIZES, fontsize=10)
        ax.set_xlabel("Message Size", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=11)
        mode_label = "Sequential" if mode == "sequential" else "Overlap"
        ax.set_title(f"{mode_label} Mode", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

        if ax_idx == 0:
            # Custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color="#3b82f6", linewidth=2.5, label="1×8 (best)",
                       marker="o", markersize=8),
                Line2D([0], [0], color="#3b82f6", linewidth=1.5, alpha=0.4,
                       label="1×8 (AUTO)", marker="o", markersize=6),
                Line2D([0], [0], color="#f59e0b", linewidth=2.5, label="2×4 (best)",
                       marker="s", markersize=8, linestyle="--"),
                Line2D([0], [0], color="#f59e0b", linewidth=1.5, alpha=0.4,
                       label="2×4 (AUTO)", marker="s", markersize=6, linestyle="--"),
                Line2D([0], [0], color="#ef4444", linewidth=2.5, label="4×2 (best)",
                       marker="D", markersize=8, linestyle=":"),
                Line2D([0], [0], color="#ef4444", linewidth=1.5, alpha=0.4,
                       label="4×2 (AUTO)", marker="D", markersize=6, linestyle=":"),
            ]
            ax.legend(handles=legend_elements, fontsize=8, loc="upper left",
                      ncol=2)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_bandwidth_utilization.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Print: Statistical Summary
# ─────────────────────────────────────────────────────────────────
def print_statistics(raw):
    """Print statistical analysis for key configs."""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (50 iterations, 95% CI)")
    print("=" * 80)

    for cfg in CONFIGS:
        print(f"\n  {cfg} — Sequential Mode:")
        print(f"  {'Size':>6s}  {'Algo':>12s}  {'Median':>8s}  {'Mean':>8s}  "
              f"{'Std':>8s}  {'95%CI':>8s}  {'P5':>8s}  {'P95':>8s}  {'N':>4s}")
        print("  " + "-" * 85)

        for size in SIZES:
            for algo in ["auto"]:
                times = raw.get(cfg, {}).get("sequential", {}).get(size, {}).get(algo, [])
                if times:
                    arr = np.array(times)
                    med = np.median(arr)
                    mean = np.mean(arr)
                    std = np.std(arr)
                    ci = 1.96 * std / np.sqrt(len(arr))
                    p5 = np.percentile(arr, 5)
                    p95 = np.percentile(arr, 95)
                    print(f"  {size:>6s}  {'AUTO':>12s}  {med:>8.2f}  {mean:>8.2f}  "
                          f"{std:>8.2f}  {ci:>7.2f}  {p5:>8.2f}  {p95:>8.2f}  {len(arr):>4d}")

            # Also print best algo
            sd = {}
            for a in ALGOS:
                t = raw.get(cfg, {}).get("sequential", {}).get(size, {}).get(a, [])
                if t:
                    sd[a] = np.median(t)
            if sd:
                best_a = min([a for a in sd if a != "auto"], key=lambda a: sd.get(a, 1e9))
                times = raw[cfg]["sequential"][size].get(best_a, [])
                if times:
                    arr = np.array(times)
                    med = np.median(arr)
                    mean = np.mean(arr)
                    std = np.std(arr)
                    ci = 1.96 * std / np.sqrt(len(arr))
                    p5 = np.percentile(arr, 5)
                    p95 = np.percentile(arr, 95)
                    label = ALGO_LABELS.get(best_a, best_a)
                    print(f"  {'':>6s}  {label:>12s}  {med:>8.2f}  {mean:>8.2f}  "
                          f"{std:>8.2f}  {ci:>7.2f}  {p5:>8.2f}  {p95:>8.2f}  {len(arr):>4d}")


def main():
    print("Loading data...")
    data = load_results()
    auto_sels = load_auto_selections()
    raw = load_raw_timings()

    n_raw = sum(1 for c in raw for m in raw[c] for s in raw[c][m]
                for a in raw[c][m][s] if raw[c][m][s][a])
    print(f"Loaded {len(data)} configs, {n_raw} raw timing series")

    print("\nGenerating figures...")
    fig_gap_heatmap(data)
    fig_auto_selection(data, auto_sels)
    fig_error_bars(data, raw)
    fig_bandwidth(data)

    print_statistics(raw)
    print("\nDone!")


if __name__ == "__main__":
    main()
