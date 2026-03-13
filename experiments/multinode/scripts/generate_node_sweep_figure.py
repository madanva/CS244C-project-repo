"""
Generate figure: AUTO gap amplification vs node count.

Reads node sweep results (1x8, 2x4, 4x2, 8x1) and shows how the
gap between NCCL AUTO and the best config grows with more inter-node hops.

Usage:
    python3 generate_node_sweep_figure.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
NODE_SWEEP_DIR = RESULTS_DIR / "node_sweep"
FIGURES_DIR = RESULTS_DIR / "paper_figures"

# Node configs in order of increasing inter-node ratio
NODE_CONFIGS = ["1x8", "2x4", "4x2", "8x1"]
NODE_LABELS = {
    "1x8": "1×8\n(0% inter-node)",
    "2x4": "2×4\n(~50% inter-node)",
    "4x2": "4×2\n(~75% inter-node)",
    "8x1": "8×1\n(100% inter-node)",
}

SIZE_ORDER = ["256KB", "1MB", "4MB", "16MB", "64MB", "256MB"]
SIZE_COLORS = {
    "256KB":  "#93c5fd",  # light blue
    "1MB":    "#60a5fa",  # blue
    "4MB":    "#3b82f6",  # medium blue
    "16MB":   "#f59e0b",  # amber
    "64MB":   "#ef4444",  # red (highlight)
    "256MB":  "#8b5cf6",  # purple
}
SIZE_LINEWIDTHS = {s: (3.0 if s == "64MB" else 1.5) for s in SIZE_ORDER}
SIZE_MARKERS = {s: ("D" if s == "64MB" else "o") for s in SIZE_ORDER}
SIZE_MARKERSIZES = {s: (10 if s == "64MB" else 6) for s in SIZE_ORDER}


def load_results():
    """Load results from each node config directory."""
    data = {}

    for cfg in NODE_CONFIGS:
        cfg_dir = NODE_SWEEP_DIR / cfg

        # Try node_sweep/{cfg}/results.json first
        results_file = cfg_dir / "results.json"
        if results_file.is_file():
            data[cfg] = json.loads(results_file.read_text())
            continue

        # For 2x4, fall back to same_cluster_comparison data
        if cfg == "2x4":
            sc_file = RESULTS_DIR / "same_cluster_comparison" / "same_cluster_results.json"
            if sc_file.is_file():
                data[cfg] = json.loads(sc_file.read_text())
                continue

    # Also try combined_sweep.json
    combined_file = NODE_SWEEP_DIR / "combined_sweep.json"
    if combined_file.is_file():
        combined = json.loads(combined_file.read_text())
        for cfg in NODE_CONFIGS:
            if cfg not in data and cfg in combined:
                entry = combined[cfg]
                if "results" in entry and not "error" in entry:
                    data[cfg] = entry["results"]

    return data


def compute_gaps(data):
    """Compute AUTO gap % for each config/size/mode."""
    gaps = {}  # {mode: {size: {config: gap%}}}

    for mode in ("sequential", "overlap"):
        gaps[mode] = {}
        for size in SIZE_ORDER:
            gaps[mode][size] = {}
            for cfg in NODE_CONFIGS:
                if cfg not in data:
                    continue
                size_data = data[cfg].get(mode, {}).get(size, {})
                valid = {k: v for k, v in size_data.items()
                         if isinstance(v, (int, float)) and v > 0}
                auto_t = valid.get("auto", 0)
                if valid and auto_t > 0:
                    best_t = min(valid.values())
                    gap = (auto_t - best_t) / auto_t * 100
                    gaps[mode][size][cfg] = round(gap, 1)

    return gaps


def generate_figure(data, gaps):
    """Generate the main gap-vs-node-count figure."""
    available_configs = [c for c in NODE_CONFIGS if c in data]

    if len(available_configs) < 2:
        print(f"Only {len(available_configs)} configs have data. Need at least 2.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle("NCCL AUTO Suboptimality Grows with Inter-Node Communication",
                 fontsize=16, fontweight="bold", y=0.98)

    x_positions = list(range(len(available_configs)))
    x_labels = [NODE_LABELS.get(c, c) for c in available_configs]

    for ax_idx, mode in enumerate(["sequential", "overlap"]):
        ax = axes[ax_idx]
        ax.set_title(f"{'Sequential' if mode == 'sequential' else 'Overlap'} Mode",
                     fontsize=14, fontweight="bold", pad=12)

        for size in SIZE_ORDER:
            y_values = []
            x_valid = []
            for i, cfg in enumerate(available_configs):
                gap = gaps[mode][size].get(cfg)
                if gap is not None:
                    y_values.append(gap)
                    x_valid.append(i)

            if len(y_values) >= 2:
                ax.plot(x_valid, y_values,
                        color=SIZE_COLORS[size],
                        linewidth=SIZE_LINEWIDTHS[size],
                        marker=SIZE_MARKERS[size],
                        markersize=SIZE_MARKERSIZES[size],
                        label=size,
                        alpha=0.9 if size == "64MB" else 0.7,
                        zorder=10 if size == "64MB" else 5)

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlabel("Node Configuration (nodes × GPUs/node)", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel("AUTO Gap (%)\n(higher = more suboptimal)", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=10)

    # Single legend on the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=11,
               title="Message Size", title_fontsize=12,
               bbox_to_anchor=(0.99, 0.5))

    plt.tight_layout(rect=[0, 0, 0.88, 0.94])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig_node_sweep_gap.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

    return out_path


def generate_absolute_bars(data):
    """Secondary figure: absolute latency at 64MB across node configs."""
    available_configs = [c for c in NODE_CONFIGS if c in data]
    if len(available_configs) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("AllReduce Latency at 64MB: Effect of Node Count",
                 fontsize=16, fontweight="bold", y=0.98)

    algo_colors = {
        "auto":        "#6b7280",
        "tree_simple": "#ef4444",
        "tree_ll":     "#f97316",
        "tree_ll128":  "#fbbf24",
        "ring_simple": "#22c55e",
        "ring_ll":     "#14b8a6",
        "ring_ll128":  "#06b6d4",
    }
    algo_labels = {
        "auto":        "AUTO",
        "tree_simple": "Tree+Simple",
        "tree_ll":     "Tree+LL",
        "tree_ll128":  "Tree+LL128",
        "ring_simple": "Ring+Simple",
        "ring_ll":     "Ring+LL",
        "ring_ll128":  "Ring+LL128",
    }

    for ax_idx, mode in enumerate(["sequential", "overlap"]):
        ax = axes[ax_idx]
        ax.set_title(f"{'Sequential' if mode == 'sequential' else 'Overlap'} Mode",
                     fontsize=14, fontweight="bold", pad=12)

        n_groups = len(available_configs)
        algos = list(algo_colors.keys())
        n_bars = len(algos)
        bar_width = 0.8 / n_bars
        x = np.arange(n_groups)

        for i, algo in enumerate(algos):
            values = []
            for cfg in available_configs:
                size_data = data[cfg].get(mode, {}).get("64MB", {})
                v = size_data.get(algo, 0)
                values.append(v if v > 0 else 0)

            offset = (i - n_bars / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, values, bar_width,
                          color=algo_colors[algo],
                          label=algo_labels[algo] if ax_idx == 0 else None,
                          edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([c for c in available_configs], fontsize=11)
        ax.set_xlabel("Node Configuration", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel("Median Latency (ms)", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=10,
               title="Config", title_fontsize=11,
               bbox_to_anchor=(0.99, 0.5))

    plt.tight_layout(rect=[0, 0, 0.86, 0.94])

    out_path = FIGURES_DIR / "fig_node_sweep_bars_64MB.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("Loading node sweep results...")
    data = load_results()

    if not data:
        print("No data found. Run the experiments first.")
        return

    print(f"Found data for configs: {list(data.keys())}")
    for cfg in data:
        modes = list(data[cfg].keys())
        sizes = list(data[cfg].get("sequential", {}).keys())
        print(f"  {cfg}: modes={modes}, sizes={sizes}")

    gaps = compute_gaps(data)

    # Print gap table
    for mode in ("sequential", "overlap"):
        print(f"\n  {mode.upper()} — AUTO Gap (%):")
        header = f"  {'Size':<8s}"
        for cfg in NODE_CONFIGS:
            if cfg in data:
                header += f"  {cfg:>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for size in SIZE_ORDER:
            row = f"  {size:<8s}"
            for cfg in NODE_CONFIGS:
                if cfg in data:
                    gap = gaps[mode][size].get(cfg)
                    if gap is not None:
                        row += f"  {gap:>+7.1f}%"
                    else:
                        row += f"  {'N/A':>8s}"
            print(row)

    print("\nGenerating figures...")
    generate_figure(data, gaps)
    generate_absolute_bars(data)
    print("Done.")


if __name__ == "__main__":
    main()
