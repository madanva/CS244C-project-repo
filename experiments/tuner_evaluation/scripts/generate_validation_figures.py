"""
Generate evaluation figures for the profile-guided tuner validation.

Reads tuner_validation_v2/{1x8,2x4,4x2}/ results including:
  - validation_results.json (medians)
  - times_val_{mode}_{size}_{config}.txt (all n=50 raw timings)

Produces 3 figures:
  Fig A: Tuner speedup % across topologies × message sizes
  Fig B: Before/After absolute latency bars
  Fig C: Latency distribution violin plots (n=50)

Usage:
    python3 generate_validation_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
VALIDATION_DIR = RESULTS_DIR / "tuner_validation_v4"
FIGURES_DIR = RESULTS_DIR / "paper_figures"

TOPOLOGIES = ["1x8", "2x4", "4x2"]
TOPO_LABELS = {"1x8": "1×8", "2x4": "2×4", "4x2": "4×2"}
TOPO_COLORS = {"1x8": "#3b82f6", "2x4": "#f59e0b", "4x2": "#ef4444"}
TOPO_MARKERS = {"1x8": "o", "2x4": "s", "4x2": "D"}

SIZE_ORDER = ["256KB", "1MB", "4MB", "16MB", "64MB", "256MB"]

AUTO_COLOR = "#6b7280"
TUNER_COLOR = "#22c55e"


def load_validation_results():
    """Load validation_results.json for each topology."""
    data = {}
    for topo in TOPOLOGIES:
        results_file = VALIDATION_DIR / topo / "validation_results.json"
        if results_file.is_file():
            data[topo] = json.loads(results_file.read_text())
            print(f"  Loaded {topo}: {results_file}")
        else:
            print(f"  Missing {topo}: {results_file}")
    return data


def load_raw_timings(topo, mode, size, config):
    """Load raw timing values from times_val_{mode}_{size}_{config}.txt."""
    filename = f"times_val_{mode}_{size}_{config}.txt"
    filepath = VALIDATION_DIR / topo / filename
    if filepath.is_file():
        lines = filepath.read_text().strip().split("\n")
        return [float(x) for x in lines if x.strip()]
    return []


# ---------------------------------------------------------------------------
# Fig A: Tuner Speedup % Across Topologies
# ---------------------------------------------------------------------------
def generate_speedup_figure(data):
    """2-panel line plot: speedup % per size, one line per topology."""
    available = [t for t in TOPOLOGIES if t in data]
    if not available:
        print("No data for speedup figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle("Profile-Guided Tuner: Speedup Over NCCL AUTO",
                 fontsize=16, fontweight="bold", y=0.98)

    for ax_idx, mode in enumerate(["sequential", "overlap"]):
        ax = axes[ax_idx]
        ax.set_title(f"{'Sequential' if mode == 'sequential' else 'Overlap'} Mode",
                     fontsize=14, fontweight="bold", pad=12)

        for topo in available:
            validation = data[topo].get("validation", {}).get(mode, {})
            speedups = []
            x_valid = []
            for i, size in enumerate(SIZE_ORDER):
                size_data = validation.get(size, {})
                auto_t = size_data.get("auto", 0)
                tuner_t = size_data.get("tuner", 0)
                if auto_t > 0 and tuner_t > 0:
                    speedup = (auto_t - tuner_t) / auto_t * 100
                    speedups.append(speedup)
                    x_valid.append(i)

            if speedups:
                ax.plot(x_valid, speedups,
                        color=TOPO_COLORS[topo],
                        linewidth=2.5,
                        marker=TOPO_MARKERS[topo],
                        markersize=9,
                        label=TOPO_LABELS[topo],
                        alpha=0.9,
                        zorder=5)

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhspan(ymin=-100, ymax=0, color="#fee2e2", alpha=0.2)  # red zone = regression
        ax.set_xticks(range(len(SIZE_ORDER)))
        ax.set_xticklabels(SIZE_ORDER, fontsize=10)
        ax.set_xlabel("Message Size", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel("Tuner Speedup (%)\n(positive = tuner faster)", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=12,
               title="Topology", title_fontsize=13,
               bbox_to_anchor=(0.99, 0.5))

    plt.tight_layout(rect=[0, 0, 0.88, 0.94])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig_tuner_validation_speedup.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig B: Before/After Latency Comparison
# ---------------------------------------------------------------------------
def generate_comparison_figure(data):
    """3×2 grid: grouped bars AUTO vs Tuner per topology × mode."""
    available = [t for t in TOPOLOGIES if t in data]
    if not available:
        print("No data for comparison figure.")
        return

    fig, axes = plt.subplots(len(available), 2, figsize=(16, 5 * len(available)),
                              squeeze=False)
    fig.suptitle("Tuner vs AUTO: Absolute Latency Comparison",
                 fontsize=16, fontweight="bold", y=0.98)

    for row_idx, topo in enumerate(available):
        for col_idx, mode in enumerate(["sequential", "overlap"]):
            ax = axes[row_idx][col_idx]
            ax.set_title(f"{TOPO_LABELS[topo]} — {'Sequential' if mode == 'sequential' else 'Overlap'}",
                         fontsize=13, fontweight="bold", pad=10)

            validation = data[topo].get("validation", {}).get(mode, {})
            auto_vals = []
            tuner_vals = []
            valid_sizes = []

            for size in SIZE_ORDER:
                size_data = validation.get(size, {})
                auto_t = size_data.get("auto", 0)
                tuner_t = size_data.get("tuner", 0)
                if auto_t > 0 and tuner_t > 0:
                    auto_vals.append(auto_t)
                    tuner_vals.append(tuner_t)
                    valid_sizes.append(size)

            if not valid_sizes:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            x = np.arange(len(valid_sizes))
            bar_width = 0.35

            ax.bar(x - bar_width/2, auto_vals, bar_width,
                   color=AUTO_COLOR, label="AUTO", edgecolor="white", linewidth=0.5)
            ax.bar(x + bar_width/2, tuner_vals, bar_width,
                   color=TUNER_COLOR, label="Tuner", edgecolor="white", linewidth=0.5)

            # Add speedup annotations
            for i in range(len(valid_sizes)):
                if auto_vals[i] > 0:
                    speedup = (auto_vals[i] - tuner_vals[i]) / auto_vals[i] * 100
                    color = "#16a34a" if speedup > 0 else "#dc2626"
                    ax.annotate(f"{speedup:+.0f}%",
                                xy=(x[i] + bar_width/2, tuner_vals[i]),
                                xytext=(0, 5), textcoords="offset points",
                                ha="center", fontsize=8, fontweight="bold",
                                color=color)

            ax.set_xticks(x)
            ax.set_xticklabels(valid_sizes, fontsize=9)
            ax.set_xlabel("Message Size", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("Median Latency (ms)", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = FIGURES_DIR / "fig_tuner_validation_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig C: Latency Distribution Violin Plots (n=50)
# ---------------------------------------------------------------------------
def generate_distribution_figure(data):
    """Violin plots showing full n=50 distribution at key sizes."""
    available = [t for t in TOPOLOGIES if t in data]
    if not available:
        print("No data for distribution figure.")
        return

    key_sizes = ["64MB", "256MB"]
    modes = ["sequential", "overlap"]

    # Layout: rows = key sizes, cols = modes
    fig, axes = plt.subplots(len(key_sizes), len(modes),
                              figsize=(16, 5 * len(key_sizes)), squeeze=False)
    fig.suptitle("Latency Distribution: AUTO vs Tuner (n=50 iterations)",
                 fontsize=16, fontweight="bold", y=0.98)

    for row_idx, size in enumerate(key_sizes):
        for col_idx, mode in enumerate(modes):
            ax = axes[row_idx][col_idx]
            ax.set_title(f"{size} — {'Sequential' if mode == 'sequential' else 'Overlap'}",
                         fontsize=13, fontweight="bold", pad=10)

            positions = []
            violin_data = []
            violin_colors = []
            tick_positions = []
            tick_labels = []

            for topo_idx, topo in enumerate(available):
                base_pos = topo_idx * 3  # space between topology groups

                # Load raw timings
                auto_times = load_raw_timings(topo, mode, size, "auto")
                tuner_times = load_raw_timings(topo, mode, size, "tuner")

                if auto_times:
                    positions.append(base_pos)
                    violin_data.append(auto_times)
                    violin_colors.append(AUTO_COLOR)

                if tuner_times:
                    positions.append(base_pos + 1)
                    violin_data.append(tuner_times)
                    violin_colors.append(TUNER_COLOR)

                tick_positions.append(base_pos + 0.5)
                tick_labels.append(TOPO_LABELS[topo])

                # Print stats
                if auto_times and tuner_times:
                    auto_med = np.median(auto_times)
                    tuner_med = np.median(tuner_times)
                    speedup = (auto_med - tuner_med) / auto_med * 100
                    print(f"  {topo} {size} {mode}: AUTO median={auto_med:.1f}ms, "
                          f"Tuner median={tuner_med:.1f}ms, speedup={speedup:+.1f}%")

            if not violin_data:
                ax.text(0.5, 0.5, "No raw timing data available",
                        ha="center", va="center", transform=ax.transAxes, fontsize=11)
                continue

            # Draw violins
            parts = ax.violinplot(violin_data, positions=positions,
                                   showmeans=False, showmedians=True,
                                   showextrema=False, widths=0.8)

            # Color each violin
            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor(violin_colors[i])
                body.set_alpha(0.7)
                body.set_edgecolor("black")
                body.set_linewidth(0.5)

            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(2)

            # Add individual points (jittered) for each violin
            for i, (pos, times) in enumerate(zip(positions, violin_data)):
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(times))
                ax.scatter(pos + jitter, times,
                           c=violin_colors[i], alpha=0.3, s=8, zorder=3,
                           edgecolors="none")

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=11)
            ax.set_xlabel("Topology", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("Latency (ms)", fontsize=11)
            ax.grid(axis="y", alpha=0.3)

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=AUTO_COLOR, alpha=0.7, edgecolor="black",
                       linewidth=0.5, label="AUTO"),
        mpatches.Patch(facecolor=TUNER_COLOR, alpha=0.7, edgecolor="black",
                       linewidth=0.5, label="Tuner"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=12,
               bbox_to_anchor=(0.98, 0.96))

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = FIGURES_DIR / "fig_tuner_validation_distributions.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(data):
    """Print a text summary of validation results."""
    print(f"\n{'='*70}")
    print(f"  TUNER VALIDATION SUMMARY")
    print(f"{'='*70}")

    for mode in ("sequential", "overlap"):
        print(f"\n  {mode.upper()}:")
        header = f"  {'Size':<8s}"
        for topo in TOPOLOGIES:
            if topo in data:
                header += f"  {TOPO_LABELS[topo]:>12s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for size in SIZE_ORDER:
            row = f"  {size:<8s}"
            for topo in TOPOLOGIES:
                if topo not in data:
                    continue
                validation = data[topo].get("validation", {}).get(mode, {})
                size_data = validation.get(size, {})
                auto_t = size_data.get("auto", 0)
                tuner_t = size_data.get("tuner", 0)
                if auto_t > 0 and tuner_t > 0:
                    speedup = (auto_t - tuner_t) / auto_t * 100
                    row += f"  {speedup:>+10.1f}%"
                else:
                    row += f"  {'N/A':>12s}"
            print(row)

    # Policy summary
    print(f"\n  GENERATED POLICIES:")
    for topo in TOPOLOGIES:
        if topo in data:
            # Support both v2 (single policy) and v3 (per-mode policies)
            seq_pol = data[topo].get("policy_sequential", data[topo].get("policy", ""))
            ovl_pol = data[topo].get("policy_overlap", "")
            if ovl_pol and ovl_pol != seq_pol:
                print(f"    {TOPO_LABELS[topo]} seq: \"{seq_pol}\"")
                print(f"    {TOPO_LABELS[topo]} ovl: \"{ovl_pol}\"")
            else:
                print(f"    {TOPO_LABELS[topo]}: \"{seq_pol}\"")


def main():
    print("Loading tuner validation results...")
    data = load_validation_results()

    if not data:
        print("No validation data found. Run the validation experiment first:")
        print("  modal run run_tuner_validation_multinode.py")
        return

    print(f"\nFound data for topologies: {list(data.keys())}")

    print_summary(data)

    print("\nGenerating figures...")
    generate_speedup_figure(data)
    generate_comparison_figure(data)
    generate_distribution_figure(data)
    print("\nDone!")


if __name__ == "__main__":
    main()
