"""
Generate paper-quality figures for RL bandit tuner validation.

Figures:
  Fig D: Bandit learning curve — arm-colored exploration + stable exploitation
  Fig E: Headline comparison — AUTO vs Bandit median latency with safety gate
  Fig F: Variance reduction — box plots showing tighter distributions

Usage:
    python3 generate_bandit_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR = Path(__file__).parent

# Data directories
BANDIT_DIR_V1 = SCRIPT_DIR / "bandit_results_download" / "bandit_validation"
BANDIT_DIR_V2 = SCRIPT_DIR / "bandit_results_v2" / "bandit_validation"

FIGURES_DIR = SCRIPT_DIR / "results" / "paper_figures"

TOPOLOGIES = ["2x4", "4x2"]
TOPO_LABELS = {"2x4": "2×4  (2 nodes × 4 GPUs)", "4x2": "4×2  (4 nodes × 2 GPUs)"}
TOPO_SHORT = {"2x4": "2×4", "4x2": "4×2"}
KEY_SIZES = ["64MB", "256MB"]

NUM_ARMS = 4
ARM_NAMES = ["Tree+Simple", "Tree+LL128", "Ring+Simple", "AUTO"]
ARM_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#6b7280"]
ARM_MARKERS = ["^", "s", "v", "o"]

# Clean color palette
AUTO_COLOR = "#64748b"       # Slate gray
BANDIT_COLOR = "#7c3aed"     # Violet
EXPLOIT_BG = "#f5f3ff"       # Light violet background
WIN_COLOR = "#16a34a"        # Green
LOSS_COLOR = "#dc2626"       # Red
NEUTRAL_COLOR = "#d97706"    # Amber

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
})


def load_timings(data_dir, topo, label, size):
    filepath = data_dir / topo / f"times_{label}_{size}.txt"
    if filepath.is_file():
        lines = filepath.read_text().strip().split("\n")
        return [float(x) for x in lines if x.strip()]
    return []


def load_rewards(data_dir, topo, size):
    filepath = data_dir / topo / f"rewards_{size}.log"
    if filepath.is_file():
        lines = filepath.read_text().strip().split("\n")
        latencies = []
        for line in lines:
            if line.strip() and not line.startswith("#"):
                parts = line.split(",")
                if len(parts) >= 5:
                    latencies.append(float(parts[4]))
        return latencies
    return []


def load_bandit_results(data_dir, topo):
    filepath = data_dir / topo / "bandit_results.json"
    if filepath.is_file():
        return json.loads(filepath.read_text())
    return None


def get_best_data_dir(topo):
    if (BANDIT_DIR_V2 / topo / "bandit_results.json").is_file():
        return BANDIT_DIR_V2
    if (BANDIT_DIR_V1 / topo / "bandit_results.json").is_file():
        return BANDIT_DIR_V1
    return None


def iqr_clip(vals, factor=1.5):
    """Clip values to IQR range for cleaner plots."""
    arr = np.array(vals)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    return arr[(arr >= lo) & (arr <= hi)]


# ---------------------------------------------------------------------------
# Fig D: Bandit Learning Curve
# ---------------------------------------------------------------------------
def generate_learning_curve():
    """
    Two-row figure (one per topology), two columns (one per size).
    Each panel: per-iteration latency colored by arm during exploration,
    single color during exploitation, AUTO median as reference line.
    Inset box shows per-arm trimmed mean from exploration.
    """
    fig, axes = plt.subplots(len(TOPOLOGIES), len(KEY_SIZES),
                              figsize=(16, 5.5 * len(TOPOLOGIES)))
    fig.suptitle("Bandit Learning: Exploration → Exploitation Transition",
                 fontsize=16, fontweight="bold", y=0.98)

    has_data = False

    for row_idx, topo in enumerate(TOPOLOGIES):
        data_dir = get_best_data_dir(topo)
        if data_dir is None:
            continue

        results = load_bandit_results(data_dir, topo)
        if results is None:
            continue

        explore_rounds = results.get("explore_rounds", 5)
        explore_iters = explore_rounds * NUM_ARMS

        for col_idx, size in enumerate(KEY_SIZES):
            ax = axes[row_idx][col_idx]

            auto_times = load_timings(data_dir, topo, "auto", size)
            bandit_times = load_timings(data_dir, topo, "bandit", size)

            if not auto_times or not bandit_times:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            has_data = True
            n = len(bandit_times)

            # AUTO reference
            auto_arr = np.array(auto_times)
            auto_med = np.median(auto_arr)
            auto_clipped = iqr_clip(auto_times)
            auto_q1, auto_q3 = np.percentile(auto_clipped, [25, 75])

            # Y-axis: clip to see the signal, not the outliers
            all_clipped = iqr_clip(bandit_times + auto_times, factor=2.5)
            y_max = np.percentile(all_clipped, 98) * 1.15

            # Background: light violet for exploit phase
            ax.axvspan(explore_iters, n, color=EXPLOIT_BG, alpha=0.5, zorder=0)

            # AUTO band
            ax.axhline(auto_med, color=AUTO_COLOR, linestyle="--", linewidth=2,
                       zorder=6, label=f"AUTO median: {auto_med:.0f}ms")
            ax.fill_between([0, n], auto_q1, auto_q3, color=AUTO_COLOR,
                            alpha=0.08, zorder=1)

            # EXPLORATION: color each point by which arm was active
            for i in range(min(explore_iters, n)):
                arm_idx = i % NUM_ARMS
                y_val = min(bandit_times[i], y_max * 0.98)  # clip for display
                ax.scatter(i, y_val, c=ARM_COLORS[arm_idx], s=30, alpha=0.7,
                           marker=ARM_MARKERS[arm_idx], zorder=5,
                           edgecolors="white", linewidths=0.3)

            # EXPLOITATION: single color scatter + rolling median
            if n > explore_iters:
                exploit_x = np.arange(explore_iters, n)
                exploit_y = np.array(bandit_times[explore_iters:])
                exploit_y_clipped = np.minimum(exploit_y, y_max * 0.98)

                ax.scatter(exploit_x, exploit_y_clipped, c=BANDIT_COLOR, s=12,
                           alpha=0.25, zorder=4, edgecolors="none")

                # Rolling median (window=20)
                window = 20
                rolling = np.array([
                    np.median(exploit_y[max(0, j-window):j+1])
                    for j in range(len(exploit_y))
                ])
                ax.plot(exploit_x, rolling, color=BANDIT_COLOR, linewidth=2.5,
                        alpha=0.9, zorder=7, label=f"Bandit rolling median")

                # Exploit median annotation
                exploit_med = np.median(exploit_y)
                gain = (auto_med - exploit_med) / auto_med * 100
                gain_color = WIN_COLOR if gain >= 5 else LOSS_COLOR if gain < 0 else NEUTRAL_COLOR

                ax.annotate(
                    f"Exploit: {exploit_med:.0f}ms ({gain:+.1f}%)",
                    xy=(n * 0.98, exploit_med), fontsize=10, fontweight="bold",
                    color=gain_color, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=gain_color,
                              alpha=0.95, linewidth=1.5))

            # Explore/exploit boundary
            ax.axvline(explore_iters, color="#ef4444", linestyle=":", linewidth=1.5,
                       alpha=0.6, zorder=8)
            ax.text(explore_iters + 2, y_max * 0.95, "exploit →",
                    fontsize=9, color="#ef4444", alpha=0.7, va="top")
            ax.text(explore_iters - 2, y_max * 0.95, "← explore",
                    fontsize=9, color="#ef4444", alpha=0.7, va="top", ha="right")

            # Per-arm exploration summary (inset text)
            arm_means = []
            for a in range(NUM_ARMS):
                arm_vals = [bandit_times[i] for i in range(min(explore_iters, n))
                            if i % NUM_ARMS == a]
                if arm_vals:
                    trimmed = iqr_clip(arm_vals)
                    arm_means.append((ARM_NAMES[a], np.mean(trimmed) if len(trimmed) > 0 else np.mean(arm_vals)))
                else:
                    arm_means.append((ARM_NAMES[a], float('inf')))

            # Sort by mean and format
            arm_means.sort(key=lambda x: x[1])
            inset_lines = ["Explore means:"]
            for name, mean in arm_means:
                if mean < float('inf'):
                    best_marker = " ★" if mean == arm_means[0][1] else ""
                    inset_lines.append(f"  {name}: {mean:.0f}ms{best_marker}")

            ax.text(0.02, 0.97, "\n".join(inset_lines),
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#e2e8f0",
                              alpha=0.9))

            ax.set_title(f"{TOPO_SHORT[topo]} — {size} AllReduce", pad=10)
            ax.set_xlabel("Iteration")
            if col_idx == 0:
                ax.set_ylabel("Latency (ms)")
            ax.set_ylim(0, y_max)
            ax.set_xlim(-2, n + 5)

    if not has_data:
        print("No data for learning curve figure.")
        plt.close(fig)
        return

    # Shared legend for arm colors
    arm_handles = [plt.Line2D([0], [0], marker=ARM_MARKERS[i], color="w",
                               markerfacecolor=ARM_COLORS[i], markersize=8,
                               label=ARM_NAMES[i])
                    for i in range(NUM_ARMS)]
    arm_handles.append(plt.Line2D([0], [0], color=AUTO_COLOR, linestyle="--",
                                   linewidth=2, label="AUTO median"))
    arm_handles.append(plt.Line2D([0], [0], color=BANDIT_COLOR, linewidth=2.5,
                                   label="Bandit rolling median"))

    fig.legend(handles=arm_handles, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, 0.01), frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig_bandit_convergence.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig E: Headline Comparison
# ---------------------------------------------------------------------------
def generate_headline_figure():
    """
    Single row of 4 panels (one per topo×size), each showing:
    AUTO bar vs Bandit bar with speedup %, safety gate decision,
    and error bars (IQR).
    """
    scenarios = []
    for topo in TOPOLOGIES:
        for size in KEY_SIZES:
            scenarios.append((topo, size))

    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.5 * len(scenarios), 5.5))
    if len(scenarios) == 1:
        axes = [axes]
    fig.suptitle("RL Bandit vs NCCL AUTO: Per-Scenario Results",
                 fontsize=16, fontweight="bold", y=1.02)

    for idx, (topo, size) in enumerate(scenarios):
        ax = axes[idx]
        data_dir = get_best_data_dir(topo)
        if data_dir is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        results = load_bandit_results(data_dir, topo)
        if results is None:
            continue

        explore_rounds = results.get("explore_rounds", 5)
        explore_iters = explore_rounds * NUM_ARMS

        auto_times = load_timings(data_dir, topo, "auto", size)
        bandit_times = load_timings(data_dir, topo, "bandit", size)

        if not auto_times or not bandit_times:
            continue

        # Compute stats on exploit phase only
        exploit_times = bandit_times[explore_iters:] if len(bandit_times) > explore_iters else bandit_times

        auto_arr = np.array(auto_times)
        exploit_arr = np.array(exploit_times)

        auto_med = np.median(auto_arr)
        auto_q1, auto_q3 = np.percentile(auto_arr, [25, 75])

        bandit_med = np.median(exploit_arr)
        bandit_q1, bandit_q3 = np.percentile(exploit_arr, [25, 75])

        gain = (auto_med - bandit_med) / auto_med * 100

        # Bars
        x = [0, 1]
        heights = [auto_med, bandit_med]
        colors = [AUTO_COLOR, BANDIT_COLOR]
        yerr_low = [auto_med - auto_q1, bandit_med - bandit_q1]
        yerr_high = [auto_q3 - auto_med, bandit_q3 - bandit_med]

        bars = ax.bar(x, heights, width=0.6, color=colors, edgecolor="white",
                      linewidth=1, zorder=3)
        ax.errorbar(x, heights, yerr=[yerr_low, yerr_high],
                    fmt="none", ecolor="black", elinewidth=1.5, capsize=6,
                    capthick=1.5, zorder=4)

        # Median value labels inside bars
        for i, (h, c) in enumerate(zip(heights, colors)):
            ax.text(i, h * 0.5, f"{h:.0f}ms", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white", zorder=5)

        # Speedup annotation above bandit bar
        gain_color = WIN_COLOR if gain >= 5 else LOSS_COLOR if gain < 0 else NEUTRAL_COLOR
        size_results = results.get("results", {}).get(size, {})
        decision = size_results.get("decision", "")

        annotation = f"{gain:+.1f}%"
        if decision:
            annotation += f"\n{decision}"

        ax.text(1, bandit_q3 + auto_med * 0.05, annotation,
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=gain_color)

        # Variance reduction annotation
        auto_std = np.std(auto_arr)
        bandit_std = np.std(exploit_arr)
        var_change = (auto_std - bandit_std) / auto_std * 100
        if abs(var_change) > 5:
            var_label = f"σ: {var_change:+.0f}%"
            var_color = WIN_COLOR if var_change > 0 else LOSS_COLOR
            ax.text(1, bandit_med * 0.15, var_label, ha="center", va="center",
                    fontsize=8, color=var_color, alpha=0.8, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(["AUTO", "Bandit"], fontsize=11)
        ax.set_title(f"{TOPO_SHORT[topo]} — {size}", pad=10)
        ax.set_ylabel("Median Latency (ms)" if idx == 0 else "")
        ax.set_ylim(0, max(auto_q3, bandit_q3) * 1.35)
        ax.grid(axis="y")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig_bandit_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig F: Variance Reduction
# ---------------------------------------------------------------------------
def generate_variance_figure():
    """
    Box plots emphasizing that the bandit not only shifts the median
    but tightens the distribution. Two panels per topology (64MB, 256MB).
    Each panel: side-by-side box plots with IQR annotated.
    """
    fig, axes = plt.subplots(len(TOPOLOGIES), len(KEY_SIZES),
                              figsize=(14, 5 * len(TOPOLOGIES)))
    fig.suptitle("Variance Reduction: Bandit Produces More Predictable Latency",
                 fontsize=16, fontweight="bold", y=0.98)

    has_data = False

    for row_idx, topo in enumerate(TOPOLOGIES):
        data_dir = get_best_data_dir(topo)
        if data_dir is None:
            continue

        results = load_bandit_results(data_dir, topo)
        if results is None:
            continue

        explore_rounds = results.get("explore_rounds", 5)
        explore_iters = explore_rounds * NUM_ARMS

        for col_idx, size in enumerate(KEY_SIZES):
            ax = axes[row_idx][col_idx]

            auto_times = load_timings(data_dir, topo, "auto", size)
            bandit_times = load_timings(data_dir, topo, "bandit", size)

            if not auto_times or not bandit_times:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            has_data = True
            exploit_times = bandit_times[explore_iters:] if len(bandit_times) > explore_iters else bandit_times

            # IQR-clipped for cleaner box plots (remove extreme outliers only)
            auto_clean = iqr_clip(auto_times, factor=3.0)
            bandit_clean = iqr_clip(exploit_times, factor=3.0)

            # Box plots
            bp = ax.boxplot(
                [auto_clean, bandit_clean],
                positions=[0, 1],
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
            )

            bp["boxes"][0].set_facecolor(AUTO_COLOR)
            bp["boxes"][0].set_alpha(0.6)
            bp["boxes"][1].set_facecolor(BANDIT_COLOR)
            bp["boxes"][1].set_alpha(0.6)

            # Jittered points behind boxes
            for i, (pos, vals) in enumerate([(0, auto_clean), (1, bandit_clean)]):
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                color = AUTO_COLOR if i == 0 else BANDIT_COLOR
                ax.scatter(pos + jitter, vals, c=color, s=8, alpha=0.2,
                           zorder=2, edgecolors="none")

            # Stats annotations
            auto_arr = np.array(auto_times)
            exploit_arr = np.array(exploit_times)

            auto_med = np.median(auto_arr)
            bandit_med = np.median(exploit_arr)
            auto_iqr = np.percentile(auto_arr, 75) - np.percentile(auto_arr, 25)
            bandit_iqr = np.percentile(exploit_arr, 75) - np.percentile(exploit_arr, 25)
            auto_std = np.std(auto_arr)
            bandit_std = np.std(exploit_arr)

            gain = (auto_med - bandit_med) / auto_med * 100
            iqr_reduction = (auto_iqr - bandit_iqr) / auto_iqr * 100
            std_reduction = (auto_std - bandit_std) / auto_std * 100

            # Stats table
            stats_text = (
                f"        {'AUTO':>8s}  {'Bandit':>8s}\n"
                f"Median  {auto_med:>7.0f}ms  {bandit_med:>7.0f}ms\n"
                f"IQR     {auto_iqr:>7.0f}ms  {bandit_iqr:>7.0f}ms\n"
                f"Std     {auto_std:>7.0f}ms  {bandit_std:>7.0f}ms\n"
                f"─────────────────────────\n"
                f"Median Δ  {gain:+.1f}%\n"
                f"IQR Δ     {iqr_reduction:+.1f}%\n"
                f"Std Δ     {std_reduction:+.1f}%"
            )

            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                    fontsize=8, va="top", ha="right", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1",
                              alpha=0.95))

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["AUTO", "Bandit\n(exploit only)"], fontsize=11)
            ax.set_title(f"{TOPO_SHORT[topo]} — {size} AllReduce", pad=10)
            if col_idx == 0:
                ax.set_ylabel("Latency (ms)")

    if not has_data:
        print("No data for variance figure.")
        plt.close(fig)
        return

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig_bandit_variance.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary():
    print(f"\n{'='*70}")
    print(f"  RL BANDIT VALIDATION SUMMARY")
    print(f"{'='*70}")

    for topo in TOPOLOGIES:
        data_dir = get_best_data_dir(topo)
        if data_dir is None:
            print(f"\n  {TOPO_SHORT[topo]}: No data")
            continue

        results = load_bandit_results(data_dir, topo)
        if results is None:
            continue

        explore_rounds = results.get("explore_rounds", 5)
        explore_iters = explore_rounds * NUM_ARMS
        version = "v2 (10 explore, safety gate)" if data_dir == BANDIT_DIR_V2 else "v1 (5 explore)"
        print(f"\n  {TOPO_SHORT[topo]} [{version}]:")

        for size in KEY_SIZES:
            auto_times = load_timings(data_dir, topo, "auto", size)
            bandit_times = load_timings(data_dir, topo, "bandit", size)
            if not auto_times or not bandit_times:
                continue

            exploit_times = bandit_times[explore_iters:] if len(bandit_times) > explore_iters else bandit_times

            auto_med = np.median(auto_times)
            bandit_med = np.median(exploit_times)
            auto_std = np.std(auto_times)
            bandit_std = np.std(exploit_times)
            gain = (auto_med - bandit_med) / auto_med * 100
            std_change = (auto_std - bandit_std) / auto_std * 100

            size_data = results.get("results", {}).get(size, {})
            decision = size_data.get("decision", "N/A")

            print(f"    {size}: median {gain:+.1f}%  std {std_change:+.1f}%  → {decision}")


def main():
    print("Loading RL bandit validation results...")

    found = False
    for topo in TOPOLOGIES:
        d = get_best_data_dir(topo)
        if d:
            found = True
            label = "v2" if d == BANDIT_DIR_V2 else "v1"
            print(f"  {topo}: {label} ({d})")
        else:
            print(f"  {topo}: NO DATA")

    if not found:
        print("\nNo bandit data found.")
        return

    print_summary()

    print("\nGenerating figures...")
    generate_learning_curve()
    generate_headline_figure()
    generate_variance_figure()
    print("\nDone!")


if __name__ == "__main__":
    main()
