"""
Cross-experiment analysis: combines data from all Phase 4 experiments to
produce the paper's key summary figures and the definitive results table.

Experiments combined:
  1. Sequential vs Overlap sweep (9 sizes × 5 configs)
  2. CTA/channel count sweep (8 CTAs × 4 sizes × 5 configs × 2 modes)
  3. Compute intensity sweep (4 sizes × 5 intensities × 5 configs)

Generates:
  1. paper_figure1_overview.png     — The main result: winner flip heatmap
  2. paper_figure2_cta_impact.png   — CTA count × message size under overlap
  3. paper_figure3_auto_gap.png     — AUTO gap: sequential vs overlap side-by-side
  4. paper_figure4_mechanism.png    — SM contention mechanism (64MB deep dive)
  5. paper_table1.txt               — Main results table (LaTeX-ready)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_ROOT = Path(__file__).parent / "results"
OVERLAP_FILE = RESULTS_ROOT / "overlap_experiment" / "overlap_experiment_results.json"
CHANNEL_FILE = RESULTS_ROOT / "channel_experiment" / "channel_experiment_results.json"
OUTPUT_DIR = RESULTS_ROOT / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE_ORDER = ["32KB", "64KB", "256KB", "1MB", "2MB", "4MB", "16MB", "64MB", "256MB"]
SIZE_BYTES = {
    "32KB": 32768, "64KB": 65536, "256KB": 262144,
    "1MB": 1048576, "2MB": 2097152, "4MB": 4194304,
    "16MB": 16777216, "64MB": 67108864, "256MB": 268435456,
}

CONFIG_SHORT = {
    "auto": "AUTO",
    "tree_simple": "Tree+S",
    "tree_ll128": "Tree+L",
    "ring_simple": "Ring+S",
    "ring_ll128": "Ring+L",
}
CONFIGS = ["auto", "tree_simple", "tree_ll128", "ring_simple", "ring_ll128"]
CONFIG_COLORS = {
    "auto": "#555555",
    "tree_simple": "#e74c3c",
    "tree_ll128": "#3498db",
    "ring_simple": "#2ecc71",
    "ring_ll128": "#f39c12",
}


def load_data():
    overlap = json.loads(OVERLAP_FILE.read_text()) if OVERLAP_FILE.exists() else {}
    channel = json.loads(CHANNEL_FILE.read_text()) if CHANNEL_FILE.exists() else {}
    return overlap, channel


def find_best(d):
    if not d:
        return "auto", 0
    return min(d, key=d.get), d[min(d, key=d.get)]


# ===================================================================
# Figure 1: Overview — Winner flip heatmap (THE main result)
# ===================================================================
def paper_figure1(overlap_data):
    """Side-by-side heatmap: best config under sequential vs overlap."""
    seq = overlap_data.get("sequential", {})
    ovl = overlap_data.get("overlap", {})
    sizes = [s for s in SIZE_ORDER if s in seq and s in ovl]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             gridspec_kw={"width_ratios": [1, 1, 0.6]})

    cfg_to_idx = {c: i for i, c in enumerate(CONFIGS)}
    colors = [CONFIG_COLORS[c] for c in CONFIGS]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)

    for ax_idx, (ax, mode_data, title) in enumerate([
        (axes[0], seq, "Sequential (No Overlap)"),
        (axes[1], ovl, "Overlap (Compute + Comm)"),
    ]):
        data = np.zeros(len(sizes), dtype=int)
        gaps = np.zeros(len(sizes))
        best_names = []

        for i, size in enumerate(sizes):
            best_cfg, best_t = find_best(mode_data[size])
            auto_t = mode_data[size].get("auto", 0)
            data[i] = cfg_to_idx.get(best_cfg, 0)
            gaps[i] = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0
            best_names.append(best_cfg)

        # Horizontal bar chart colored by best config
        bars = ax.barh(range(len(sizes)), gaps, color=[CONFIG_COLORS[best_names[i]] for i in range(len(sizes))],
                       edgecolor="white", linewidth=1.5, height=0.7)

        for i, (bar, name) in enumerate(zip(bars, best_names)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f"{CONFIG_SHORT[name]} ({gaps[i]:.1f}%)",
                    va="center", fontsize=9, fontweight="bold")

        ax.set_yticks(range(len(sizes)))
        ax.set_yticklabels(sizes, fontsize=11)
        ax.set_xlabel("AUTO Gap (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, max(gaps) * 1.6)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    # Panel 3: Winner flip indicator
    ax3 = axes[2]
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.5, len(sizes) - 0.5)

    for i, size in enumerate(sizes):
        seq_best = find_best(seq[size])[0]
        ovl_best = find_best(ovl[size])[0]
        flipped = seq_best != ovl_best

        color = "#e74c3c" if flipped else "#95a5a6"
        marker = "X" if flipped else "o"
        label = "FLIP" if flipped else "same"

        ax3.scatter(0.3, i, s=200, c=color, marker=marker, zorder=5,
                    edgecolors="black", linewidths=1)
        ax3.text(0.55, i, f"{CONFIG_SHORT[seq_best]} -> {CONFIG_SHORT[ovl_best]}",
                 va="center", fontsize=9,
                 fontweight="bold" if flipped else "normal",
                 color="#c0392b" if flipped else "#7f8c8d")

    ax3.set_yticks(range(len(sizes)))
    ax3.set_yticklabels(sizes, fontsize=11)
    ax3.set_xticks([])
    ax3.set_title("Winner Flips", fontsize=14, fontweight="bold")
    ax3.invert_yaxis()

    flip_count = sum(1 for s in sizes
                     if find_best(seq[s])[0] != find_best(ovl[s])[0])
    ax3.text(0.5, len(sizes) + 0.3,
             f"{flip_count}/{len(sizes)} sizes flip",
             ha="center", fontsize=12, fontweight="bold", color="#c0392b",
             transform=ax3.get_xaxis_transform())

    # Legend
    patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=CONFIG_SHORT[c]) for c in CONFIGS]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Optimal NCCL Configuration Shifts Under Compute-Communication Overlap\n"
                 "(AllReduce, 8x A100 NVLink, single-node)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "paper_figure1_overview.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ===================================================================
# Figure 2: CTA impact — U-curve at 64MB
# ===================================================================
def paper_figure2(channel_data):
    """CTA count impact: focus on 64MB where the effect is dramatic."""
    ovl_sweep = channel_data.get("overlap_cta_sweep", {})
    seq_sweep = channel_data.get("sequential_cta_sweep", {})

    if "64MB" not in ovl_sweep:
        print("  Skipping Figure 2 (no 64MB overlap data)")
        return

    cta_order = ["1", "2", "4", "8", "12", "16", "24", "32"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: All configs at 64MB overlap
    x = list(range(len(cta_order)))
    for cfg in CONFIGS:
        vals = [ovl_sweep["64MB"].get(c, {}).get(cfg, 0) for c in cta_order]
        ax1.plot(x, vals, "o-", label=CONFIG_SHORT[cfg],
                 color=CONFIG_COLORS[cfg], linewidth=2.5, markersize=8)

    ax1.fill_between(x,
                     [min(ovl_sweep["64MB"].get(c, {}).values()) for c in cta_order],
                     [max(ovl_sweep["64MB"].get(c, {}).values()) for c in cta_order],
                     alpha=0.1, color="gray")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}" for c in cta_order], fontsize=11)
    ax1.set_xlabel("NCCL_MAX_CTAS (SMs dedicated to communication)", fontsize=12)
    ax1.set_ylabel("Iteration Time (ms)", fontsize=12)
    ax1.set_title("64MB AllReduce Under Overlap:\nSM Contention Creates U-Shaped Curve",
                   fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(alpha=0.3)

    # Annotate the key insight
    best_idx = 4  # CTA=12
    worst_idx = 0  # CTA=1
    best_val = ovl_sweep["64MB"]["12"].get("auto", 0)
    worst_val = ovl_sweep["64MB"]["1"].get("auto", 0)
    speedup = (worst_val - best_val) / worst_val * 100

    ax1.annotate(f"CTA=1: {worst_val:.1f}ms\n(too few SMs for comm)",
                 xy=(worst_idx, worst_val), fontsize=9, fontweight="bold",
                 xytext=(worst_idx + 1.5, worst_val + 0.5),
                 arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2),
                 color="#e74c3c")
    ax1.annotate(f"CTA=12: {best_val:.1f}ms\n(Pareto optimal)",
                 xy=(best_idx, best_val), fontsize=9, fontweight="bold",
                 xytext=(best_idx + 1, best_val - 0.5),
                 arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2),
                 color="#2ecc71")

    # Panel 2: Best-config time per CTA, overlap vs sequential, ALL sizes
    sizes = [s for s in ["1MB", "4MB", "16MB", "64MB"] if s in ovl_sweep]
    width = 0.35
    x2 = np.arange(len(sizes))

    ovl_best_times = []
    seq_best_times = []
    ovl_best_ctas = []
    seq_best_ctas = []

    for size in sizes:
        # Find overall best across all CTAs
        ovl_min_t, ovl_min_cta = 999, "?"
        for c in cta_order:
            if c in ovl_sweep.get(size, {}):
                t = min(ovl_sweep[size][c].values())
                if t < ovl_min_t:
                    ovl_min_t, ovl_min_cta = t, c
        ovl_best_times.append(ovl_min_t)
        ovl_best_ctas.append(ovl_min_cta)

        seq_min_t, seq_min_cta = 999, "?"
        for c in cta_order:
            if c in seq_sweep.get(size, {}):
                t = min(seq_sweep[size][c].values())
                if t < seq_min_t:
                    seq_min_t, seq_min_cta = t, c
        seq_best_times.append(seq_min_t if seq_min_t < 999 else 0)
        seq_best_ctas.append(seq_min_cta)

    bars1 = ax2.bar(x2 - width/2, seq_best_times, width, label="Sequential",
                     color="#3498db", alpha=0.85, edgecolor="white")
    bars2 = ax2.bar(x2 + width/2, ovl_best_times, width, label="Overlap",
                     color="#e74c3c", alpha=0.85, edgecolor="white")

    # Add CTA labels
    for i, (bar, cta) in enumerate(zip(bars1, seq_best_ctas)):
        if seq_best_times[i] > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{cta}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                     color="#2c3e50")
    for i, (bar, cta) in enumerate(zip(bars2, ovl_best_ctas)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{cta}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color="#c0392b")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(sizes, fontsize=11)
    ax2.set_xlabel("Message Size", fontsize=12)
    ax2.set_ylabel("Best Iteration Time (ms)", fontsize=12)
    ax2.set_title("Optimal CTA Count Differs by Mode\n(numbers show optimal CTAs)",
                   fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("CTA Count Controls SM Contention Trade-off Under Overlap",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "paper_figure2_cta_impact.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ===================================================================
# Figure 3: AUTO gap comparison
# ===================================================================
def paper_figure3(overlap_data, channel_data):
    """AUTO gap amplification: sequential vs overlap, with CTA tuning boost."""
    seq = overlap_data.get("sequential", {})
    ovl = overlap_data.get("overlap", {})
    sizes = [s for s in SIZE_ORDER if s in seq and s in ovl]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: AUTO gap at default settings
    width = 0.35
    x = np.arange(len(sizes))

    seq_gaps = []
    ovl_gaps = []
    for size in sizes:
        seq_auto = seq[size].get("auto", 0)
        seq_best = min(seq[size].values())
        seq_gaps.append((seq_auto - seq_best) / seq_auto * 100 if seq_auto > 0 else 0)

        ovl_auto = ovl[size].get("auto", 0)
        ovl_best = min(ovl[size].values())
        ovl_gaps.append((ovl_auto - ovl_best) / ovl_auto * 100 if ovl_auto > 0 else 0)

    bars1 = ax1.bar(x - width/2, seq_gaps, width, label="Sequential",
                     color="#3498db", alpha=0.85)
    bars2 = ax1.bar(x + width/2, ovl_gaps, width, label="Overlap",
                     color="#e74c3c", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes, fontsize=9, rotation=45, ha="right")
    ax1.set_xlabel("Message Size", fontsize=12)
    ax1.set_ylabel("AUTO Gap vs Best Config (%)", fontsize=12)
    ax1.set_title("AUTO Suboptimality: Default CTA Count",
                   fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # Highlight where overlap gap > sequential gap
    for i in range(len(sizes)):
        if ovl_gaps[i] > seq_gaps[i] + 0.2:
            ax1.annotate("", xy=(x[i] + width/2, ovl_gaps[i]),
                        xytext=(x[i] + width/2, ovl_gaps[i] + 0.3),
                        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))

    # Panel 2: Compound gap with CTA tuning (from channel experiment)
    ovl_sweep = channel_data.get("overlap_cta_sweep", {})
    if ovl_sweep:
        cta_sizes = [s for s in SIZE_ORDER if s in ovl_sweep]
        x2 = np.arange(len(cta_sizes))

        # Gap: AUTO at default (CTA=8) vs our best (any CTA, any config)
        compound_gaps = []
        default_gaps = []
        for size in cta_sizes:
            default_auto = ovl_sweep[size].get("8", {}).get("auto", 0)

            # Best across all CTAs and configs
            best_t = 999
            for cta_data in ovl_sweep[size].values():
                t = min(cta_data.values())
                if t < best_t:
                    best_t = t

            # Default gap (AUTO at CTA=8 vs best at CTA=8)
            default_8_best = min(ovl_sweep[size].get("8", {}).values()) if "8" in ovl_sweep[size] else 0
            def_gap = (default_auto - default_8_best) / default_auto * 100 if default_auto > 0 else 0
            default_gaps.append(def_gap)

            # Compound gap (AUTO at CTA=8 vs best at any CTA)
            comp_gap = (default_auto - best_t) / default_auto * 100 if default_auto > 0 else 0
            compound_gaps.append(comp_gap)

        bars3 = ax2.bar(x2 - width/2, default_gaps, width, label="Config tuning only",
                         color="#f39c12", alpha=0.85)
        bars4 = ax2.bar(x2 + width/2, compound_gaps, width, label="Config + CTA tuning",
                         color="#e74c3c", alpha=0.85)

        ax2.set_xticks(x2)
        ax2.set_xticklabels(cta_sizes, fontsize=10)
        ax2.set_xlabel("Message Size", fontsize=12)
        ax2.set_ylabel("Improvement over AUTO@CTA=8 (%)", fontsize=12)
        ax2.set_title("Compound Gains: Config + CTA Tuning\n(Overlap mode)",
                       fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("AUTO Gap Grows Under Overlap; CTA Tuning Compounds the Gain",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "paper_figure3_auto_gap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ===================================================================
# Figure 4: SM contention mechanism — 64MB deep dive
# ===================================================================
def paper_figure4(channel_data):
    """The mechanism: show how protocols have different SM footprints."""
    ovl_sweep = channel_data.get("overlap_cta_sweep", {})
    seq_sweep = channel_data.get("sequential_cta_sweep", {})

    if "64MB" not in ovl_sweep:
        print("  Skipping Figure 4 (no 64MB data)")
        return

    cta_order = ["1", "2", "4", "8", "12", "16", "24", "32"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Overlap 64MB — all configs show U-curve
    ax1 = axes[0]
    x = list(range(len(cta_order)))
    for cfg in CONFIGS:
        if cfg == "auto":
            continue  # skip AUTO for clarity
        vals = [ovl_sweep["64MB"].get(c, {}).get(cfg, 0) for c in cta_order]
        ax1.plot(x, vals, "o-", label=CONFIG_SHORT[cfg],
                 color=CONFIG_COLORS[cfg], linewidth=2.5, markersize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cta_order, fontsize=10)
    ax1.set_xlabel("NCCL_MAX_CTAS", fontsize=12)
    ax1.set_ylabel("Iteration Time (ms)", fontsize=12)
    ax1.set_title("Overlap @ 64MB\n(compute steals SMs from comm)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: Sequential 64MB (if we have data)
    ax2 = axes[1]
    if "64MB" in seq_sweep:
        for cfg in CONFIGS:
            if cfg == "auto":
                continue
            vals = [seq_sweep["64MB"].get(c, {}).get(cfg, 0) for c in cta_order]
            if any(v > 0 for v in vals):
                ax2.plot(x[:len(vals)], vals, "o-", label=CONFIG_SHORT[cfg],
                         color=CONFIG_COLORS[cfg], linewidth=2.5, markersize=7)

        ax2.set_xticks(x)
        ax2.set_xticklabels(cta_order, fontsize=10)
        ax2.set_xlabel("NCCL_MAX_CTAS", fontsize=12)
        ax2.set_ylabel("Iteration Time (ms)", fontsize=12)
        ax2.set_title("Sequential @ 64MB\n(comm has all SMs)", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
    else:
        # Use 16MB sequential as proxy
        if "16MB" in seq_sweep:
            for cfg in CONFIGS:
                if cfg == "auto":
                    continue
                ctas_available = [c for c in cta_order if c in seq_sweep["16MB"]]
                vals = [seq_sweep["16MB"].get(c, {}).get(cfg, 0) for c in ctas_available]
                ax2.plot(range(len(vals)), vals, "o-", label=CONFIG_SHORT[cfg],
                         color=CONFIG_COLORS[cfg], linewidth=2.5, markersize=7)
            ax2.set_xticks(range(len(ctas_available)))
            ax2.set_xticklabels(ctas_available, fontsize=10)
        ax2.set_xlabel("NCCL_MAX_CTAS", fontsize=12)
        ax2.set_ylabel("Iteration Time (ms)", fontsize=12)
        ax2.set_title("Sequential @ 16MB\n(control: no SM contention)", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

    # Panel 3: Protocol sensitivity — spread at CTA=1 vs CTA=12
    ax3 = axes[2]
    cta_points = ["1", "4", "12", "32"]
    non_auto_configs = [c for c in CONFIGS if c != "auto"]

    for j, cta in enumerate(cta_points):
        if cta not in ovl_sweep.get("64MB", {}):
            continue
        data = ovl_sweep["64MB"][cta]
        vals = [data.get(cfg, 0) for cfg in non_auto_configs]
        spread = max(vals) - min(vals)
        mean_val = np.mean(vals)

        bar = ax3.bar(j, spread, color=plt.cm.RdYlGn_r(j / len(cta_points)),
                       edgecolor="black", linewidth=1, width=0.6)
        ax3.text(j, spread + 0.05, f"{spread:.1f}ms",
                 ha="center", fontsize=10, fontweight="bold")

    ax3.set_xticks(range(len(cta_points)))
    ax3.set_xticklabels([f"CTA={c}" for c in cta_points], fontsize=11)
    ax3.set_xlabel("CTA Count", fontsize=12)
    ax3.set_ylabel("Config Spread (max-min, ms)", fontsize=12)
    ax3.set_title("Protocol Sensitivity @ 64MB Overlap\n(spread = room for tuning)",
                   fontsize=13, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    plt.suptitle("The SM Contention Mechanism: Why Protocol Choice Matters Under Overlap",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "paper_figure4_mechanism.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ===================================================================
# Table 1: Main results table (LaTeX-ready)
# ===================================================================
def paper_table1(overlap_data, channel_data):
    """Generate the main results table."""
    seq = overlap_data.get("sequential", {})
    ovl = overlap_data.get("overlap", {})
    sizes = [s for s in SIZE_ORDER if s in seq and s in ovl]

    lines = []
    lines.append("=" * 120)
    lines.append("  TABLE 1: Workload-Aware Tuning Results (AllReduce, 8x A100 NVLink)")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"  {'Size':<8s} | {'SEQ Best':>12s} {'SEQ ms':>8s} {'Gap%':>6s} | "
                 f"{'OVL Best':>12s} {'OVL ms':>8s} {'Gap%':>6s} | {'Flip':>5s}")
    lines.append(f"  {'-'*95}")

    total_seq_saved = 0
    total_ovl_saved = 0
    flip_count = 0

    for size in sizes:
        seq_best_cfg, seq_best_t = find_best(seq[size])
        ovl_best_cfg, ovl_best_t = find_best(ovl[size])
        seq_auto = seq[size].get("auto", 0)
        ovl_auto = ovl[size].get("auto", 0)
        seq_gap = (seq_auto - seq_best_t) / seq_auto * 100 if seq_auto > 0 else 0
        ovl_gap = (ovl_auto - ovl_best_t) / ovl_auto * 100 if ovl_auto > 0 else 0
        flipped = seq_best_cfg != ovl_best_cfg
        if flipped:
            flip_count += 1
        total_seq_saved += seq_gap
        total_ovl_saved += ovl_gap

        lines.append(f"  {size:<8s} | {CONFIG_SHORT[seq_best_cfg]:>12s} {seq_best_t:>8.3f} {seq_gap:>+5.1f}% | "
                     f"{CONFIG_SHORT[ovl_best_cfg]:>12s} {ovl_best_t:>8.3f} {ovl_gap:>+5.1f}% | "
                     f"{'FLIP' if flipped else '':>5s}")

    lines.append(f"  {'-'*95}")
    lines.append(f"  {'Mean':>8s} | {'':>12s} {'':>8s} {total_seq_saved/len(sizes):>+5.1f}% | "
                 f"{'':>12s} {'':>8s} {total_ovl_saved/len(sizes):>+5.1f}% | "
                 f"{flip_count}/{len(sizes)}")

    # CTA tuning results
    ovl_sweep = channel_data.get("overlap_cta_sweep", {})
    if ovl_sweep:
        lines.append("")
        lines.append("=" * 120)
        lines.append("  TABLE 2: CTA Count Tuning Impact (Overlap Mode)")
        lines.append("=" * 120)
        lines.append("")
        lines.append(f"  {'Size':<8s} | {'Default CTA=8':>14s} {'Optimal CTA':>12s} "
                     f"{'Optimal ms':>11s} {'Improvement':>12s} {'Best Config':>12s}")
        lines.append(f"  {'-'*80}")

        for size in sorted(ovl_sweep.keys(), key=lambda s: SIZE_BYTES.get(s, 0)):
            default_auto = ovl_sweep[size].get("8", {}).get("auto", 0)

            # Find best across all CTAs
            best_t, best_cta, best_cfg = 999, "8", "auto"
            for cta, data in ovl_sweep[size].items():
                for cfg, t in data.items():
                    if t < best_t:
                        best_t, best_cta, best_cfg = t, cta, cfg

            imp = (default_auto - best_t) / default_auto * 100 if default_auto > 0 else 0
            lines.append(f"  {size:<8s} | {default_auto:>14.3f} {best_cta + ' CTAs':>12s} "
                         f"{best_t:>11.3f} {imp:>+11.1f}% {CONFIG_SHORT[best_cfg]:>12s}")

    # CTA × Compute interaction (Block C)
    cta_compute = channel_data.get("cta_compute_interaction", {})
    if cta_compute:
        lines.append("")
        lines.append("=" * 120)
        lines.append("  TABLE 3: CTA × Compute Intensity Interaction (4MB, Overlap)")
        lines.append("=" * 120)
        lines.append("")
        lines.append(f"  {'Intensity':<10s} | {'Opt CTAs':>10s} {'Best Config':>12s} "
                     f"{'Best ms':>10s} {'CTA=8/AUTO':>10s} {'Improvement':>12s}")
        lines.append(f"  {'-'*72}")

        for intensity in ["medium", "default", "heavy"]:
            if intensity not in cta_compute:
                continue
            cta_data = cta_compute[intensity]
            default_auto = cta_data.get("8", {}).get("auto", 0)

            best_t, best_cta, best_cfg = 999, "8", "auto"
            for cta, data in cta_data.items():
                for cfg, t in data.items():
                    if t < best_t:
                        best_t, best_cta, best_cfg = t, cta, cfg

            imp = (default_auto - best_t) / default_auto * 100 if default_auto > 0 else 0
            lines.append(f"  {intensity:<10s} | {best_cta + ' CTAs':>10s} {CONFIG_SHORT.get(best_cfg, best_cfg):>12s} "
                         f"{best_t:>10.3f} {default_auto:>10.3f} {imp:>+11.1f}%")

    text = "\n".join(lines)
    out = OUTPUT_DIR / "paper_table1.txt"
    out.write_text(text)
    print(f"  Saved {out}")
    print(text)


# ===================================================================
# Figure 5: CTA × Compute intensity interaction (Block C)
# ===================================================================
def paper_figure5(channel_data):
    """Block C: how compute intensity changes the optimal CTA count."""
    cta_compute = channel_data.get("cta_compute_interaction", {})
    if not cta_compute:
        print("  Skipping Figure 5 (no Block C data)")
        return

    cta_order = ["1", "2", "4", "8", "12", "16", "24", "32"]
    intensity_labels = {"medium": "Medium (2048²)", "default": "Default (4096²)", "heavy": "Heavy (6144²)"}
    intensity_colors = {"medium": "#3498db", "default": "#2ecc71", "heavy": "#e74c3c"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: AUTO time vs CTA for each compute intensity
    ax1 = axes[0]
    x = list(range(len(cta_order)))
    for intensity in ["medium", "default", "heavy"]:
        if intensity not in cta_compute:
            continue
        vals = [cta_compute[intensity].get(c, {}).get("auto", 0) for c in cta_order]
        ax1.plot(x, vals, "o-", label=intensity_labels.get(intensity, intensity),
                 color=intensity_colors[intensity], linewidth=2.5, markersize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cta_order, fontsize=10)
    ax1.set_xlabel("NCCL_MAX_CTAS", fontsize=12)
    ax1.set_ylabel("Iteration Time (ms)", fontsize=12)
    ax1.set_title("AUTO Time vs CTA Count\nby Compute Intensity (4MB Overlap)",
                   fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Panel 2: Config spread (max-min) at each CTA for each intensity
    ax2 = axes[1]
    width = 0.25
    for i, intensity in enumerate(["medium", "default", "heavy"]):
        if intensity not in cta_compute:
            continue
        spreads = []
        for c in cta_order:
            data = cta_compute[intensity].get(c, {})
            if data:
                vals = list(data.values())
                spreads.append(max(vals) - min(vals))
            else:
                spreads.append(0)
        offset = (i - 1) * width
        ax2.bar([xi + offset for xi in x], spreads, width,
                label=intensity_labels.get(intensity, intensity),
                color=intensity_colors[intensity], alpha=0.8, edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(cta_order, fontsize=10)
    ax2.set_xlabel("NCCL_MAX_CTAS", fontsize=12)
    ax2.set_ylabel("Config Spread (max-min, ms)", fontsize=12)
    ax2.set_title("Protocol Sensitivity vs CTA Count\n(higher = more room to tune)",
                   fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Heatmap — best config at each (intensity, CTA)
    ax3 = axes[2]
    intensities = [i for i in ["medium", "default", "heavy"] if i in cta_compute]
    cfg_to_idx = {c: i for i, c in enumerate(CONFIGS)}

    heatmap = np.zeros((len(intensities), len(cta_order)))
    annotations = []
    for i, intensity in enumerate(intensities):
        row_annots = []
        for j, cta in enumerate(cta_order):
            data = cta_compute[intensity].get(cta, {})
            if data:
                best = min(data, key=data.get)
                heatmap[i, j] = cfg_to_idx.get(best, 0)
                row_annots.append(CONFIG_SHORT.get(best, best))
            else:
                heatmap[i, j] = 0
                row_annots.append("?")
        annotations.append(row_annots)

    colors_list = [CONFIG_COLORS[c] for c in CONFIGS]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(colors_list)
    bounds = [-0.5 + i for i in range(len(CONFIGS) + 1)]
    norm = BoundaryNorm(bounds, cmap.N)

    ax3.imshow(heatmap, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(intensities)):
        for j in range(len(cta_order)):
            ax3.text(j, i, annotations[i][j], ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    ax3.set_xticks(range(len(cta_order)))
    ax3.set_xticklabels(cta_order, fontsize=10)
    ax3.set_yticks(range(len(intensities)))
    ax3.set_yticklabels([intensity_labels.get(i, i) for i in intensities], fontsize=10)
    ax3.set_xlabel("NCCL_MAX_CTAS", fontsize=12)
    ax3.set_title("Best Config by (Intensity, CTA)\n(4MB Overlap)",
                   fontsize=13, fontweight="bold")

    patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=CONFIG_SHORT[c]) for c in CONFIGS]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("CTA × Compute Intensity Interaction: Optimal Config Depends on Both",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "paper_figure5_cta_compute.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ===================================================================
# Main
# ===================================================================
def main():
    print("Loading experimental data...")
    overlap_data, channel_data = load_data()

    print("\nGenerating paper figures...\n")

    paper_figure1(overlap_data)
    paper_figure2(channel_data)
    paper_figure3(overlap_data, channel_data)
    paper_figure4(channel_data)
    paper_figure5(channel_data)
    paper_table1(overlap_data, channel_data)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
