"""
Topology-Aware Workload Tuner: Generate per-topology policies and evaluate.

This is the core contribution of the paper. Using node sweep profiling data,
we build a tuner that:
  1. Detects cluster topology (nodes × GPUs/node)
  2. Looks up the profiled-optimal NCCL config for each (size, mode)
  3. Shows the achievable speedup vs NCCL AUTO

Inputs:
  - results/node_sweep/{1x8,2x4,4x2}/results.json
  - results/node_sweep/{1x8,2x4,4x2}/auto_selections.json
  - Raw timing files for statistical analysis

Outputs:
  - results/tuner_topology_policy.json       — per-topology policy table
  - results/tuner_topology_policy_table.h    — C header for plugin
  - results/paper_figures/fig_tuner_evaluation.png  — evaluation figure
  - results/paper_figures/fig_tuner_recommendation_table.png — visual table
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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
    "256KB": 256*1024, "1MB": 1024**2, "4MB": 4*1024**2,
    "16MB": 16*1024**2, "64MB": 64*1024**2, "256MB": 256*1024**2,
}
ALGOS = ["auto", "tree_simple", "tree_ll", "tree_ll128",
         "ring_simple", "ring_ll", "ring_ll128"]
ALGO_LABELS = {
    "tree_simple": "Tree+Simple", "tree_ll": "Tree+LL",
    "tree_ll128": "Tree+LL128", "ring_simple": "Ring+Simple",
    "ring_ll": "Ring+LL", "ring_ll128": "Ring+LL128",
}
MODES = ["sequential", "overlap"]

# NCCL algo/proto IDs
ALGO_IDS = {"tree": 0, "ring": 1}
PROTO_IDS = {"ll": 0, "ll128": 1, "simple": 2}


def config_to_ids(config_name):
    """Convert config name like 'tree_simple' to (algo_id, proto_id)."""
    parts = config_name.split("_", 1)
    algo = ALGO_IDS.get(parts[0], -1)
    proto = PROTO_IDS.get(parts[1], -1) if len(parts) > 1 else -1
    return algo, proto


def load_data():
    """Load results for all topologies."""
    data = {}
    for cfg in CONFIGS:
        f = NODE_SWEEP_DIR / cfg / "results.json"
        if f.is_file():
            data[cfg] = json.loads(f.read_text())
    return data


def load_raw_timings():
    """Load raw timing files for variance analysis."""
    raw = {}
    for cfg in CONFIGS:
        raw[cfg] = {}
        for mode in MODES:
            raw[cfg][mode] = {}
            for size in SIZES:
                raw[cfg][mode][size] = {}
                for algo in ALGOS:
                    times = _find_timing(cfg, mode, size, algo)
                    if times:
                        raw[cfg][mode][size][algo] = times
    return raw


def _find_timing(cfg, mode, size, algo):
    patterns = [
        NODE_SWEEP_DIR / cfg / f"times_ns_{cfg}_{mode}_{size}_{algo}.txt",
        SC_DIR / f"times_sc_{mode}_{size}_{algo}.txt",
    ]
    for p in patterns:
        if p.is_file():
            try:
                text = p.read_text().strip()
                if not text:
                    continue
                return [float(x) for x in text.split("\n") if x.strip()]
            except (ValueError, IOError):
                continue
    return None


# ─────────────────────────────────────────────────────────────────
# Phase 1: Build per-topology policy from profiling data
# ─────────────────────────────────────────────────────────────────
def build_policies(data):
    """Build the tuner's recommendation table for each topology."""
    policies = {}

    for topo in CONFIGS:
        if topo not in data:
            continue
        policies[topo] = {"sequential": {}, "overlap": {}}

        for mode in MODES:
            for size in SIZES:
                sd = data[topo].get(mode, {}).get(size, {})
                auto_t = sd.get("auto", 0)

                # Find best non-auto config
                best_algo = None
                best_t = float("inf")
                for algo in ALGOS:
                    if algo == "auto":
                        continue
                    t = sd.get(algo, float("inf"))
                    if isinstance(t, (int, float)) and 0 < t < best_t:
                        best_t = t
                        best_algo = algo

                if best_algo and auto_t > 0:
                    gap_pct = (auto_t - best_t) / best_t * 100
                    algo_id, proto_id = config_to_ids(best_algo)

                    policies[topo][mode][size] = {
                        "recommended": best_algo,
                        "recommended_label": ALGO_LABELS.get(best_algo, best_algo),
                        "recommended_time_ms": round(best_t, 3),
                        "auto_time_ms": round(auto_t, 3),
                        "speedup_pct": round(max(gap_pct, 0), 1),
                        "algo_id": algo_id,
                        "proto_id": proto_id,
                    }

    return policies


# ─────────────────────────────────────────────────────────────────
# Phase 2: Generate C header for topology-aware plugin
# ─────────────────────────────────────────────────────────────────
def generate_c_header(policies):
    """Generate C header with per-topology lookup tables."""
    lines = [
        "/* Auto-generated topology-aware tuner policy table */",
        "/* Generated from node sweep profiling data */",
        "",
        "#ifndef TOPOLOGY_AWARE_POLICY_H",
        "#define TOPOLOGY_AWARE_POLICY_H",
        "",
        "typedef struct {",
        "    int algo;    /* NCCL_ALGO_TREE=0, NCCL_ALGO_RING=1 */",
        "    int proto;   /* NCCL_PROTO_LL=0, NCCL_PROTO_LL128=1, NCCL_PROTO_SIMPLE=2 */",
        "} PolicyEntry;",
        "",
        "/* Size bands: 0=256KB, 1=1MB, 2=4MB, 3=16MB, 4=64MB, 5=256MB */",
        "/* Modes: 0=sequential, 1=overlap */",
        "",
    ]

    topo_map = {"1x8": "1_NODE", "2x4": "2_NODE", "4x2": "4_NODE"}

    for topo in CONFIGS:
        if topo not in policies:
            continue
        name = topo_map.get(topo, topo.upper())
        lines.append(f"/* Topology: {topo} ({name}) */")
        lines.append(f"static const PolicyEntry policy_{name}[2][6] = {{")

        for mode_idx, mode in enumerate(MODES):
            entries = []
            for size in SIZES:
                p = policies[topo][mode].get(size, {})
                algo = p.get("algo_id", -1)
                proto = p.get("proto_id", -1)
                rec = p.get("recommended", "auto")
                entries.append(f"    {{{algo}, {proto}}}  /* {size}: {rec} */")

            lines.append(f"    {{ /* {mode} */")
            lines.append(",\n".join(f"        {e.strip()}" for e in entries))
            lines.append("    },")

        lines.append("};")
        lines.append("")

    lines.append("#endif /* TOPOLOGY_AWARE_POLICY_H */")

    out = RESULTS_DIR / "tuner_topology_policy_table.h"
    out.write_text("\n".join(lines))
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Figure: Tuner Recommendation Table (visual)
# ─────────────────────────────────────────────────────────────────
def fig_recommendation_table(policies):
    """Visual table: recommended config per (topology, size, mode)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Topology-Aware Tuner: Recommended Configurations",
                 fontsize=15, fontweight="bold", y=1.02)

    # Color each algo family
    algo_color = {
        "tree_simple": "#ef4444", "tree_ll": "#f97316", "tree_ll128": "#fbbf24",
        "ring_simple": "#22c55e", "ring_ll": "#14b8a6", "ring_ll128": "#06b6d4",
    }

    for ax_idx, mode in enumerate(MODES):
        ax = axes[ax_idx]
        ax.set_xlim(-0.5, len(SIZES) - 0.5)
        ax.set_ylim(-0.5, len(CONFIGS) - 0.5)

        for i, topo in enumerate(CONFIGS):
            for j, size in enumerate(SIZES):
                p = policies.get(topo, {}).get(mode, {}).get(size, {})
                rec = p.get("recommended", "auto")
                label = p.get("recommended_label", "AUTO")
                speedup = p.get("speedup_pct", 0)
                color = algo_color.get(rec, "#6b7280")

                # Draw colored cell
                rect = plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                     facecolor=color, alpha=0.7, edgecolor="white",
                                     linewidth=2)
                ax.add_patch(rect)

                # Label: algo name
                ax.text(j, i + 0.1, label, ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
                # Speedup annotation
                if speedup > 0:
                    ax.text(j, i - 0.2, f"+{speedup:.0f}%", ha="center",
                            va="center", fontsize=7, color="white", alpha=0.9)

        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels(SIZES, fontsize=9)
        ax.set_yticks(range(len(CONFIGS)))
        ax.set_yticklabels(["1×8\n(1 node)", "2×4\n(2 nodes)", "4×2\n(4 nodes)"],
                           fontsize=10)
        ax.set_xlabel("Message Size", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Topology", fontsize=11)
        mode_label = "Sequential" if mode == "sequential" else "Overlap"
        ax.set_title(f"{mode_label} Mode", fontsize=13, fontweight="bold")
        ax.invert_yaxis()

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#ef4444", alpha=0.7, label="Tree+Simple"),
        mpatches.Patch(facecolor="#f97316", alpha=0.7, label="Tree+LL"),
        mpatches.Patch(facecolor="#fbbf24", alpha=0.7, label="Tree+LL128"),
        mpatches.Patch(facecolor="#22c55e", alpha=0.7, label="Ring+Simple"),
        mpatches.Patch(facecolor="#14b8a6", alpha=0.7, label="Ring+LL"),
        mpatches.Patch(facecolor="#06b6d4", alpha=0.7, label="Ring+LL128"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    out = FIGURES_DIR / "fig_tuner_recommendation_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Figure: Tuner Evaluation (speedup vs AUTO)
# ─────────────────────────────────────────────────────────────────
def fig_tuner_evaluation(policies, data):
    """Main evaluation figure: speedup from using tuner vs AUTO."""
    fig = plt.figure(figsize=(18, 10))

    # Layout: top row = 2 bar charts (seq + overlap speedup by topology)
    #         bottom row = summary stats + latency comparison
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # ── Panel A: Speedup bars (sequential) ──
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_speedup_bars(ax1, policies, "sequential", "A) Sequential Speedup")

    # ── Panel B: Speedup bars (overlap) ──
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_speedup_bars(ax2, policies, "overlap", "B) Overlap Speedup")

    # ── Panel C: Summary statistics ──
    ax3 = fig.add_subplot(gs[0, 2])
    _draw_summary_stats(ax3, policies)

    # ── Panel D: Latency comparison at 64MB ──
    ax4 = fig.add_subplot(gs[1, 0:2])
    _draw_latency_comparison(ax4, data, policies)

    # ── Panel E: Cumulative improvement ──
    ax5 = fig.add_subplot(gs[1, 2])
    _draw_cumulative(ax5, policies)

    fig.suptitle("Topology-Aware Tuner Evaluation: Profiled-Optimal vs NCCL AUTO",
                 fontsize=16, fontweight="bold", y=1.01)

    out = FIGURES_DIR / "fig_tuner_evaluation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def _draw_speedup_bars(ax, policies, mode, title):
    """Draw grouped bar chart of speedup % for each topology × size."""
    x = np.arange(len(SIZES))
    width = 0.25
    colors = {"1x8": "#3b82f6", "2x4": "#f59e0b", "4x2": "#ef4444"}

    for i, topo in enumerate(CONFIGS):
        speedups = []
        for size in SIZES:
            p = policies.get(topo, {}).get(mode, {}).get(size, {})
            speedups.append(p.get("speedup_pct", 0))

        offset = (i - 1) * width
        bars = ax.bar(x + offset, speedups, width, color=colors[topo],
                      label=topo, alpha=0.8, edgecolor="white")

        # Annotate high speedups
        for j, s in enumerate(speedups):
            if s > 20:
                ax.text(j + offset, s + 1, f"{s:.0f}%", ha="center",
                        va="bottom", fontsize=7, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SIZES, fontsize=9, rotation=45)
    ax.set_ylabel("Speedup over AUTO (%)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5)


def _draw_summary_stats(ax, policies):
    """Draw summary statistics table."""
    ax.axis("off")
    ax.set_title("C) Summary", fontsize=12, fontweight="bold")

    rows = []
    for topo in CONFIGS:
        all_speedups = []
        for mode in MODES:
            for size in SIZES:
                p = policies.get(topo, {}).get(mode, {}).get(size, {})
                s = p.get("speedup_pct", 0)
                if s > 0:
                    all_speedups.append(s)

        if all_speedups:
            avg = np.mean(all_speedups)
            med = np.median(all_speedups)
            mx = max(all_speedups)
            n_improved = len(all_speedups)
            total = len(MODES) * len(SIZES)
            rows.append([topo, f"{avg:.1f}%", f"{med:.1f}%", f"{mx:.0f}%",
                         f"{n_improved}/{total}"])

    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=["Topology", "Mean\nSpeedup", "Median\nSpeedup",
                        "Max\nSpeedup", "Configs\nImproved"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Color the cells
        for i in range(len(rows)):
            for j in range(5):
                cell = table[i + 1, j]
                if j >= 1:  # speedup columns
                    val = float(rows[i][j].replace("%", "").split("/")[0])
                    if val > 20:
                        cell.set_facecolor("#fee2e2")
                    elif val > 10:
                        cell.set_facecolor("#fef3c7")
                    else:
                        cell.set_facecolor("#dcfce7")


def _draw_latency_comparison(ax, data, policies):
    """Compare AUTO vs tuner latency at key sizes."""
    ax.set_title("D) Latency: AUTO vs Tuner (Sequential Mode)", fontsize=12,
                 fontweight="bold")

    key_sizes = ["16MB", "64MB", "256MB"]
    n_groups = len(key_sizes) * len(CONFIGS)
    x = np.arange(n_groups)
    width = 0.35

    auto_vals = []
    tuner_vals = []
    labels = []

    for size in key_sizes:
        for topo in CONFIGS:
            sd = data.get(topo, {}).get("sequential", {}).get(size, {})
            p = policies.get(topo, {}).get("sequential", {}).get(size, {})

            auto_vals.append(sd.get("auto", 0))
            tuner_vals.append(p.get("recommended_time_ms", 0))
            labels.append(f"{topo}\n{size}")

    ax.bar(x - width/2, auto_vals, width, color="#6b7280", alpha=0.8,
           label="NCCL AUTO", edgecolor="white")
    ax.bar(x + width/2, tuner_vals, width, color="#22c55e", alpha=0.8,
           label="Tuner", edgecolor="white")

    # Annotate speedup
    for i in range(n_groups):
        if auto_vals[i] > 0 and tuner_vals[i] > 0:
            speedup = (auto_vals[i] - tuner_vals[i]) / tuner_vals[i] * 100
            if speedup > 5:
                y = max(auto_vals[i], tuner_vals[i])
                ax.text(i, y * 1.02, f"-{speedup:.0f}%", ha="center",
                        va="bottom", fontsize=8, fontweight="bold",
                        color="#ef4444")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Median Latency (ms)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Add vertical separators between size groups
    for i in range(1, len(key_sizes)):
        ax.axvline(x=i * len(CONFIGS) - 0.5, color="gray", linestyle="--",
                    alpha=0.3)


def _draw_cumulative(ax, policies):
    """Show cumulative % of conditions where tuner helps, by threshold."""
    ax.set_title("E) Tuner Impact Distribution", fontsize=12, fontweight="bold")

    all_speedups = []
    for topo in CONFIGS:
        for mode in MODES:
            for size in SIZES:
                p = policies.get(topo, {}).get(mode, {}).get(size, {})
                all_speedups.append(p.get("speedup_pct", 0))

    all_speedups.sort()
    n = len(all_speedups)
    thresholds = np.arange(0, 75, 1)
    pcts_above = [(np.sum(np.array(all_speedups) >= t) / n * 100)
                  for t in thresholds]

    ax.fill_between(thresholds, pcts_above, alpha=0.3, color="#3b82f6")
    ax.plot(thresholds, pcts_above, color="#3b82f6", linewidth=2)

    # Mark key thresholds
    for thresh, label in [(10, "10%"), (20, "20%"), (40, "40%")]:
        pct = np.sum(np.array(all_speedups) >= thresh) / n * 100
        ax.axvline(x=thresh, color="gray", linestyle=":", alpha=0.5)
        ax.plot(thresh, pct, "ro", markersize=6)
        ax.text(thresh + 1, pct + 2, f"{pct:.0f}% of configs\n≥{label} faster",
                fontsize=8, va="bottom")

    ax.set_xlabel("Speedup Threshold (%)", fontsize=10)
    ax.set_ylabel("% of Conditions Above Threshold", fontsize=10)
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} topologies: {list(data.keys())}")

    print("\nPhase 1: Building per-topology policies...")
    policies = build_policies(data)

    # Print policy summary
    for topo in CONFIGS:
        if topo not in policies:
            continue
        print(f"\n  {topo}:")
        for mode in MODES:
            print(f"    {mode}:")
            for size in SIZES:
                p = policies[topo][mode].get(size, {})
                rec = p.get("recommended_label", "?")
                speedup = p.get("speedup_pct", 0)
                auto_t = p.get("auto_time_ms", 0)
                rec_t = p.get("recommended_time_ms", 0)
                print(f"      {size:>6s}: {rec:<14s} ({rec_t:>8.1f}ms vs AUTO {auto_t:>8.1f}ms, +{speedup:.1f}%)")

    print("\nPhase 2: Generating C header...")
    generate_c_header(policies)

    print("\nPhase 3: Generating policy JSON...")
    policy_json = {
        "description": "Topology-aware workload tuner policy",
        "platform": "A100 40GB, AllReduce",
        "topologies": {}
    }
    for topo in CONFIGS:
        if topo in policies:
            policy_json["topologies"][topo] = policies[topo]
    out = RESULTS_DIR / "tuner_topology_policy.json"
    out.write_text(json.dumps(policy_json, indent=2))
    print(f"Saved: {out}")

    print("\nPhase 4: Generating figures...")
    fig_recommendation_table(policies)
    fig_tuner_evaluation(policies, data)

    # Overall summary
    print("\n" + "=" * 70)
    print("TUNER EVALUATION SUMMARY")
    print("=" * 70)
    for topo in CONFIGS:
        all_sp = []
        for mode in MODES:
            for size in SIZES:
                p = policies.get(topo, {}).get(mode, {}).get(size, {})
                sp = p.get("speedup_pct", 0)
                all_sp.append(sp)
        print(f"  {topo}: avg={np.mean(all_sp):.1f}%, max={max(all_sp):.0f}%, "
              f"median={np.median(all_sp):.1f}%")

    all_sp = []
    for topo in CONFIGS:
        for mode in MODES:
            for size in SIZES:
                p = policies.get(topo, {}).get(mode, {}).get(size, {})
                all_sp.append(p.get("speedup_pct", 0))
    print(f"\n  Overall: avg={np.mean(all_sp):.1f}%, max={max(all_sp):.0f}%, "
          f"median={np.median(all_sp):.1f}%")
    n_improved = sum(1 for s in all_sp if s > 5)
    print(f"  Conditions with >5% improvement: {n_improved}/{len(all_sp)} "
          f"({n_improved/len(all_sp)*100:.0f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
