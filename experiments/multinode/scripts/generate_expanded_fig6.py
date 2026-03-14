"""
Generate expanded Figure 6: comprehensive algo×proto comparison for multi-node.

Reads from multinode_expanded_results.json (produced by run_modal_multinode_expanded.py)
and generates a publication-quality figure showing ALL NCCL algo×proto combinations.

Layout: 2-row × 2-col grid:
  Top-left:  Sequential — all configs bar chart (log scale)
  Top-right: Overlap — all configs bar chart (log scale)
  Bottom-left:  AUTO gap per config (sequential)
  Bottom-right: AUTO gap per config (overlap)

Configs tested:
  Ring:              Simple, LL, LL128
  Tree:              Simple, LL, LL128
  CollNet Direct:    Simple
  CollNet Chain:     Simple
  NVLS:              Simple
  NVLS Tree:         Simple
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
EXPANDED_FILE = RESULTS_ROOT / "multinode_expanded" / "multinode_expanded_results.json"
OUTPUT_DIR = RESULTS_ROOT / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MN_SIZES = ["256KB", "1MB", "4MB", "16MB", "64MB", "256MB"]

# ---- Config display names and colors ----
# Group by algorithm family for visual clarity
CONFIG_ORDER = [
    "auto",
    "tree_simple", "tree_ll", "tree_ll128",
    "ring_simple", "ring_ll", "ring_ll128",
    "collnet_direct_simple", "collnet_chain_simple",
    "nvls_simple", "nvls_tree_simple",
    "pat_simple", "pat_ll", "pat_ll128",
]

CONFIG_SHORT = {
    "auto":                  "AUTO",
    "tree_simple":           "Tree+S",
    "tree_ll":               "Tree+LL",
    "tree_ll128":            "Tree+LL128",
    "ring_simple":           "Ring+S",
    "ring_ll":               "Ring+LL",
    "ring_ll128":            "Ring+LL128",
    "collnet_direct_simple": "CNet-D+S",
    "collnet_chain_simple":  "CNet-C+S",
    "nvls_simple":           "NVLS+S",
    "nvls_tree_simple":      "NVLSTree+S",
    "pat_simple":            "PAT+S",
    "pat_ll":                "PAT+LL",
    "pat_ll128":             "PAT+LL128",
}

# Color scheme: group by algorithm family
CONFIG_COLORS = {
    "auto":                  "#555555",   # Grey
    # Tree family — reds
    "tree_simple":           "#e74c3c",   # Red
    "tree_ll":               "#c0392b",   # Dark red
    "tree_ll128":            "#f1948a",   # Light red
    # Ring family — greens
    "ring_simple":           "#2ecc71",   # Green
    "ring_ll":               "#27ae60",   # Dark green
    "ring_ll128":            "#82e0aa",   # Light green
    # CollNet family — blues
    "collnet_direct_simple": "#3498db",   # Blue
    "collnet_chain_simple":  "#2980b9",   # Dark blue
    # NVLS family — purples/oranges
    "nvls_simple":           "#9b59b6",   # Purple
    "nvls_tree_simple":      "#f39c12",   # Orange
    # PAT family — teals
    "pat_simple":            "#1abc9c",   # Teal
    "pat_ll":                "#16a085",   # Dark teal
    "pat_ll128":             "#76d7c4",   # Light teal
}


def load_expanded_data():
    """Load expanded multinode results."""
    if not EXPANDED_FILE.exists():
        print(f"ERROR: {EXPANDED_FILE} not found.")
        print(f"Run the expanded experiment first:")
        print(f"  modal run run_modal_multinode_expanded.py")
        sys.exit(1)
    return json.loads(EXPANDED_FILE.read_text())


def get_available_configs(data):
    """Determine which configs actually have data (non-zero)."""
    available = set()
    for mode in ("sequential", "overlap"):
        for size in MN_SIZES:
            if size in data.get(mode, {}):
                for cfg, val in data[mode][size].items():
                    if val > 0:
                        available.add(cfg)
    # Preserve order from CONFIG_ORDER
    return [c for c in CONFIG_ORDER if c in available]


def expanded_figure6(data):
    """
    Comprehensive algo×proto figure for multi-node.

    4-panel layout:
      Top row:    Bar charts (sequential, overlap) — all configs
      Bottom row: AUTO gap bar charts per config
    """
    seq = data.get("sequential", {})
    ovl = data.get("overlap", {})
    if not seq or not ovl:
        print("  No sequential or overlap data found.")
        return

    sizes = [s for s in MN_SIZES if s in seq and s in ovl]
    configs = get_available_configs(data)

    n_configs = len(configs)
    print(f"  Available configs: {n_configs}")
    for c in configs:
        print(f"    {CONFIG_SHORT.get(c, c)}")

    fig, axes = plt.subplots(2, 2, figsize=(24, 14))

    # ---- Top Left: Sequential bars ----
    ax = axes[0, 0]
    x = np.arange(len(sizes))
    width = 0.8 / n_configs
    for i, cfg in enumerate(configs):
        vals = [seq[s].get(cfg, 0) for s in sizes]
        # Replace 0s with NaN so they don't show up
        vals = [v if v > 0 else float('nan') for v in vals]
        offset = (i - (n_configs - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=CONFIG_SHORT[cfg],
               color=CONFIG_COLORS.get(cfg, "#aaaaaa"), edgecolor="white",
               linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=11)
    ax.set_xlabel("Message Size", fontsize=13)
    ax.set_ylabel("Iteration Time (ms)", fontsize=13)
    ax.set_title("Multi-Node Sequential", fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    # ---- Top Right: Overlap bars ----
    ax = axes[0, 1]
    for i, cfg in enumerate(configs):
        vals = [ovl[s].get(cfg, 0) for s in sizes]
        vals = [v if v > 0 else float('nan') for v in vals]
        offset = (i - (n_configs - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=CONFIG_SHORT[cfg],
               color=CONFIG_COLORS.get(cfg, "#aaaaaa"), edgecolor="white",
               linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=11)
    ax.set_xlabel("Message Size", fontsize=13)
    ax.set_ylabel("Iteration Time (ms)", fontsize=13)
    ax.set_title("Multi-Node Overlap", fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    # ---- Bottom Left: AUTO gap per config (Sequential) ----
    ax = axes[1, 0]
    non_auto_cfgs = [c for c in configs if c != "auto"]
    x2 = np.arange(len(sizes))
    width2 = 0.8 / len(non_auto_cfgs)
    for i, cfg in enumerate(non_auto_cfgs):
        gaps = []
        for s in sizes:
            auto_t = seq[s].get("auto", 0)
            cfg_t = seq[s].get(cfg, 0)
            if auto_t > 0 and cfg_t > 0:
                # Positive = config is FASTER than AUTO (good)
                gap = (auto_t - cfg_t) / auto_t * 100
            else:
                gap = float('nan')
            gaps.append(gap)
        offset = (i - (len(non_auto_cfgs) - 1) / 2) * width2
        bars = ax.bar(x2 + offset, gaps, width2, label=CONFIG_SHORT[cfg],
                      color=CONFIG_COLORS.get(cfg, "#aaaaaa"), edgecolor="white",
                      linewidth=0.3)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x2)
    ax.set_xticklabels(sizes, fontsize=11)
    ax.set_xlabel("Message Size", fontsize=13)
    ax.set_ylabel("Improvement over AUTO (%)", fontsize=13)
    ax.set_title("Sequential: Config vs AUTO\n(positive = faster than AUTO)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ---- Bottom Right: AUTO gap per config (Overlap) ----
    ax = axes[1, 1]
    for i, cfg in enumerate(non_auto_cfgs):
        gaps = []
        for s in sizes:
            auto_t = ovl[s].get("auto", 0)
            cfg_t = ovl[s].get(cfg, 0)
            if auto_t > 0 and cfg_t > 0:
                gap = (auto_t - cfg_t) / auto_t * 100
            else:
                gap = float('nan')
            gaps.append(gap)
        offset = (i - (len(non_auto_cfgs) - 1) / 2) * width2
        bars = ax.bar(x2 + offset, gaps, width2, label=CONFIG_SHORT[cfg],
                      color=CONFIG_COLORS.get(cfg, "#aaaaaa"), edgecolor="white",
                      linewidth=0.3)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x2)
    ax.set_xticklabels(sizes, fontsize=11)
    ax.set_xlabel("Message Size", fontsize=13)
    ax.set_ylabel("Improvement over AUTO (%)", fontsize=13)
    ax.set_title("Overlap: Config vs AUTO\n(positive = faster than AUTO)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Multi-Node AllReduce: Comprehensive Algorithm × Protocol Comparison\n"
        f"(2 nodes × 4 A100 GPUs, {len(configs)} configs, inter-node network)",
        fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "fig6_expanded_multinode.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def summary_table(data):
    """Print a summary table of all configs × sizes × modes."""
    seq = data.get("sequential", {})
    ovl = data.get("overlap", {})
    configs = get_available_configs(data)
    sizes = [s for s in MN_SIZES if s in seq]

    print(f"\n{'='*120}")
    print(f"  EXPANDED MULTI-NODE RESULTS SUMMARY")
    print(f"{'='*120}")

    for mode_label, mode_data in [("SEQUENTIAL", seq), ("OVERLAP", ovl)]:
        print(f"\n  {mode_label}:")
        header = f"  {'Size':<8s}"
        for cfg in configs:
            header += f"  {CONFIG_SHORT.get(cfg, cfg):>12s}"
        header += f"  {'Best':>15s}  {'AUTO Gap':>10s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for s in sizes:
            if s not in mode_data:
                continue
            row = f"  {s:<8s}"
            valid = {}
            for cfg in configs:
                val = mode_data[s].get(cfg, 0)
                if val > 0:
                    row += f"  {val:>12.3f}"
                    valid[cfg] = val
                else:
                    row += f"  {'N/A':>12s}"

            if valid:
                best_cfg = min(valid, key=valid.get)
                best_t = valid[best_cfg]
                auto_t = valid.get("auto", 0)
                gap = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0
                row += f"  {CONFIG_SHORT.get(best_cfg, best_cfg):>15s}  {gap:>+9.1f}%"
            print(row)

    print()


def main():
    print("Loading expanded multi-node data...")
    data = load_expanded_data()
    print("Generating expanded Figure 6...")
    expanded_figure6(data)
    summary_table(data)
    print("Done!")


if __name__ == "__main__":
    main()
