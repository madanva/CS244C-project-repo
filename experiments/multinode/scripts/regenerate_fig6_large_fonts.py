#!/usr/bin/env python3
"""Regenerate fig6_multinode with larger fonts for SIGCOMM column-width readability."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Data ---
DATA_FILE = Path(__file__).parent / "results" / "multinode_experiment" / "multinode_results.json"
OUTPUT = Path(__file__).parent / "../../paper/figures/fig6_multinode.png"

with open(DATA_FILE) as f:
    multinode_data = json.load(f)

seq = multinode_data["sequential"]
ovl = multinode_data["overlap"]

CONFIGS = ["auto", "tree_simple", "tree_ll128", "ring_simple", "ring_ll128"]
CONFIG_SHORT = {
    "auto": "AUTO", "tree_simple": "Tree+S", "tree_ll128": "Tree+L",
    "ring_simple": "Ring+S", "ring_ll128": "Ring+L",
}
CONFIG_COLORS = {
    "auto": "#555555", "tree_simple": "#e74c3c", "tree_ll128": "#3498db",
    "ring_simple": "#2ecc71", "ring_ll128": "#f39c12",
}

mn_sizes = ["256KB", "1MB", "4MB", "16MB", "64MB", "256MB"]
sizes = [s for s in mn_sizes if s in seq and s in ovl]
x = np.arange(len(sizes))

# --- Figure: give right panel more width via gridspec ---
fig, axes = plt.subplots(1, 3, figsize=(22, 8),
                         gridspec_kw={"width_ratios": [1, 1, 1.15]})

# Font sizes optimized for ~3.33in column width
TICK_SIZE = 15
LABEL_SIZE = 17
TITLE_SIZE = 18
LEGEND_SIZE = 12
ANNOT_SIZE = 12
SUPTITLE_SIZE = 20

width = 0.15

# Panel 1: Sequential
ax1 = axes[0]
for i, cfg in enumerate(CONFIGS):
    vals = [seq[s].get(cfg, 0) for s in sizes]
    ax1.bar(x + (i - 2) * width, vals, width, label=CONFIG_SHORT[cfg],
            color=CONFIG_COLORS[cfg], edgecolor="white")
ax1.set_xticks(x)
ax1.set_xticklabels(sizes, fontsize=TICK_SIZE)
ax1.set_xlabel("Message Size", fontsize=LABEL_SIZE)
ax1.set_ylabel("Iteration Time (ms)", fontsize=LABEL_SIZE)
ax1.set_title("Multi-Node Sequential", fontsize=TITLE_SIZE, fontweight="bold")
ax1.legend(fontsize=LEGEND_SIZE, loc="upper left")
ax1.grid(axis="y", alpha=0.3)
ax1.set_yscale("log")
ax1.tick_params(axis='y', labelsize=TICK_SIZE)

# Panel 2: Overlap
ax2 = axes[1]
for i, cfg in enumerate(CONFIGS):
    vals = [ovl[s].get(cfg, 0) for s in sizes]
    ax2.bar(x + (i - 2) * width, vals, width, label=CONFIG_SHORT[cfg],
            color=CONFIG_COLORS[cfg], edgecolor="white")
ax2.set_xticks(x)
ax2.set_xticklabels(sizes, fontsize=TICK_SIZE)
ax2.set_xlabel("Message Size", fontsize=LABEL_SIZE)
ax2.set_ylabel("Iteration Time (ms)", fontsize=LABEL_SIZE)
ax2.set_title("Multi-Node Overlap", fontsize=TITLE_SIZE, fontweight="bold")
ax2.legend(fontsize=LEGEND_SIZE, loc="upper left")
ax2.grid(axis="y", alpha=0.3)
ax2.set_yscale("log")
ax2.tick_params(axis='y', labelsize=TICK_SIZE)

# Panel 3: AUTO gap — wider, with staggered annotations
ax3 = axes[2]
seq_gaps, ovl_gaps = [], []
for s in sizes:
    sa = seq[s].get("auto", 0)
    sb = min(seq[s].values())
    seq_gaps.append((sa - sb) / sa * 100 if sa > 0 else 0)
    oa = ovl[s].get("auto", 0)
    ob = min(ovl[s].values())
    ovl_gaps.append((oa - ob) / oa * 100 if oa > 0 else 0)

bar_w = 0.35
bars1 = ax3.bar(x - bar_w/2, seq_gaps, bar_w, label="Sequential", color="#3498db", alpha=0.85)
bars2 = ax3.bar(x + bar_w/2, ovl_gaps, bar_w, label="Overlap", color="#e74c3c", alpha=0.85)

# Annotations: nudge blue (seq) slightly left, red (ovl) slightly right
NUDGE_RED = 0.06
NUDGE_BLUE = 0.03  # half the red nudge, opposite direction
for bar, gap in zip(bars1, seq_gaps):
    if gap > 1:
        ax3.text(bar.get_x() + bar.get_width()/2 - NUDGE_BLUE, bar.get_height() + 1.0,
                 f"{gap:.0f}%", ha="center", va="bottom", fontsize=ANNOT_SIZE,
                 fontweight="bold", color="#2471a3")
for bar, gap in zip(bars2, ovl_gaps):
    if gap > 1:
        ax3.text(bar.get_x() + bar.get_width()/2 + NUDGE_RED, bar.get_height() + 1.0,
                 f"{gap:.0f}%", ha="center", va="bottom", fontsize=ANNOT_SIZE,
                 fontweight="bold", color="#c0392b")

ax3.set_xticks(x)
ax3.set_xticklabels(sizes, fontsize=TICK_SIZE, rotation=45, ha="right")
ax3.set_xlabel("Message Size", fontsize=LABEL_SIZE)
ax3.set_ylabel("AUTO Gap (%)", fontsize=LABEL_SIZE)
ax3.set_title("AUTO Suboptimality\n(Multi-Node)", fontsize=TITLE_SIZE, fontweight="bold")
ax3.legend(fontsize=LEGEND_SIZE)
ax3.grid(axis="y", alpha=0.3)
ax3.tick_params(axis='y', labelsize=TICK_SIZE)
# Add headroom for annotations above 57%
ax3.set_ylim(top=max(max(seq_gaps), max(ovl_gaps)) * 1.15)

plt.suptitle("Multi-Node AllReduce: NCCL AUTO Leaves Up to 57% on the Table\n"
             "(2 nodes \u00d7 4 A100 GPUs, inter-node network)",
             fontsize=SUPTITLE_SIZE, fontweight="bold", y=1.02)
plt.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=250, bbox_inches="tight")
plt.close()
print(f"Saved {OUTPUT} (optimized for column-width readability)")
