"""
Generate a high-level figure illustrating the Profile-Guided Tuning workflow
as it would be used in practice for LLM training.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "results" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, color, fontsize=11, text_color="white",
        bold=True, edgecolor=None, alpha=0.95):
    """Rounded box with centered text."""
    ec = edgecolor or color
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.12", linewidth=2,
                        edgecolor=ec, facecolor=color, alpha=alpha, zorder=2)
    ax.add_patch(p)
    wt = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=wt, color=text_color, zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#444", lw=2.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=1)


def code_box(ax, x, y, text, fontsize=8, bg="#f8f9fa", border="#ccc"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontfamily="monospace", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=bg,
                      edgecolor=border, linewidth=1.2), zorder=3)


def generate_figure():
    fig, ax = plt.subplots(1, 1, figsize=(26, 16))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2.5, 11)
    ax.axis("off")

    # =================================================================
    # Title
    # =================================================================
    ax.text(7, 10.5, "Profile-Guided NCCL Tuning for LLM Training",
            ha="center", fontsize=26, fontweight="bold", color="#1a1a2e")
#     ax.text(7, 9.9,
#             "A quick profiling phase adapts collective tuning to each "
#             "cluster before training begins",
#             ha="center", fontsize=13, color="#666", style="italic")

    # =================================================================
    # STEP 1 — Cluster Assigned  (left column)
    # =================================================================
    box(ax, 1.5, 8.3, 2.4, 1.0, "1  Cluster Assigned", "#34495e",
        fontsize=13)

    code_box(ax, 1.5, 7.2,
             "2 nodes x 4 A100 GPUs\n"
             "InfiniBand interconnect\n"
             "Unique topology & routing",
             fontsize=9, bg="#eef2f7", border="#34495e")

    arrow(ax, 2.8, 8.3, 4.0, 8.3, color="#34495e")

    # =================================================================
    # STEP 2 — Profile Cluster  (center-left)
    # =================================================================
    # Background
    prof_bg = FancyBboxPatch((3.8, 5.4), 5.0, 4.6,
                              boxstyle="round,pad=0.2", linewidth=2,
                              edgecolor="#2980b9", facecolor="#ebf5fb",
                              alpha=0.35, zorder=0)
    ax.add_patch(prof_bg)

    box(ax, 6.3, 9.5, 2.8, 0.7, "2  Profile Cluster", "#2980b9",
        fontsize=13)

    ax.text(6.3, 8.85, "~2 min overhead  |  runs once per cluster",
            ha="center", fontsize=10, color="#2980b9", fontweight="bold")

    ax.text(4.3, 8.2, "For each message size, benchmark all configs:",
            ha="left", fontsize=10, color="#2c3e50", fontweight="bold")

    # Config results
    configs = [
        ("Tree + Simple", "11.0ms", True),
        ("Tree + LL128",  "12.4ms", False),
        ("Ring + Simple", "12.7ms", False),
        ("Ring + LL128",  "11.8ms", False),
        ("Tree + LL",     "13.5ms", False),
        ("Ring + LL",     "16.5ms", False),
    ]
    for i, (name, time, is_best) in enumerate(configs):
        yy = 7.7 - i * 0.38
        color = "#27ae60" if is_best else "#555"
        weight = "bold" if is_best else "normal"
        suffix = "  \u2190 BEST" if is_best else ""
        ax.text(4.5, yy, f"{name:<18s}{time}{suffix}",
                fontsize=9.5, fontfamily="monospace", color=color,
                fontweight=weight, zorder=3)

    # Size sweep
    ax.text(6.3, 5.5,
            "Sweep:  256KB   1MB   4MB   16MB   64MB   256MB",
            ha="center", fontsize=9, fontfamily="monospace", color="#7f8c8d",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#bdc3c7"), zorder=3)

    # Arrow to step 3
    arrow(ax, 8.9, 8.3, 10.0, 8.3, color="#2980b9", lw=3)
    ax.text(9.45, 8.65, "measured\noptimal", ha="center", fontsize=9,
            color="#2c3e50", fontweight="bold")

    # =================================================================
    # STEP 3 — Generate Policy  (right column)
    # =================================================================
    box(ax, 11.5, 9.0, 2.8, 0.7, "3  Generate Policy", "#e67e22",
        fontsize=13)

    code_box(ax, 11.5, 7.9,
             "Band 0-1 (<1MB):     Ring+Simple\n"
             "Band 2-4 (1-32MB):   Tree+Simple\n"
             "Band 5   (32-128MB): Tree+Simple\n"
             "Band 6   (>128MB):   Tree+LL128",
             fontsize=8.5, bg="#fef9e7", border="#e67e22")

    code_box(ax, 11.5, 6.7,
             'NCCL_TUNER_POLICY=\n"0:1:2,1:1:2,5:0:2,6:0:1"',
             fontsize=8.5, bg="#fdebd0", border="#e67e22")

    # Arrow down to training
    arrow(ax, 11.5, 6.2, 11.5, 5.0, color="#e67e22", lw=3)

    # =================================================================
    # STEP 4 — LLM Training  (bottom, full width)
    # =================================================================
    train_bg = FancyBboxPatch((0.0, 1.2), 14.0, 3.5,
                               boxstyle="round,pad=0.2", linewidth=2.5,
                               edgecolor="#27ae60", facecolor="#eafaf1",
                               alpha=0.3, zorder=0)
    ax.add_patch(train_bg)

    box(ax, 3.0, 4.2, 3.5, 0.7, "4  LLM Training Begins", "#27ae60",
        fontsize=14)

    # Tuner callout (right side, no overlap)
    callout = (
        "Tuner reads NCCL_TUNER_POLICY\n"
        "Sets cost=0 for measured-best algo/proto\n"
        "\u2192 NCCL uses cluster-optimal config\n"
        "\u2192 All 8 ranks use same selection (safe)"
    )
    ax.text(10.5, 4.2, callout, ha="center", va="center", fontsize=9,
            color="#1e8449",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#27ae60", linewidth=1.5), zorder=3)

    # Training loop
    ax.text(0.8, 3.05, "Training Loop:", ha="left", fontsize=11,
            fontweight="bold", color="#2c3e50")

    box(ax, 2.2, 2.1, 1.8, 0.8, "Forward\nPass", "#3498db",
        fontsize=10, alpha=0.85)
    arrow(ax, 3.15, 2.1, 4.05, 2.1, color="#555", lw=2)

    box(ax, 5.0, 2.1, 1.8, 0.8, "Backward\nPass", "#3498db",
        fontsize=10, alpha=0.85)
    arrow(ax, 5.95, 2.1, 6.85, 2.1, color="#555", lw=2)

    box(ax, 8.2, 2.1, 2.5, 0.9, "AllReduce\nTuner Active", "#27ae60",
        fontsize=12, edgecolor="#1e8449")
    arrow(ax, 9.5, 2.1, 10.55, 2.1, color="#555", lw=2)

    box(ax, 11.8, 2.1, 1.8, 0.8, "Optimizer\nStep", "#3498db",
        fontsize=10, alpha=0.85)

    # Loop-back arrow
    ax.annotate("", xy=(1.3, 2.1), xytext=(11.8, 1.3),
                arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.5,
                                connectionstyle="arc3,rad=-0.25",
                                linestyle="dashed"), zorder=1)
    ax.text(7.0, 1.15, "repeat for each training step", ha="center",
            fontsize=9, color="#999", style="italic")

    # =================================================================
    # Bottom — Result comparison
    # =================================================================
    ax.plot([-0.5, 14.5], [0.2, 0.2], color="#ddd", lw=1.5, zorder=0)

    ax.text(3.5, -0.4, "Without Tuner (NCCL AUTO):", ha="center",
            fontsize=11, fontweight="bold", color="#e74c3c")
    ax.text(3.5, -0.9,
            "Static cost model \u2192 picks Ring when Tree is 57% faster",
            ha="center", fontsize=10, color="#e74c3c")

    ax.text(7.0, -0.65, "vs", ha="center", fontsize=16,
            fontweight="bold", color="#888")

    ax.text(10.5, -0.4, "With Profile-Guided Tuner:", ha="center",
            fontsize=11, fontweight="bold", color="#27ae60")
    ax.text(10.5, -0.9,
            "Measured on THIS cluster \u2192 always picks the fastest config",
            ha="center", fontsize=10, color="#27ae60")

    ax.text(7.0, -1.7,
            "faster AllReduce \u2192 faster training iterations",
            ha="center", fontsize=14, fontweight="bold", color="#1a1a2e",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffde7",
                      edgecolor="#f9a825", linewidth=2.5), zorder=3)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_profile_guided_tuning.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    generate_figure()
