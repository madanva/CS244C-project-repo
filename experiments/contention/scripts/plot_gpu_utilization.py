"""Plot GPU utilization from CSV (e.g. from nvidia-smi logs). Same as L40S; use Agg backend for headless."""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_file = os.environ.get(
        "GPU_UTIL_CSV",
        os.path.join(os.path.dirname(__file__), "gpu_utilization_logs", "gpu_utilization_log_high.csv"),
    )
    if len(sys.argv) >= 2:
        csv_file = sys.argv[1]
    if not os.path.isfile(csv_file):
        print(f"File not found: {csv_file}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv_file)
    df["memory_used"] = df["memory_used"].apply(lambda x: int(str(x).strip()))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "tab:blue"
    ax1.set_xlabel("Timestamp (s)")
    ax1.set_ylabel("GPU Utilization (%)", color=color1)
    l1, = ax1.plot(df["timestamp"], df["gpu_utilization"], color=color1, marker="o", label="GPU Utilization (%)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True)
    max_util = df["gpu_utilization"].max()
    if max_util < 10:
        ax1.set_ylim(0, 5)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Memory Used (MiB)", color=color2)
    l2, = ax2.plot(df["timestamp"], df["memory_used"], color=color2, marker="x", label="Memory Used (MiB)")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("GPU Utilization and Memory Usage Over Time")
    lines, labels = [l1, l2], [line.get_label() for line in [l1, l2]]
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout()

    base = os.path.basename(csv_file)
    if base.startswith("gpu_utilization_log_") and base.endswith(".csv"):
        suffix = base[len("gpu_utilization_log_"): -len(".csv")]
        plot_name = f"gpu_utilization_plot_{suffix}.png"
    else:
        plot_name = "gpu_utilization_plot.png"
    outdir = os.path.dirname(csv_file)
    save_path = os.path.join(outdir, plot_name)
    plt.savefig(save_path)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
