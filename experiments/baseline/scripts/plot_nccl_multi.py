#!/usr/bin/env python3
"""
Plot multiple NCCL test results from different folders, labeling each line by folder.
Uses both bandwidth and latency plotting functions from existing scripts.

Example usage:
python3 plot_nccl_multi.py /path/to/folder1 /path/to/folder2 --output_dir multi_graphs --arch "A100"
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import parsing functions from existing scripts
from plot_nccl_bw import parse_nccl_output as parse_bw
from plot_nccl_latency import parse_nccl_results as parse_latency

def plot_multi_bandwidth(folder_files, output_dir, arch_label=""):
    fig, ax = plt.subplots(figsize=(12, 6))
    folder_names = list(folder_files.keys())
    folder_label = '_'.join(folder_names)
    arch_suffix = f"_{arch_label}" if arch_label else ""
    for folder, files in folder_files.items():
        for f in files:
            sizes, oop_bw, ip_bw = parse_bw(f)
            if len(sizes) == 0:
                continue
            # Plot out-of-place
            ax.plot(sizes, oop_bw, label=f"{folder} (Out-of-place)", linewidth=2)
            # Plot in-place
            ax.plot(sizes, ip_bw, label=f"{folder} (In-place)", linewidth=2, linestyle='--')
    ax.set_xlabel('Message Size (Bytes)', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    title = f"NCCL All-Reduce Bandwidth [{folder_label}] {arch_label}" if arch_label else f"NCCL All-Reduce Bandwidth [{folder_label}]"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"multi_bw_plot_{folder_label}{arch_suffix}.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-bandwidth plot saved to: {output_file}")

def plot_multi_latency(folder_files, output_dir, arch_label=""):
    fig, ax = plt.subplots(figsize=(12, 6))
    folder_names = list(folder_files.keys())
    folder_label = '_'.join(folder_names)
    arch_suffix = f"_{arch_label}" if arch_label else ""
    for folder, files in folder_files.items():
        for f in files:
            sizes, out_times, in_times = parse_latency(f)
            if len(sizes) == 0:
                continue
            # Plot out-of-place
            ax.plot(sizes, out_times, label=f"{folder} (Out-of-place)", linewidth=2)
            # Plot in-place
            ax.plot(sizes, in_times, label=f"{folder} (In-place)", linewidth=2, linestyle='--')
    ax.set_xlabel('Message Size (Bytes)', fontsize=12)
    ax.set_ylabel('Latency (us)', fontsize=12)
    title = f"NCCL All-Reduce Latency [{folder_label}] {arch_label}" if arch_label else f"NCCL All-Reduce Latency [{folder_label}]"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"multi_latency_plot_{folder_label}{arch_suffix}.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-latency plot saved to: {output_file}")

def find_txt_files(folders):
    folder_files = {}
    for folder in folders:
        txt_files = []
        for f in os.listdir(folder):
            full_path = os.path.join(folder, f)
            if os.path.isfile(full_path) and f.endswith('.txt'):
                txt_files.append(full_path)
        if txt_files:
            folder_files[os.path.basename(folder)] = txt_files
    return folder_files

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot multiple NCCL test results from folders.')
    parser.add_argument('folders', nargs='+', help='Folders containing NCCL .txt result files')
    parser.add_argument('--output_dir', type=str, default='multi_graphs', help='Output directory for plots')
    parser.add_argument('--arch', type=str, default='', help='GPU architecture label for plot titles')
    args = parser.parse_args()

    folder_files = find_txt_files(args.folders)
    if not folder_files:
        print("No .txt files found in provided folders.")
        sys.exit(1)

    plot_multi_bandwidth(folder_files, args.output_dir, args.arch)
    plot_multi_latency(folder_files, args.output_dir, args.arch)
