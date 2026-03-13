# GPU Compute Contention

NCCL AllReduce performance under concurrent GPU compute stress at low, medium, and high intensity levels.

## What this measures

- How GPU compute contention affects collective communication bandwidth and latency
- Whether SM contention from concurrent kernels shifts the optimal algorithm/protocol
- Baseline for understanding compute-communication interaction before full overlap experiments

## Hardware

- **A100**: 8x A100 80GB, NVLink 3.0 (Modal)
- **L40S**: 2-GPU and 4-GPU, PCIe Gen4 (FarmShare)

## Contents

| Path | Description |
|------|-------------|
| `run_modal.py` | Modal runner with configurable GPU stress levels |
| `run_nccl_with_contention.sh` | FarmShare runner with contention |
| `gpu_stress_benchmark.cu` | CUDA kernel that generates configurable GPU compute load |
| `a100/results/` | Raw results at each stress level (8x A100) |
| `l40s/2gpu/` | Results for 2x L40S |
| `l40s/4gpu/` | Results for 4x L40S |
| `scripts/` | Plotting scripts |

## Key scripts

- `scripts/plot_gpu_utilization.py` -- GPU utilization and contention impact plots
- `scripts/check_gpu_utilization.sh` -- Utility to verify stress level during experiments
