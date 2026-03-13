# Compute-Communication Overlap

Measures how concurrent compute (matmul on a separate CUDA stream) changes the optimal NCCL algorithm and protocol, and sweeps the NCCL channel count.

## What this measures

- **Iteration proxy**: Simulates a training step (compute + AllReduce) and measures end-to-end iteration time under each algorithm-protocol pair
- **Overlap experiment**: Compares sequential vs overlapped execution to identify "winner flips" -- message sizes where overlap changes the optimal configuration
- **Channel sweep**: Effect of NCCL channel count on overlap performance

## Key finding

6 of 9 message sizes change optimal configuration under overlap on a single NVLink-connected 8-GPU node. Protocols that consume fewer SMs (Simple) become preferable under overlap because LL128's sustained SM polling competes with compute kernels.

## Contents

| Path | Description |
|------|-------------|
| `iteration_proxy.py` | Training-step proxy: matmul + AllReduce on separate streams |
| `run_modal_proxy.py` | Modal runner for the iteration proxy experiment |
| `run_modal_overlap.py` | Modal runner for sequential vs overlap comparison |
| `run_modal_channels.py` | Modal runner for NCCL channel count sweep |
| `proxy_results/` | Raw iteration time measurements |
| `overlap_results/` | Sequential vs overlap results |
| `channel_results/` | Channel sweep results |
| `scripts/` | Analysis and plotting |

## Key scripts

- `scripts/analyze_iteration_times.py` -- Statistical analysis of iteration times
- `scripts/plot_iteration_times.py` -- Iteration time comparison plots
- `scripts/analyze_overlap.py` -- Sequential vs overlap analysis
- `scripts/analyze_channels.py` -- Channel count sweep analysis
- `scripts/cross_experiment_analysis.py` -- Cross-experiment figures (generates several paper figures)
