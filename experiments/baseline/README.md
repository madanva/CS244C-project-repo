# Baseline Characterization

NCCL AllReduce performance across all algorithm-protocol combinations and message sizes, measured in isolation (no concurrent compute).

## What this measures

- Bandwidth and latency for each (algorithm, protocol) pair: Tree/Ring x Simple/LL128/LL
- Protocol transition points (LL128 to Simple at ~4-16 MB)
- `AUTO` selection accuracy when no compute is running

## Hardware

- **A100**: 8x A100 80GB, NVLink 3.0, NVSwitch (Modal)
- **L40S**: 2-GPU and 4-GPU configurations, PCIe Gen4 (Stanford FarmShare)

## Contents

| Path | Description |
|------|-------------|
| `run_modal.py` | Modal runner: sweeps algorithms, protocols, message sizes |
| `run_nccl_farmshare.sh` | FarmShare runner for L40S |
| `a100/results/` | Raw nccl-tests output for 8x A100 |
| `l40s/2gpu_results/` | Raw results for 2x L40S |
| `l40s/4gpu_results/` | Raw results for 4x L40S |
| `topology/` | NCCL graph/topo XML files and cluster topology report |
| `scripts/` | Plotting and analysis scripts |

## Key scripts

- `scripts/plot_nccl_bw.py` -- Bandwidth vs message size plots
- `scripts/plot_nccl_latency.py` -- Latency vs message size plots
- `scripts/plot_nccl_multi.py` -- Multi-protocol comparison plots
- `scripts/analyze_transitions.py` -- Protocol transition point analysis
- `scripts/compare_protocols.py` -- Head-to-head protocol comparison
