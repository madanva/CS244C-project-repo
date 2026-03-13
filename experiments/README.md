# Experiments

Each subdirectory contains the runner scripts, raw results, and analysis/plotting scripts for one set of experiments. Runner scripts target either [Modal](https://modal.com/) (cloud GPUs) or Stanford FarmShare.

## Directory Map

| Directory | Paper Section | What it measures |
|-----------|---------------|------------------|
| `baseline/` | Section 3 (Characterization) | NCCL AllReduce performance across algorithms, protocols, and message sizes on A100 and L40S |
| `contention/` | Section 3 | Impact of concurrent GPU compute stress on collective communication bandwidth/latency |
| `overlap/` | Sections 3-4 | Compute-communication overlap effects on optimal configuration; NCCL channel count sweep |
| `multinode/` | Section 3 | Multi-node amplification of AUTO suboptimality across 1x8, 2x4, 4x2 topologies |
| `tuner_evaluation/` | Section 5 | Validation of profile-guided and RL bandit tuners against AUTO baseline |

## Common structure

Each experiment directory follows this layout:

```
experiment_name/
├── run_*.py              # Modal runner scripts
├── run_*.sh              # FarmShare runner scripts (where applicable)
├── results/              # Raw output (timing files, JSON summaries)
│   └── ...
└── scripts/              # Analysis and figure generation
    └── ...
```

## Hardware

- **A100**: 8x NVIDIA A100 80GB SXM4, NVLink 3.0, NVSwitch (single-node) + InfiniBand (multi-node)
- **L40S**: NVIDIA L40S, PCIe Gen4 (Stanford FarmShare)
