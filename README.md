# Bandit-Guided NCCL Tuning for Multi-Node GPU Clusters

**Varun Madan** (vmadan@stanford.edu) and **Ashley Raigosa** (raigosa@stanford.edu)
Stanford University -- CS244C: Topics in Networked Systems, Winter 2026

## Key Finding

NCCL's built-in `AUTO` selector picks suboptimal algorithm-protocol pairs under compute-communication overlap, and these gaps amplify across nodes -- up to 57% latency penalty at 64 MB on a 2-node A100 cluster. An online RL bandit tuner learns the optimal configuration in ~40 iterations and achieves 25-42% latency reductions with zero regressions.

## Repository Structure

```
.
├── tuner/                       # NCCL tuner plugin source code (the main contribution)
├── experiments/
│   ├── baseline/                # NCCL baseline characterization (A100 + L40S)
│   ├── contention/              # GPU compute contention experiments
│   ├── overlap/                 # Compute-communication overlap + channel sweep
│   ├── multinode/               # Multi-node amplification experiments
│   └── tuner_evaluation/        # Bandit + profile-guided tuner validation
├── data/                        # Curated JSON datasets (used by figure scripts)
├── figures/                     # Final paper figures
└── nccl-tests/                  # NVIDIA nccl-tests benchmark suite (submodule)
```

## The Tuner

The core contribution is an NCCL external tuner plugin (`tuner/`) that replaces the built-in `AUTO` cost model. Three variants are provided:

| File | Description |
|------|-------------|
| `rl_bandit_tuner_v2.c` | **Online RL bandit** -- deterministic round-robin exploration, IQR-trimmed mean, 5% safety gate |
| `workload_aware_tuner_v3.c` | Profile-guided static tuner -- loads a pre-computed policy table |
| `workload_aware_tuner_v2.c` | Overlap-aware static tuner -- separate sequential/overlap policies |

### Build and Use

```bash
cd tuner/
gcc -shared -fPIC -o libnccl-tuner.so rl_bandit_tuner_v2.c -I. -ldl
export NCCL_TUNER_PLUGIN=$(pwd)/libnccl-tuner.so
# Now run any NCCL workload -- the bandit tunes automatically
```

Requires NCCL >= 2.19 (tuner plugin API).

## Reproducing Results

Each experiment directory contains runner scripts (Modal cloud or Farmshare) and analysis/plotting scripts. See [`experiments/`](experiments/) for a mapping of directories to paper sections.

### Hardware

- **A100 cluster**: 8x NVIDIA A100 80GB SXM4, NVLink 3.0 (600 GB/s bisection), inter-node InfiniBand
- **L40S cluster**: NVIDIA L40S GPUs, PCIe Gen4

### Running an experiment

```bash
# Example: run the multi-node overlap experiment on Modal
cd experiments/multinode/
modal run run_modal_multinode_expanded.py

# Generate the figure
cd scripts/
python regenerate_fig6_large_fonts.py
```

## Citation

```bibtex
@inproceedings{madan2026bandit,
  title     = {Bandit-Guided {NCCL} Tuning for Multi-Node {GPU} Clusters},
  author    = {Varun Madan and Ashley Raigosa},
  booktitle = {Stanford CS244C: Topics in Networked Systems},
  year      = {2026}
}
```
