# Multi-Node Experiments

Measures how NCCL AUTO suboptimality amplifies when scaling from single-node to multi-node topologies.

## What this measures

- AllReduce performance across 1x8, 2x4, and 4x2 A100 GPU topologies
- AUTO gap: percentage latency penalty of AUTO vs the best configuration at each message size
- Same-cluster comparison to control for inter-node network variability
- Node-count sweep to quantify scaling of the AUTO gap

## Key finding

AUTO's latency penalty grows from ~1.2% (single-node, 1x8) to 57.2% (2x4) at 64 MB. AUTO selects Ring for every message size on multi-node even though Tree delivers up to 2.9x higher throughput.

## Contents

| Path | Description |
|------|-------------|
| `run_modal_multinode_expanded.py` | Modal runner: full algorithm-protocol sweep across topologies |
| `run_modal_node_sweep.py` | Modal runner: node-count sweep (1x8, 2x4, 4x2, 8x1) |
| `run_modal_same_cluster_comparison.py` | Modal runner: same-cluster consistency check |
| `run_bandit_validation_multinode.py` | Bandit tuner validation on multi-node |
| `profile_cluster.py` | Cluster topology profiling utility |
| `results/multinode_experiment/` | 2-node (2x4) raw timing data |
| `results/multinode_expanded/` | Expanded topology sweep results |
| `results/node_sweep/` | Node-count sweep results (1x8 through 8x1) |
| `results/same_cluster_comparison/` | Same-cluster control experiment |
| `scripts/` | Figure generation scripts |

## Key scripts

- `scripts/regenerate_fig6_large_fonts.py` -- Generates the 3-panel multi-node figure (Fig 3 in paper)
- `scripts/generate_expanded_fig6.py` -- Extended version with all topologies
- `scripts/generate_node_sweep_figure.py` -- Node-count scaling figure

## AllGather and AWS experiments

AllGather experiments were also run on both Modal (A100) and AWS g5.xlarge (A10G) instances to verify that the AUTO suboptimality pattern generalizes beyond AllReduce. Results are in `allgather-preview/` and `aws-validation/`.
