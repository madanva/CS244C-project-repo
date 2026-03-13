# Curated Data

Pre-processed JSON datasets used by the figure generation scripts. These are the clean, paper-ready versions extracted from the raw experiment results.

## Files

| File | Description | Source experiment |
|------|-------------|-------------------|
| `overlap_experiment.json` | Sequential vs overlap iteration times per (algorithm, protocol, message_size) | `experiments/overlap/` |
| `channel_experiment.json` | Iteration times across NCCL channel counts | `experiments/overlap/` |
| `multinode_results.json` | Multi-node AllReduce times for 2x4 topology (sequential + overlap) | `experiments/multinode/` |
| `tuner_policy.json` | Optimal (algorithm, protocol) per (message_size, mode) -- the profiled policy | `experiments/tuner_evaluation/` |
