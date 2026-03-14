# NCCL Tuner Plugins

External tuner plugins for NCCL (>= 2.19) that override the built-in `AUTO` algorithm/protocol selection. These plugins use the NCCL tuner plugin API to intercept collective calls and substitute a learned or pre-computed configuration.

## Files

### Tuner implementations (newest first)

| File | Approach | Description |
|------|----------|-------------|
| `rl_bandit_tuner_v2.c` | Online RL bandit | 4-armed bandit with deterministic round-robin exploration, IQR-trimmed mean rewards, and a 5% safety gate. This is the primary contribution. |
| `rl_bandit_tuner_plugin.c` | Online RL bandit (v1) | Epsilon-greedy / UCB1 bandit. Superseded by v2. |
| `workload_aware_tuner_v3.c` | Profile-guided static | Loads a C lookup table (`tuner_policy_table.h`) generated from profiling data. |
| `workload_aware_tuner_v2.c` | Overlap-aware static | Maintains separate sequential and overlap policies, selected at runtime. |
| `workload_aware_tuner_plugin.c` | Static CSV lookup (v1) | Reads `workload_aware_8gpu.conf` at init. |

### Supporting files

| File | Purpose |
|------|---------|
| `tuner.h` | NCCL tuner plugin API header (defines `ncclTuner_v3_t`) |
| `common.h` | NCCL common type definitions |
| `err.h` | NCCL error codes |
| `workload_aware_8gpu.conf` | CSV policy: maps (collective, size_range, nNodes, nRanks) to (algorithm, protocol) |
| `tuner_policy_table.h` | Generated C lookup table from profiling data (used by v3) |

## Build

```bash
gcc -shared -fPIC -o libnccl-tuner.so rl_bandit_tuner_v2.c -I. -ldl
```

## Usage

```bash
export NCCL_TUNER_PLUGIN=$(pwd)/libnccl-tuner.so
# Run any NCCL workload (nccl-tests, PyTorch DDP, etc.)
```

The plugin intercepts every `ncclAllReduce` (and other collectives) and overrides the algorithm and protocol selection. The bandit converges in ~40 iterations per message-size bucket.

## How the bandit works

1. **Arms**: {Tree+Simple, Tree+LL128, Ring+Simple, Ring+LL128}
2. **Exploration**: Deterministic round-robin across all arms for 10 rounds per bucket
3. **Reward**: Inverse of IQR-trimmed mean latency (robust to outliers)
4. **Exploitation**: Best arm selected after exploration, with a 5% safety gate that falls back to AUTO if the bandit's choice is worse
