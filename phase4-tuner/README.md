# Phase 4: Workload-Aware NCCL Tuner Plugin

## Thesis

NCCL's algorithm/protocol selection is optimized for **isolated** collectives. Real training workloads **overlap** compute and communication on concurrent CUDA streams. Under overlap, protocol choice affects SM contention with the compute kernel, changing which (algo, proto) is optimal. NCCL's cost model cannot account for this because it has no visibility into concurrent compute. A workload-aware tuner that accounts for compute context can close this gap.

## Key Insight

Protocols differ in GPU resource consumption:

- **Simple** — large-chunk transfers, high SM usage for data copies
- **LL128** — 128-byte flag-based transfers, fewer SMs but more memory bandwidth
- **LL** — 8-byte units, lowest SM occupancy per transfer

When compute (e.g., matmul) runs concurrently on the same GPU, SM-heavy protocols (Simple) compete for resources, potentially making SM-lighter protocols (LL128) faster for overall iteration time — even though Simple is faster for the collective in isolation.

**No existing paper studies this effect.** Prior work (TACCL, TE-CCL, MSCCL, MCCS, AdapCC) optimizes collectives in isolation. The overlap characterization literature (Lee et al., ISPASS 2025; "Demystifying NCCL" 2025) measures the problem but never varies protocol selection.

## How the tuner works

NCCL loads an external shared library (tuner plugin API v4/v5):

- **pluginInit**: called once per communicator (receives nRanks, nNodes, topology)
- **pluginGetCollInfo**: called per collective with `collType`, `nBytes`. The plugin writes a **cost table** (per algo × proto); NCCL picks the lowest-cost entry
- **pluginFinalize**: cleanup

The API does **not** pass compute context. Workload-awareness is achieved by profiling, then encoding the overlap-aware policy into the plugin's cost table.

## Experiment design

### Experiment 1: Sequential baseline sweep (`run_modal.py`)

Characterize NCCL AUTO in isolation — the "it works fine" baseline.

- **9 message sizes** × **5 configs** (AUTO, Tree+Simple, Tree+LL128, Ring+Simple, Ring+LL128)
- 50 iterations, 10 warmup per config
- No overlap (sequential compute → allreduce)

**Result**: AUTO is near-optimal in isolation (0–2.5% gap). Largest gap at 16MB (tree→ring transition).

### Experiment 2: Overlap sweep (`run_modal_overlap.py`, Block 1-2)

Same 9 sizes × 5 configs, but with `--overlap` (concurrent CUDA streams).

**Key question**: Does the best config change under overlap? If the winner flips (e.g., Simple → LL128), that proves NCCL's isolation-based cost model is wrong under overlap.

### Experiment 3: Compute intensity sweep (`run_modal_overlap.py`, Block 3)

At 4 key sizes (1MB, 4MB, 16MB, 64MB), sweep compute intensity:

| Compute level | Matmul size | Expected SM pressure |
|---------------|-------------|----------------------|
| light | 1024×1024 | ~1 ms — comm-dominated |
| medium | 2048×2048 | ~4 ms — balanced |
| default | 4096×4096 | ~25 ms — typical training layer |
| heavy | 6144×6144 | ~80 ms — compute-heavy |
| extreme | 8192×8192 | ~190 ms — SM-saturated |

This produces a 2D landscape: **message size × compute intensity → optimal (algo, proto)**.

**Hypothesis**: As compute load increases, SM-lighter protocols (LL128, LL) increasingly dominate because they steal fewer SMs from the matmul.

## Configs tested

| Config | NCCL_ALGO | NCCL_PROTO |
|--------|-----------|------------|
| auto | (default) | (default) |
| tree_simple | Tree | Simple |
| tree_ll128 | Tree | LL128 |
| ring_simple | Ring | Simple |
| ring_ll128 | Ring | LL128 |

## Analysis plots

### From `analyze_and_plot.py` (sequential sweep):

| Plot | Description |
|------|-------------|
| `phase4_landscape.png` | Iteration time vs message size for all configs |
| `phase4_auto_gap.png` | AUTO overhead vs best config at each size (%) |

### From `analyze_overlap.py` (overlap experiment):

| Plot | Description |
|------|-------------|
| `seq_vs_overlap.png` | Side-by-side bar charts: sequential vs overlap |
| `winner_flip.png` | Heatmap: which config wins at each size, seq vs overlap |
| `auto_gap_comparison.png` | AUTO gap in sequential vs overlap (the key finding) |
| `compute_heatmap.png` | Best config across (msg_size × compute_intensity) |
| `compute_sweep_lines.png` | Iteration time vs compute intensity per config |
| `sm_contention.png` | Protocol slowdown relative to sequential baseline |

## Running

```bash
cd phase4-tuner/a100-8gpu-new

# Experiment 1: Sequential baselines (already completed)
modal run --detach run_modal.py

# Experiments 2-3: Overlap + compute intensity sweep
modal run --detach run_modal_overlap.py
```

## Reference

- NCCL tuner plugin example: https://github.com/NVIDIA/nccl/tree/master/ext-tuner/example
- "Demystifying NCCL" (2025): protocol internals characterization
- Lee et al. (ISPASS 2025): compute-communication overlap characterization
- TACCL (NSDI 2023), TE-CCL (SIGCOMM 2024): topology-aware collective synthesis
- MCCS (SIGCOMM 2024): managed collective communication for multi-tenant

## Tuner plugin code

### Static CSV-based tuner

`workload_aware_8gpu.conf` — policy derived from profiling results.

```bash
export NCCL_TUNER_PLUGIN=libnccl-tuner-example.so
export NCCL_TUNER_CONFIG_FILE=/path/to/phase4-tuner/workload_aware_8gpu.conf
```

### RL bandit tuner plugin

`rl_bandit_tuner_plugin.c` — online bandit that learns the best (algo, proto) per workload context. Supports epsilon-greedy, UCB1, and Thompson Sampling.

```bash
export NCCL_TUNER_PLUGIN=libnccl-tuner-rl-bandit.so
export NCCL_TUNER_STRATEGY=ucb1
```

## Status

- **Experiment 1** (sequential sweep): Complete. AUTO near-optimal in isolation (0-2.5% gap).
- **Experiment 2-3** (overlap + compute intensity): Ready to run (`run_modal_overlap.py`).
- **Plugin code**: Static CSV tuner + RL bandit tuner (`rl_bandit_tuner_plugin.c`).
- **Header discovery**: `get_nccl_tuner_info.py` — dumps NCCL tuner headers from runtime.
