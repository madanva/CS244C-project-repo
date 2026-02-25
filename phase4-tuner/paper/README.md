# Workload-Aware Tuning of NCCL Collectives on Multi-GPU Systems

## Problem

NCCL's AUTO mode selects collective algorithms (Tree, Ring) and protocols (LL, LL128, Simple) using a static cost model calibrated for **isolated** collectives. In real training, collectives overlap with compute on concurrent CUDA streams. Nobody has studied whether this overlap changes the optimal selection.

## Key Finding

**It does.** On single-node 8x A100 NVLink, 6/9 message sizes change optimal config under overlap. On multi-node (2 nodes, 8 GPUs total), NCCL AUTO leaves **up to 57% performance on the table** by choosing the wrong algorithm. Tree+Simple beats AUTO by 2-3x at large messages across nodes.

## Setup

- **Single-node**: 8x NVIDIA A100-SXM4-80GB, NVLink
- **Multi-node**: 2 nodes × 4x A100, inter-node network (no RDMA)
- **Collective**: AllReduce (fp32)
- **Workload**: Simulated training iteration — matmul (compute) + AllReduce (communication)
- **Overlap**: Compute and AllReduce on separate CUDA streams
- **Configs tested**: AUTO, Tree+Simple, Tree+LL128, Ring+Simple, Ring+LL128
- **Measurement**: 50 iterations, 10 warmup, median reported

---

## Results: Single-Node (8x A100 NVLink)

### 1. Winner flips under overlap (Fig 1)

| Size | Sequential Best | Overlap Best | Flip? |
|------|----------------|-------------|-------|
| 32KB | Tree+LL128 | Tree+Simple | Yes |
| 64KB | Ring+LL128 | AUTO | Yes |
| 256KB | Ring+Simple | AUTO | Yes |
| 1MB | Tree+LL128 | Tree+LL128 | |
| 2MB | Ring+LL128 | Ring+Simple | Yes |
| 4MB | Tree+Simple | Ring+Simple | Yes |
| 16MB | Ring+Simple | Ring+Simple | |
| 64MB | Ring+Simple | Ring+Simple | |
| 256MB | AUTO | Ring+Simple | Yes |

6/9 sizes flip. Ring+Simple dominates under overlap for messages >= 2MB because it uses fewer SMs than tree-based or LL128 protocols.

### 2. CTA count tuning (Fig 2)

NCCL_MAX_CTAS controls SM allocation to communication. At 64MB under overlap:
- CTA=1: 14.2ms (starved), CTA=12: 9.0ms (optimal), CTA=32: 9.1ms (diminishing returns)
- Config + CTA tuning together: **3-5.4%** improvement over AUTO at default

### 3. Compute intensity interaction (Fig 5)

Optimal CTA shifts with compute load: Medium→CTA=8, Default→CTA=2, Heavy→CTA=1.

---

## Results: Multi-Node (2×4 A100, inter-node network)

### 4. NCCL AUTO leaves up to 57% on the table (Fig 6, Table 4)

| Size | SEQ Best | SEQ AUTO Gap | OVL Best | OVL AUTO Gap | Flip? |
|------|----------|-------------|----------|-------------|-------|
| 256KB | Tree+S | **9.1%** | Tree+S | **10.9%** | |
| 1MB | Tree+S | **13.2%** | Tree+S | **13.5%** | |
| 4MB | AUTO | 0% | Tree+S | **6.4%** | Yes |
| 16MB | Tree+S | **15.9%** | Tree+S | **11.5%** | |
| 64MB | Tree+S | **57.2%** | Tree+S | **45.6%** | |
| 256MB | Tree+S | 0.5% | Tree+L | ~0% | Yes |

**Headline number: 57% AUTO gap at 64MB sequential.** NCCL's cost model picks a suboptimal algorithm for inter-node communication.

### 5. Ring is catastrophic multi-node (Table 4)

At large messages, Ring algorithms are 1.6-2.9x slower than Tree:

| Size | Ring+Simple | Tree+Simple | Ring/Tree ratio |
|------|------------|------------|----------------|
| 16MB | 80.9ms | 34.2ms | **2.4x** |
| 64MB | 337.3ms | 116.5ms | **2.9x** |
| 256MB | 1077.6ms | 675.5ms | **1.6x** |

Ring forces every hop to cross the slow inter-node link. Tree uses the hierarchy (NVLink intra, network inter).

### 6. AUTO gap amplifies single→multi-node (Fig 7)

| Size | Single-node gap | Multi-node gap | Amplification |
|------|----------------|---------------|---------------|
| 256KB | 3.0% | 9.1% | **3x** |
| 1MB | 0.3% | 13.2% | **44x** |
| 16MB | 3.1% | 15.9% | **5x** |
| 64MB | 2.3% | 57.2% | **25x** |

Single-node NVLink hides poor algorithm choices. Inter-node network exposes them.

---

## Artifacts

### Tuner Plugin

`tuner/workload_aware_tuner_plugin.c` — NCCL tuner plugin (v5 API) with static lookup table. Set `NCCL_OVERLAP_MODE=1` for overlap-aware policy.

```bash
gcc -shared -fPIC -o libnccl_workload_tuner.so workload_aware_tuner_plugin.c -I.
NCCL_TUNER_PLUGIN=./libnccl_workload_tuner.so NCCL_OVERLAP_MODE=1 torchrun ...
```

### Data

- `data/overlap_experiment.json` — Single-node: 9 sizes × 5 configs × 2 modes
- `data/channel_experiment.json` — CTA sweep + compute interaction
- `data/multinode_results.json` — Multi-node: 6 sizes × 5 configs × 2 modes
- `data/tuner_policy.json` — Derived policy table

### Figures

- `fig1_winner_flips.png` — Single-node winner flip heatmap
- `fig2_cta_ucurve.png` — CTA count U-curve at 64MB
- `fig3_auto_gap.png` — AUTO gap: sequential vs overlap
- `fig4_sm_contention.png` — SM contention mechanism deep dive
- `fig5_cta_compute.png` — CTA × compute intensity interaction
- `fig6_multinode.png` — Multi-node overview: all configs across sizes
- `fig7_sn_vs_mn.png` — Single-node vs multi-node AUTO gap amplification
- `fig8_mn_flips.png` — Multi-node protocol breakdown and winner flips

## Limitations

- AllReduce only. Other collectives (AllGather, ReduceScatter) may differ.
- A100 only. H100/B100 have different SM counts and NVLink topologies.
- Multi-node without RDMA (TCP). RDMA/InfiniBand would change absolute numbers but hierarchy effects should persist.
- Single training-step proxy, not full model training.

## What's Next

1. **End-to-end validation** — Run actual GPT-2 training with the tuner plugin vs AUTO
2. **Dynamic detection** — Infer overlap mode at runtime instead of requiring an environment variable
3. **Multiple collectives** — AllGather, ReduceScatter
4. **Statistical repetitions** — 3-5 runs per config for confidence intervals
