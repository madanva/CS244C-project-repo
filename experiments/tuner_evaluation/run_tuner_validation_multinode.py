"""
Modal app: Profile-guided tuner validation across multiple topologies.

For each topology (1×8, 2×4, 4×2):
  Phase 1 — PROFILE: Quick benchmark of all algorithm/protocol combos.
  Phase 2 — GENERATE POLICY: Pick best config per size band.
  Phase 3 — VALIDATE: Run AUTO vs tuner-with-profiled-policy.

Uses orchestrator pattern to run topologies sequentially (max 10 GPUs).
"""

import json
import os
import time
from pathlib import Path

import modal
import modal.experimental

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Image: CUDA + PyTorch + compiled tuner plugin
tuner_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget", "libibverbs-dev", "libibverbs1")
    .run_commands("pip install --upgrade pip")
    .pip_install("torch", "numpy", "scipy")
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
    .run_commands(
        "cd /repo/phase4-tuner && "
        "gcc -shared -fPIC -o /repo/libnccl_tuner_v3.so "
        "workload_aware_tuner_v3.c -I. && "
        "echo 'Tuner plugin compiled successfully' && "
        "ls -la /repo/libnccl_tuner_v3.so"
    )
)

volume = modal.Volume.from_name("browser-networking-tests-storage", create_if_missing=True)
VOLUME_PATH = "/results"

app = modal.App("browser-networking-tests")

WORLD_SIZE = 8

MSG_SIZES = [
    ("256KB",  65_536),
    ("1MB",    262_144),
    ("4MB",    1_048_576),
    ("16MB",   4_194_304),
    ("64MB",   16_777_216),
    ("256MB",  67_108_864),
]

# Configs to profile (no tuner plugin — use NCCL env vars to force)
PROFILE_CONFIGS = [
    ("auto",         {}),
    ("tree_simple",  {"NCCL_ALGO": "Tree", "NCCL_PROTO": "Simple"}),
    ("tree_ll",      {"NCCL_ALGO": "Tree", "NCCL_PROTO": "LL"}),
    ("tree_ll128",   {"NCCL_ALGO": "Tree", "NCCL_PROTO": "LL128"}),
    ("ring_simple",  {"NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple"}),
    ("ring_ll",      {"NCCL_ALGO": "Ring", "NCCL_PROTO": "LL"}),
    ("ring_ll128",   {"NCCL_ALGO": "Ring", "NCCL_PROTO": "LL128"}),
]

ALGO_PROTO_MAP = {
    "tree_simple":  (0, 2),
    "tree_ll":      (0, 0),
    "tree_ll128":   (0, 1),
    "ring_simple":  (1, 2),
    "ring_ll":      (1, 0),
    "ring_ll128":   (1, 1),
}

PROFILE_ITERS = 50
PROFILE_WARMUP = 15
VALIDATE_ITERS = 50
VALIDATE_WARMUP = 10
COMPUTE_MUL = 4096

# v4: Multi-gate decision thresholds
P_VALUE_THRESHOLD = 0.01       # Mann-Whitney U one-sided p-value
CLIFFS_DELTA_THRESHOLD = 0.33  # Medium effect size
MIN_IMPROVEMENT_SMALL = 0.05   # 5% for sizes < 64MB
MIN_IMPROVEMENT_LARGE = 0.10   # 10% for sizes >= 64MB (higher variance)
LARGE_SIZES = {"64MB", "256MB"}


def compute_band(nBytes):
    if nBytes < 1024:           return 0
    if nBytes < 256 * 1024:     return 1
    if nBytes < 1024 * 1024:    return 2
    if nBytes < 4 * 1024**2:    return 3
    if nBytes < 32 * 1024**2:   return 4
    if nBytes < 128 * 1024**2:  return 5
    return 6


# ---------------------------------------------------------------------------
# Worker function (parameterized for any topology)
# ---------------------------------------------------------------------------
def worker_fn(local_rank, gpus_per_node, node_rank, master_addr, master_port,
              nccl_env, size_elems, overlap, out_path, iters, warmup):
    """Single GPU worker for profiling or validation."""
    import torch
    import torch.distributed as dist

    global_rank = node_rank * gpus_per_node + local_rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_TUNER_PLUGIN",
                "NCCL_TUNER_POLICY", "NCCL_TUNER_REWARD_FILE"):
        os.environ.pop(key, None)
    for k, v in nccl_env.items():
        os.environ[k] = v
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    os.environ["LOCAL_RANK"] = str(local_rank)

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=WORLD_SIZE)

    grad = torch.randn(size_elems, device=device, dtype=torch.float32) / WORLD_SIZE

    if not overlap:
        for _ in range(warmup):
            a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            tmp = grad.clone()
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            tmp = grad.clone()
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
    else:
        compute_stream = torch.cuda.Stream(device=device)
        comm_stream = torch.cuda.Stream(device=device)

        for _ in range(warmup):
            with torch.cuda.stream(compute_stream):
                a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                _ = torch.matmul(a, b)
            with torch.cuda.stream(comm_stream):
                tmp = grad.clone()
                dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.stream(compute_stream):
                a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                _ = torch.matmul(a, b)
            with torch.cuda.stream(comm_stream):
                tmp = grad.clone()
                dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    if global_rank == 0 and out_path:
        Path(out_path).write_text("\n".join(f"{t:.3f}" for t in times_ms) + "\n")

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Core: run full validation on a given cluster
# ---------------------------------------------------------------------------
def clean_timings(vals):
    """IQR-based outlier removal."""
    import numpy as np
    arr = np.array(vals)
    if len(arr) < 5:
        return arr
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    mask = (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
    return arr[mask]


def cliffs_delta(a, b):
    """Cliff's delta effect size. Positive = a tends to be larger than b."""
    import numpy as np
    # Vectorized for speed
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff_matrix = a[:, None] - b[None, :]
    more = np.sum(diff_matrix > 0)
    less = np.sum(diff_matrix < 0)
    n = len(a) * len(b)
    return (more - less) / n if n > 0 else 0.0


def split_half_agrees(auto_vals, cfg_vals):
    """Check if both halves of the data agree that cfg is faster than auto."""
    import numpy as np
    # Split into odd/even indexed samples
    auto_odd = auto_vals[1::2]
    auto_even = auto_vals[::2]
    cfg_odd = cfg_vals[1::2]
    cfg_even = cfg_vals[::2]
    if len(auto_odd) < 3 or len(cfg_odd) < 3:
        return False
    # Both halves must agree cfg is faster (lower mean)
    half1_faster = np.mean(cfg_odd) < np.mean(auto_odd)
    half2_faster = np.mean(cfg_even) < np.mean(auto_even)
    return half1_faster and half2_faster


def generate_policy_for_mode(results_dir, mode_label, node_rank):
    """Generate NCCL_TUNER_POLICY using multi-gate confidence-aware decisions.

    Gates (ALL must pass to override AUTO):
      1. IQR outlier trimming on raw profiling data
      2. Mann-Whitney U test: p < 0.01 (one-sided)
      3. Cliff's delta: |d| > 0.33 (medium effect)
      4. Split-half consistency: both halves agree
      5. Size-adaptive improvement threshold: 5% small / 10% large
    """
    import numpy as np
    from scipy.stats import mannwhitneyu

    policy_entries = []
    policy_details = {}
    decision_stats = {}

    for size_label, size_elems in MSG_SIZES:
        nBytes = size_elems * 4
        band = compute_band(nBytes)
        min_imp = MIN_IMPROVEMENT_LARGE if size_label in LARGE_SIZES else MIN_IMPROVEMENT_SMALL

        # Load raw AUTO timings
        auto_path = results_dir / f"times_profile_{mode_label}_{size_label}_auto.txt"
        if not auto_path.is_file():
            continue
        auto_raw = [float(x) for x in auto_path.read_text().strip().split("\n") if x.strip()]
        auto_clean = clean_timings(auto_raw)
        if len(auto_clean) < 10:
            continue
        auto_mean = float(np.mean(auto_clean))

        # Evaluate each config
        best_cfg = None
        best_stats = None

        for cfg_label in ALGO_PROTO_MAP:
            cfg_path = results_dir / f"times_profile_{mode_label}_{size_label}_{cfg_label}.txt"
            if not cfg_path.is_file():
                continue
            cfg_raw = [float(x) for x in cfg_path.read_text().strip().split("\n") if x.strip()]
            cfg_clean = clean_timings(cfg_raw)
            if len(cfg_clean) < 10:
                continue
            cfg_mean = float(np.mean(cfg_clean))

            improvement = (auto_mean - cfg_mean) / auto_mean

            # Gate 1: Basic improvement check
            if improvement <= 0:
                continue

            # Gate 2: Mann-Whitney U test (one-sided: auto > cfg?)
            try:
                stat, p_two = mannwhitneyu(auto_clean, cfg_clean, alternative='greater')
                p_val = float(p_two)
            except Exception:
                p_val = 1.0

            # Gate 3: Cliff's delta (auto vs cfg: positive = auto tends to be larger = cfg faster)
            delta = cliffs_delta(auto_clean, cfg_clean)

            # Gate 4: Split-half consistency
            half_ok = split_half_agrees(
                np.array(auto_raw),  # use raw (pre-trimming) for split-half
                np.array(cfg_raw),
            )

            # Gate 5: Size-adaptive threshold
            passes_threshold = improvement > min_imp

            # All gates
            all_pass = (p_val < P_VALUE_THRESHOLD and
                        delta > CLIFFS_DELTA_THRESHOLD and
                        half_ok and
                        passes_threshold)

            stats = {
                "cfg": cfg_label,
                "auto_mean": round(auto_mean, 2),
                "cfg_mean": round(cfg_mean, 2),
                "improvement": round(improvement * 100, 1),
                "p_value": round(p_val, 6),
                "cliffs_delta": round(delta, 3),
                "split_half": half_ok,
                "threshold": round(min_imp * 100, 0),
                "all_pass": all_pass,
            }

            # Track best config that passes all gates (or best overall for logging)
            if all_pass:
                if best_cfg is None or cfg_mean < best_stats["cfg_mean"]:
                    best_cfg = cfg_label
                    best_stats = stats
            elif best_stats is None or (not best_stats.get("all_pass") and cfg_mean < (best_stats or {}).get("cfg_mean", 999999)):
                if best_stats is None or not best_stats.get("all_pass"):
                    best_stats = stats

        # Log decision
        if node_rank == 0:
            if best_stats and best_stats.get("all_pass"):
                print(f"  {size_label:>8s} (band {band}): OVERRIDE → {best_cfg} "
                      f"[imp={best_stats['improvement']:+.1f}% p={best_stats['p_value']:.4f} "
                      f"δ={best_stats['cliffs_delta']:.2f} split={best_stats['split_half']}]",
                      flush=True)
            elif best_stats:
                gates_failed = []
                if best_stats["p_value"] >= P_VALUE_THRESHOLD:
                    gates_failed.append(f"p={best_stats['p_value']:.4f}≥{P_VALUE_THRESHOLD}")
                if best_stats["cliffs_delta"] <= CLIFFS_DELTA_THRESHOLD:
                    gates_failed.append(f"δ={best_stats['cliffs_delta']:.2f}≤{CLIFFS_DELTA_THRESHOLD}")
                if not best_stats["split_half"]:
                    gates_failed.append("split-half disagrees")
                if best_stats["improvement"] <= min_imp * 100:
                    gates_failed.append(f"imp={best_stats['improvement']:.1f}%≤{min_imp*100:.0f}%")
                print(f"  {size_label:>8s} (band {band}): KEEP AUTO — {best_stats['cfg']} "
                      f"imp={best_stats['improvement']:+.1f}% FAILED: {', '.join(gates_failed)}",
                      flush=True)
            else:
                print(f"  {size_label:>8s} (band {band}): KEEP AUTO — no config faster",
                      flush=True)

        # Record decision
        decision_stats[size_label] = best_stats or {"cfg": "auto", "all_pass": False}

        if best_stats and best_stats.get("all_pass") and band not in policy_details:
            algo_id, proto_id = ALGO_PROTO_MAP[best_cfg]
            policy_entries.append(f"{band}:{algo_id}:{proto_id}")
            policy_details[band] = best_cfg

    policy_str = ",".join(policy_entries)
    if node_rank == 0:
        print(f"  → {mode_label} POLICY: \"{policy_str}\"", flush=True)
    return policy_str, policy_details, decision_stats


def run_validation_on_cluster(mp, node_rank, master_addr, master_port,
                               gpus_per_node, topo_label):
    """Run profile → generate policy → validate on this cluster."""
    results_dir = Path(VOLUME_PATH) / "tuner_validation_v4" / topo_label
    results_dir.mkdir(parents=True, exist_ok=True)

    def run_bench(nccl_env, size_elems, overlap, tag, iters, warmup):
        out_path = str(results_dir / f"times_{tag}.txt")
        try:
            mp.spawn(
                worker_fn,
                args=(gpus_per_node, node_rank, master_addr, master_port,
                      nccl_env, size_elems, overlap, out_path, iters, warmup),
                nprocs=gpus_per_node,
                join=True,
            )
            if node_rank == 0 and Path(out_path).is_file():
                times = [float(x) for x in
                         Path(out_path).read_text().strip().split("\n") if x.strip()]
                if times:
                    times_sorted = sorted(times)
                    return times_sorted[len(times_sorted) // 2]
        except Exception as e:
            if node_rank == 0:
                print(f"ERROR: {str(e)[:300]}", flush=True)
        return 0

    # ── PHASE 1: PROFILE BOTH MODES ──
    if node_rank == 0:
        print(f"\n{'='*60}")
        print(f"  PHASE 1: PROFILING {topo_label} (BOTH modes, {PROFILE_ITERS} iters)")
        print(f"{'='*60}\n", flush=True)

    profile_data = {"sequential": {}, "overlap": {}}
    for mode_label, overlap in [("sequential", False), ("overlap", True)]:
        if node_rank == 0:
            print(f"\n  --- Profiling {mode_label.upper()} mode ---", flush=True)
        for size_label, size_elems in MSG_SIZES:
            size_results = {}
            for cfg_label, cfg_env in PROFILE_CONFIGS:
                if node_rank == 0:
                    print(f"  [PROFILE/{mode_label[:3]}] {cfg_label:15s} @ {size_label} ...",
                          end=" ", flush=True)
                median_t = run_bench(cfg_env, size_elems, overlap,
                                     f"profile_{mode_label}_{size_label}_{cfg_label}",
                                     PROFILE_ITERS, PROFILE_WARMUP)
                if node_rank == 0:
                    size_results[cfg_label] = round(median_t, 3)
                    print(f"median={median_t:.3f}ms", flush=True)
            if node_rank == 0:
                profile_data[mode_label][size_label] = size_results

    # ── PHASE 2: GENERATE PER-MODE POLICIES (confidence-aware) ──
    seq_policy = ""
    ovl_policy = ""
    seq_details = {}
    ovl_details = {}
    seq_decision_stats = {}
    ovl_decision_stats = {}
    if node_rank == 0:
        print(f"\n{'='*60}")
        print(f"  PHASE 2: CONFIDENCE-AWARE POLICY GENERATION for {topo_label}")
        print(f"  Gates: p<{P_VALUE_THRESHOLD} + δ>{CLIFFS_DELTA_THRESHOLD} + "
              f"split-half + imp>{MIN_IMPROVEMENT_SMALL*100:.0f}%/{MIN_IMPROVEMENT_LARGE*100:.0f}%")
        print(f"{'='*60}\n", flush=True)

        print(f"  Sequential mode:", flush=True)
        seq_policy, seq_details, seq_decision_stats = generate_policy_for_mode(
            results_dir, "sequential", node_rank)

        print(f"\n  Overlap mode:", flush=True)
        ovl_policy, ovl_details, ovl_decision_stats = generate_policy_for_mode(
            results_dir, "overlap", node_rank)

    # ── PHASE 3: VALIDATE WITH PER-MODE POLICIES ──
    if node_rank == 0:
        print(f"\n{'='*60}")
        print(f"  PHASE 3: VALIDATION {topo_label} ({VALIDATE_ITERS} iters)")
        print(f"  Sequential policy: \"{seq_policy}\"")
        print(f"  Overlap policy:    \"{ovl_policy}\"")
        print(f"{'='*60}\n", flush=True)

    validation_results = {"sequential": {}, "overlap": {}}
    for mode_label, overlap in [("sequential", False), ("overlap", True)]:
        policy = seq_policy if mode_label == "sequential" else ovl_policy
        validate_configs = [
            ("auto", {}),
            ("tuner", {
                "NCCL_TUNER_PLUGIN": "/repo/libnccl_tuner_v3.so",
                "NCCL_TUNER_POLICY": policy,
            }),
        ]

        if node_rank == 0:
            print(f"\n  --- {mode_label.upper()} (policy: \"{policy}\") ---", flush=True)

        for size_label, size_elems in MSG_SIZES:
            size_data = {}
            for cfg_label, cfg_env in validate_configs:
                if node_rank == 0:
                    print(f"  [{mode_label}] {cfg_label:8s} @ {size_label} ...",
                          end=" ", flush=True)
                median_t = run_bench(cfg_env, size_elems, overlap,
                                     f"val_{mode_label}_{size_label}_{cfg_label}",
                                     VALIDATE_ITERS, VALIDATE_WARMUP)
                if node_rank == 0:
                    size_data[cfg_label] = round(median_t, 3)
                    print(f"median={median_t:.3f}ms", flush=True)

            if node_rank == 0 and size_data:
                validation_results[mode_label][size_label] = size_data
                auto_t = size_data.get("auto", 0)
                tuner_t = size_data.get("tuner", 0)
                if auto_t > 0 and tuner_t > 0:
                    gain = (auto_t - tuner_t) / auto_t * 100
                    winner = "TUNER" if tuner_t < auto_t else "AUTO"
                    print(f"    → {winner}: {gain:+.1f}%", flush=True)

    # ── SAVE ──
    if node_rank == 0:
        output = {
            "topology": topo_label,
            "profile": profile_data,
            "policy_sequential": seq_policy,
            "policy_overlap": ovl_policy,
            "decision_stats": {
                "sequential": seq_decision_stats,
                "overlap": ovl_decision_stats,
            },
            "validation": validation_results,
        }
        (results_dir / "validation_results.json").write_text(
            json.dumps(output, indent=2, default=str))
        volume.commit()

        # Print summary
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {topo_label}")
        print(f"{'='*60}")
        print(f"  Sequential policy: \"{seq_policy}\"")
        print(f"  Overlap policy:    \"{ovl_policy}\"")
        for mode in ("sequential", "overlap"):
            print(f"  {mode}:")
            for size, row in validation_results.get(mode, {}).items():
                auto_t = row.get("auto", 0)
                tuner_t = row.get("tuner", 0)
                if auto_t > 0 and tuner_t > 0:
                    gain = (auto_t - tuner_t) / auto_t * 100
                    print(f"    {size:>6s}: AUTO={auto_t:.1f}ms  TUNER={tuner_t:.1f}ms  "
                          f"gain={gain:+.1f}%")
        print(flush=True)

        return output
    return {}


# ---------------------------------------------------------------------------
# Per-topology Modal functions
# ---------------------------------------------------------------------------
@app.function(
    name="tuner-val-1x8",
    image=tuner_image,
    gpu="A100:8",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
def validate_1x8():
    import torch.multiprocessing as mp
    return run_validation_on_cluster(
        mp, node_rank=0, master_addr="127.0.0.1", master_port=29500,
        gpus_per_node=8, topo_label="1x8",
    )


@app.function(
    name="tuner-val-2x4",
    image=tuner_image,
    gpu="A100:4",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
@modal.experimental.clustered(2)
def validate_2x4():
    import torch.multiprocessing as mp
    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = list(cluster_info.container_ips)[0]
    return run_validation_on_cluster(
        mp, node_rank=node_rank, master_addr=master_addr, master_port=29500,
        gpus_per_node=4, topo_label="2x4",
    )


@app.function(
    name="tuner-val-4x2",
    image=tuner_image,
    gpu="A100:2",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
@modal.experimental.clustered(4)
def validate_4x2():
    import torch.multiprocessing as mp
    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = list(cluster_info.container_ips)[0]
    return run_validation_on_cluster(
        mp, node_rank=node_rank, master_addr=master_addr, master_port=29500,
        gpus_per_node=2, topo_label="4x2",
    )


# ---------------------------------------------------------------------------
# Orchestrator: run topologies sequentially
# ---------------------------------------------------------------------------
@app.function(
    name="tuner-val-orchestrator",
    image=modal.Image.debian_slim(python_version="3.11"),
    timeout=21600,
    volumes={VOLUME_PATH: volume},
)
def orchestrator(config: str = "all"):
    """Run tuner validation sequentially across topologies."""
    import time as _time

    configs = {
        "1x8": validate_1x8,
        "2x4": validate_2x4,
        "4x2": validate_4x2,
    }

    if config == "all":
        run_order = ["1x8", "2x4", "4x2"]
    else:
        run_order = [c.strip() for c in config.split(",") if c.strip() in configs]

    combined = {}
    for topo in run_order:
        print(f"\n{'='*70}")
        print(f"  LAUNCHING TUNER VALIDATION: {topo}")
        print(f"{'='*70}\n", flush=True)

        for attempt in range(2):
            try:
                print(f"  Attempt {attempt+1}/2...", flush=True)
                result = configs[topo].remote()
                combined[topo] = result
                print(f"  {topo} COMPLETE", flush=True)

                # Save incremental progress
                progress_path = Path(VOLUME_PATH) / "tuner_validation_v4" / "combined_validation.json"
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.write_text(json.dumps(combined, indent=2, default=str))
                volume.commit()
                break
            except Exception as e:
                print(f"  {topo} FAILED (attempt {attempt+1}): {str(e)[:200]}", flush=True)
                if attempt < 1:
                    print(f"  Retrying in 10s...", flush=True)
                    _time.sleep(10)

    print(f"\n{'='*70}")
    print(f"  ALL VALIDATIONS COMPLETE")
    print(f"  Topologies: {list(combined.keys())}")
    print(f"{'='*70}", flush=True)

    return combined


@app.local_entrypoint()
def main(config: str = "all"):
    print(f"Profile-Guided Tuner Validation v4 (confidence-aware)")
    print(f"  Topologies: {config}")
    print(f"  Per topology: Profile ({PROFILE_ITERS} iters) → Confidence-Aware Policy → Validate ({VALIDATE_ITERS} iters)")
    print(f"  Decision gates: p<{P_VALUE_THRESHOLD} + δ>{CLIFFS_DELTA_THRESHOLD} + split-half + imp threshold")
    print(f"  Message sizes: {', '.join(s[0] for s in MSG_SIZES)}")
    print(f"  Modes: sequential + overlap")
    print()
    orchestrator.remote(config=config)
