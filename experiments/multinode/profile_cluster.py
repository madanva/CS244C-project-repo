"""
Profile-Guided NCCL Tuner — Cluster Profiler

This is the core of the tuner pipeline (matches the paper figure):

  1. Cluster Assigned  →  auto-detect topology (nodes, GPUs/node)
  2. Profile Cluster   →  benchmark all (algo, proto) combos per message size
  3. Generate Policy   →  output NCCL_TUNER_POLICY string for the C plugin

Usage on Modal (any topology — auto-detects):
    modal run -d profile_cluster.py

Usage on bare metal:
    torchrun --nproc_per_node=N profile_cluster.py

The generated policy string is passed to the v3 tuner plugin:
    NCCL_TUNER_PLUGIN=./libnccl_tuner_v3.so \\
    NCCL_TUNER_POLICY="4:0:2,5:0:2,6:0:2" \\
    torchrun train.py

Profiling cost: ~2 minutes for 6 sizes × 7 configs × 20 iterations.
This runs ONCE per cluster assignment, amortized over training.
"""

import json
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MSG_SIZES = [
    ("256KB",  65_536),     # 256KB of float32
    ("1MB",    262_144),
    ("4MB",    1_048_576),
    ("16MB",   4_194_304),
    ("64MB",   16_777_216),
    ("256MB",  67_108_864),
]

PROFILE_CONFIGS = [
    ("auto",         {}),
    ("tree_simple",  {"NCCL_ALGO": "Tree", "NCCL_PROTO": "Simple"}),
    ("tree_ll",      {"NCCL_ALGO": "Tree", "NCCL_PROTO": "LL"}),
    ("tree_ll128",   {"NCCL_ALGO": "Tree", "NCCL_PROTO": "LL128"}),
    ("ring_simple",  {"NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple"}),
    ("ring_ll",      {"NCCL_ALGO": "Ring", "NCCL_PROTO": "LL"}),
    ("ring_ll128",   {"NCCL_ALGO": "Ring", "NCCL_PROTO": "LL128"}),
]

ALGO_PROTO_IDS = {
    "tree_simple":  (0, 2),   # NCCL_ALGO_TREE=0, NCCL_PROTO_SIMPLE=2
    "tree_ll":      (0, 0),   # NCCL_ALGO_TREE=0, NCCL_PROTO_LL=0
    "tree_ll128":   (0, 1),   # NCCL_ALGO_TREE=0, NCCL_PROTO_LL128=1
    "ring_simple":  (1, 2),   # NCCL_ALGO_RING=1, NCCL_PROTO_SIMPLE=2
    "ring_ll":      (1, 0),   # NCCL_ALGO_RING=1, NCCL_PROTO_LL=0
    "ring_ll128":   (1, 1),   # NCCL_ALGO_RING=1, NCCL_PROTO_LL128=1
}

PROFILE_ITERS = 20
WARMUP_ITERS = 5
MIN_IMPROVEMENT = 0.02  # 2% minimum to include in policy


def compute_band(nBytes):
    """Map byte count to NCCL tuner size band (must match tuner_v3.c)."""
    if nBytes < 1024:           return 0
    if nBytes < 256 * 1024:     return 1
    if nBytes < 1024 * 1024:    return 2
    if nBytes < 4 * 1024**2:    return 3
    if nBytes < 32 * 1024**2:   return 4
    if nBytes < 128 * 1024**2:  return 5
    return 6


# ---------------------------------------------------------------------------
# Step 1: Auto-detect cluster topology
# ---------------------------------------------------------------------------
def detect_topology():
    """
    Detect cluster topology from environment.
    Returns (n_nodes, gpus_per_node, world_size, node_rank, master_addr, master_port).

    Works with:
      - Modal @clustered() — reads cluster_info
      - torchrun — reads WORLD_SIZE, LOCAL_WORLD_SIZE, RANK
      - Single node — detects available GPUs
    """
    import torch

    # Try Modal cluster info first
    try:
        import modal.experimental
        cluster_info = modal.experimental.get_cluster_info()
        container_ips = list(cluster_info.container_ips)
        n_nodes = len(container_ips)
        node_rank = cluster_info.rank
        master_addr = container_ips[0]
        gpus_per_node = torch.cuda.device_count()
        world_size = n_nodes * gpus_per_node
        return n_nodes, gpus_per_node, world_size, node_rank, master_addr, 29500
    except Exception:
        pass

    # Try torchrun environment
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE",
                                               torch.cuda.device_count()))
        n_nodes = world_size // local_world_size
        node_rank = int(os.environ.get("GROUP_RANK", "0"))
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        return n_nodes, local_world_size, world_size, node_rank, master_addr, master_port

    # Fallback: single node
    gpus = torch.cuda.device_count()
    return 1, gpus, gpus, 0, "127.0.0.1", 29500


# ---------------------------------------------------------------------------
# Step 2: Profile cluster
# ---------------------------------------------------------------------------
def worker_fn(local_rank, gpus_per_node, world_size, node_rank,
              master_addr, master_port, nccl_env, size_elems,
              out_path, iters, warmup):
    """Worker: run AllReduce benchmark on a single GPU."""
    import torch
    import torch.distributed as dist

    global_rank = node_rank * gpus_per_node + local_rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Set NCCL env
    for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_TUNER_PLUGIN",
                "NCCL_TUNER_POLICY"):
        os.environ.pop(key, None)
    for k, v in nccl_env.items():
        os.environ[k] = v
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    dist.init_process_group(backend="nccl", rank=global_rank,
                            world_size=world_size)

    data = torch.randn(size_elems, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        tmp = data.clone()
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

    # Timed iterations
    times_ms = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tmp = data.clone()
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    if global_rank == 0 and out_path:
        Path(out_path).write_text("\n".join(f"{t:.3f}" for t in times_ms) + "\n")

    dist.destroy_process_group()


def profile_cluster(n_nodes, gpus_per_node, world_size, node_rank,
                    master_addr, master_port, output_dir):
    """
    Phase 2: Profile all (algo, proto) × message sizes on THIS cluster.
    Returns profile_data dict on rank 0.
    """
    import torch.multiprocessing as mp

    topo_label = f"{n_nodes}x{gpus_per_node}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if node_rank == 0:
        print(f"\n{'='*60}")
        print(f"  PROFILING CLUSTER: {topo_label}")
        print(f"  {n_nodes} nodes × {gpus_per_node} GPUs = {world_size} total")
        print(f"  {len(MSG_SIZES)} sizes × {len(PROFILE_CONFIGS)} configs × {PROFILE_ITERS} iters")
        print(f"{'='*60}\n", flush=True)

    profile_data = {}
    profile_start = time.time()

    for size_label, size_elems in MSG_SIZES:
        size_results = {}
        for cfg_label, cfg_env in PROFILE_CONFIGS:
            tag = f"profile_{size_label}_{cfg_label}"
            out_path = str(output_dir / f"times_{tag}.txt")

            if node_rank == 0:
                print(f"  {cfg_label:15s} @ {size_label:>6s} ...", end=" ", flush=True)

            try:
                mp.spawn(
                    worker_fn,
                    args=(gpus_per_node, world_size, node_rank, master_addr,
                          master_port, cfg_env, size_elems, out_path,
                          PROFILE_ITERS, WARMUP_ITERS),
                    nprocs=gpus_per_node,
                    join=True,
                )

                if node_rank == 0 and Path(out_path).is_file():
                    times = [float(x) for x in
                             Path(out_path).read_text().strip().split("\n")
                             if x.strip()]
                    if times:
                        median = sorted(times)[len(times) // 2]
                        size_results[cfg_label] = round(median, 3)
                        print(f"{median:.3f}ms", flush=True)
                    else:
                        print("NO DATA", flush=True)
                elif node_rank == 0:
                    print("NO FILE", flush=True)
            except Exception as e:
                if node_rank == 0:
                    size_results[cfg_label] = 0
                    print(f"ERROR: {str(e)[:100]}", flush=True)

        if node_rank == 0:
            profile_data[size_label] = size_results

            # Show best for this size
            auto_t = size_results.get("auto", 0)
            if auto_t > 0:
                best_cfg = min(
                    [c for c in ALGO_PROTO_IDS if size_results.get(c, 0) > 0],
                    key=lambda c: size_results[c],
                    default=None
                )
                if best_cfg and size_results[best_cfg] < auto_t:
                    gain = (auto_t - size_results[best_cfg]) / auto_t * 100
                    print(f"    → best: {best_cfg} ({gain:+.1f}% vs AUTO)\n", flush=True)
                else:
                    print(f"    → AUTO is optimal\n", flush=True)

    profile_time = time.time() - profile_start
    if node_rank == 0:
        print(f"  Profiling completed in {profile_time:.1f}s\n", flush=True)

    return profile_data, profile_time


# ---------------------------------------------------------------------------
# Step 3: Generate policy
# ---------------------------------------------------------------------------
def generate_policy(profile_data):
    """
    Phase 3: Generate NCCL_TUNER_POLICY string from profiling results.
    Returns (policy_string, policy_details).
    """
    print(f"\n{'='*60}")
    print(f"  GENERATING POLICY")
    print(f"{'='*60}\n", flush=True)

    policy_entries = []
    policy_details = {}

    for size_label, size_elems in MSG_SIZES:
        nBytes = size_elems * 4  # float32
        band = compute_band(nBytes)
        results = profile_data.get(size_label, {})
        auto_t = results.get("auto", 0)

        if auto_t <= 0:
            continue

        # Find best non-auto config
        best_cfg = None
        best_time = auto_t
        for cfg_label in ALGO_PROTO_IDS:
            t = results.get(cfg_label, 0)
            if t > 0 and t < best_time:
                best_time = t
                best_cfg = cfg_label

        improvement = (auto_t - best_time) / auto_t if best_cfg else 0
        include = improvement > MIN_IMPROVEMENT

        status = f"→ {best_cfg}" if include else "→ AUTO (optimal)"
        print(f"  {size_label:>6s} (band {band}): "
              f"auto={auto_t:.1f}ms, best={best_cfg or 'auto'}={best_time:.1f}ms, "
              f"gain={improvement*100:+.1f}% {status}", flush=True)

        if include and band not in policy_details:
            algo_id, proto_id = ALGO_PROTO_IDS[best_cfg]
            policy_entries.append(f"{band}:{algo_id}:{proto_id}")
            policy_details[band] = {
                "size": size_label,
                "config": best_cfg,
                "improvement": round(improvement * 100, 1),
                "algo_id": algo_id,
                "proto_id": proto_id,
            }

    policy_str = ",".join(policy_entries) if policy_entries else ""

    print(f"\n  ┌────────────────────────────────────────────┐")
    print(f"  │  NCCL_TUNER_POLICY=\"{policy_str}\"")
    print(f"  └────────────────────────────────────────────┘")
    print(f"  Active bands: {len(policy_details)}/{len(MSG_SIZES)}")
    for band, info in sorted(policy_details.items()):
        print(f"    band {band} ({info['size']}): {info['config']} "
              f"(+{info['improvement']}%)")
    print(flush=True)

    return policy_str, policy_details


# ---------------------------------------------------------------------------
# Modal integration (auto-detect topology via @clustered or single-node)
# ---------------------------------------------------------------------------
try:
    import modal
    import modal.experimental

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent

    profiler_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.6.0-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("wget", "libibverbs-dev", "libibverbs1")
        .run_commands("pip install --upgrade pip")
        .pip_install("torch", "numpy")
        .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
        .run_commands(
            "cd /repo/phase4-tuner && "
            "gcc -shared -fPIC -o /repo/libnccl_tuner_v3.so "
            "workload_aware_tuner_v3.c -I. && "
            "echo 'Tuner compiled' && ls -la /repo/libnccl_tuner_v3.so"
        )
    )

    volume = modal.Volume.from_name("browser-networking-tests-storage",
                                     create_if_missing=True)
    VOLUME_PATH = "/results"

    modal_app = modal.App("browser-networking-tests")

    # --- Single node (1×8) ---
    @modal_app.function(
        name="profiler-1x8",
        image=profiler_image,
        gpu="A100:8",
        timeout=3600,
        volumes={VOLUME_PATH: volume},
    )
    def profile_1x8():
        topo = detect_topology()  # auto-detects 1 node, 8 GPUs
        n_nodes, gpus_per_node, world_size, node_rank, master_addr, port = topo
        label = f"{n_nodes}x{gpus_per_node}"
        out_dir = Path(VOLUME_PATH) / "tuner_profiling" / label

        profile_data, elapsed = profile_cluster(
            n_nodes, gpus_per_node, world_size, node_rank,
            master_addr, port, out_dir)

        policy_str, details = generate_policy(profile_data)

        result = {
            "topology": label,
            "profile": profile_data,
            "policy": policy_str,
            "policy_details": details,
            "profiling_time_s": round(elapsed, 1),
        }
        (out_dir / "profiler_output.json").write_text(json.dumps(result, indent=2))
        volume.commit()
        return result

    # --- 2 nodes (2×4) ---
    @modal_app.function(
        name="profiler-2x4",
        image=profiler_image,
        gpu="A100:4",
        timeout=3600,
        volumes={VOLUME_PATH: volume},
    )
    @modal.experimental.clustered(2)
    def profile_2x4():
        topo = detect_topology()
        n_nodes, gpus_per_node, world_size, node_rank, master_addr, port = topo
        label = f"{n_nodes}x{gpus_per_node}"
        out_dir = Path(VOLUME_PATH) / "tuner_profiling" / label

        profile_data, elapsed = profile_cluster(
            n_nodes, gpus_per_node, world_size, node_rank,
            master_addr, port, out_dir)

        if node_rank == 0:
            policy_str, details = generate_policy(profile_data)
            result = {
                "topology": label,
                "profile": profile_data,
                "policy": policy_str,
                "policy_details": details,
                "profiling_time_s": round(elapsed, 1),
            }
            (out_dir / "profiler_output.json").write_text(json.dumps(result, indent=2))
            volume.commit()
            return result
        return {}

    # --- 4 nodes (4×2) ---
    @modal_app.function(
        name="profiler-4x2",
        image=profiler_image,
        gpu="A100:2",
        timeout=3600,
        volumes={VOLUME_PATH: volume},
    )
    @modal.experimental.clustered(4)
    def profile_4x2():
        topo = detect_topology()
        n_nodes, gpus_per_node, world_size, node_rank, master_addr, port = topo
        label = f"{n_nodes}x{gpus_per_node}"
        out_dir = Path(VOLUME_PATH) / "tuner_profiling" / label

        profile_data, elapsed = profile_cluster(
            n_nodes, gpus_per_node, world_size, node_rank,
            master_addr, port, out_dir)

        if node_rank == 0:
            policy_str, details = generate_policy(profile_data)
            result = {
                "topology": label,
                "profile": profile_data,
                "policy": policy_str,
                "policy_details": details,
                "profiling_time_s": round(elapsed, 1),
            }
            (out_dir / "profiler_output.json").write_text(json.dumps(result, indent=2))
            volume.commit()
            return result
        return {}

    # --- Orchestrator ---
    @modal_app.function(
        name="profiler-orchestrator",
        image=modal.Image.debian_slim(python_version="3.11"),
        timeout=10800,
        volumes={VOLUME_PATH: volume},
    )
    def profiler_orchestrator(config: str = "all"):
        """Run profiler across topologies sequentially."""
        import time as _time

        funcs = {"1x8": profile_1x8, "2x4": profile_2x4, "4x2": profile_4x2}

        if config == "all":
            run_order = ["1x8", "2x4", "4x2"]
        else:
            run_order = [c.strip() for c in config.split(",") if c.strip() in funcs]

        all_results = {}
        for topo in run_order:
            print(f"\n{'='*60}")
            print(f"  PROFILING {topo}")
            print(f"{'='*60}\n", flush=True)

            for attempt in range(2):
                try:
                    result = funcs[topo].remote()
                    all_results[topo] = result
                    policy = result.get("policy", "")
                    elapsed = result.get("profiling_time_s", 0)
                    print(f"  {topo} DONE in {elapsed}s")
                    print(f"  Policy: NCCL_TUNER_POLICY=\"{policy}\"", flush=True)

                    # Save progress
                    out = Path(VOLUME_PATH) / "tuner_profiling" / "all_policies.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(all_results, indent=2, default=str))
                    volume.commit()
                    break
                except Exception as e:
                    print(f"  {topo} FAILED: {str(e)[:200]}", flush=True)
                    if attempt < 1:
                        _time.sleep(10)

        # Summary
        print(f"\n{'='*60}")
        print(f"  PROFILING COMPLETE — GENERATED POLICIES")
        print(f"{'='*60}")
        for topo, result in all_results.items():
            policy = result.get("policy", "")
            elapsed = result.get("profiling_time_s", 0)
            n_bands = len(result.get("policy_details", {}))
            print(f"  {topo}: NCCL_TUNER_POLICY=\"{policy}\"  "
                  f"({n_bands} bands, {elapsed}s)")
        print(f"\n  Usage:")
        print(f"    NCCL_TUNER_PLUGIN=./libnccl_tuner_v3.so \\")
        print(f"    NCCL_TUNER_POLICY=\"<policy from above>\" \\")
        print(f"    torchrun train.py")
        print(flush=True)

        return all_results

    @modal_app.local_entrypoint()
    def main(config: str = "all"):
        print(f"Profile-Guided NCCL Tuner — Cluster Profiler")
        print(f"  Topologies: {config}")
        print(f"  Per topology: {len(MSG_SIZES)} sizes × {len(PROFILE_CONFIGS)} configs × {PROFILE_ITERS} iters")
        print()
        profiler_orchestrator.remote(config=config)

except ImportError:
    # Non-Modal fallback: use torchrun or bare metal
    pass


# ---------------------------------------------------------------------------
# Standalone entry point (non-Modal)
# ---------------------------------------------------------------------------
if __name__ == "__main__" and "MODAL_" not in "".join(os.environ.keys()):
    print("Profile-Guided NCCL Tuner — Standalone Mode")
    topo = detect_topology()
    n_nodes, gpus_per_node, world_size, node_rank, master_addr, port = topo
    label = f"{n_nodes}x{gpus_per_node}"

    print(f"  Detected: {label} ({n_nodes} nodes × {gpus_per_node} GPUs)")

    out_dir = Path(f"./profiler_output_{label}")
    profile_data, elapsed = profile_cluster(
        n_nodes, gpus_per_node, world_size, node_rank,
        master_addr, port, out_dir)

    if node_rank == 0:
        policy_str, details = generate_policy(profile_data)

        result = {
            "topology": label,
            "profile": profile_data,
            "policy": policy_str,
            "policy_details": details,
            "profiling_time_s": round(elapsed, 1),
        }
        (out_dir / "profiler_output.json").write_text(json.dumps(result, indent=2))

        print(f"\n  To use this policy during training:")
        print(f"    NCCL_TUNER_PLUGIN=./libnccl_tuner_v3.so \\")
        print(f"    NCCL_TUNER_POLICY=\"{policy_str}\" \\")
        print(f"    torchrun --nproc_per_node={gpus_per_node} train.py")
