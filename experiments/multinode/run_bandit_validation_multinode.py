"""
Modal app: Online RL bandit tuner validation across multi-node topologies.

For each topology (2×4, 4×2):
  For each key message size (64MB, 256MB):
    Phase A — AUTO baseline: 200 iterations, no tuner plugin.
    Phase B — BANDIT: 200 iterations with rl_bandit_tuner_v2.so.
              First M×4 iters = exploration (round-robin), rest = exploitation.

The bandit learns online by reading rewards logged per-iteration.
Skips 1×8 (NVLink-dominated, nothing to optimize).

Uses orchestrator pattern to run topologies sequentially (max 10 GPUs).
"""

import json
import os
import time
from pathlib import Path

import modal
import modal.experimental

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Image: CUDA + PyTorch + compiled BOTH tuner plugins
bandit_image = (
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
        "gcc -shared -fPIC -o /repo/libnccl_bandit_v2.so "
        "rl_bandit_tuner_v2.c -I. -lm && "
        "echo 'Both plugins compiled successfully' && "
        "ls -la /repo/libnccl_tuner_v3.so /repo/libnccl_bandit_v2.so"
    )
)

volume = modal.Volume.from_name("browser-networking-tests-storage", create_if_missing=True)
VOLUME_PATH = "/results"

app = modal.App("browser-networking-tests")

WORLD_SIZE = 8

# Key sizes where variance is highest and v3 showed wins/regressions
KEY_SIZES = [
    ("64MB",   16_777_216),
    ("256MB",  67_108_864),
]

BANDIT_ITERS = 240
BANDIT_WARMUP = 15
COMPUTE_MUL = 4096
EXPLORE_ROUNDS = 10  # 10 rounds × 4 arms = 40 exploration iterations → 200 exploit iterations


# ---------------------------------------------------------------------------
# Worker: runs N iterations within a SINGLE process group (bandit needs this)
# ---------------------------------------------------------------------------
def bandit_worker_fn(local_rank, gpus_per_node, node_rank, master_addr, master_port,
                     nccl_env, size_elems, out_path, reward_path, iters, warmup):
    """Single GPU worker for bandit validation. Keeps process group alive."""
    import torch
    import torch.distributed as dist

    global_rank = node_rank * gpus_per_node + local_rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Set NCCL env
    for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_TUNER_PLUGIN",
                "NCCL_TUNER_POLICY", "NCCL_TUNER_REWARD_FILE",
                "NCCL_TUNER_RANK", "NCCL_TUNER_EXPLORE_ROUNDS",
                "NCCL_TUNER_LOG"):
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
    nBytes = size_elems * 4
    num_nodes = int(os.environ.get("NCCL_TUNER_NODES", 1))

    # Sequential mode only (simpler, cleaner signal)
    # Warmup
    for _ in range(warmup):
        a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
        b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        tmp = grad.clone()
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

    # Timed iterations
    times_ms = []
    for i in range(iters):
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
        elapsed = (t1 - t0) * 1000.0
        times_ms.append(elapsed)

        # Write reward for bandit (rank 0 only)
        if global_rank == 0 and reward_path:
            with open(reward_path, "a") as rf:
                rf.write(f"allreduce,{nBytes},{num_nodes},{WORLD_SIZE},{elapsed:.3f}\n")

    # Save all timings
    if global_rank == 0 and out_path:
        Path(out_path).write_text("\n".join(f"{t:.3f}" for t in times_ms) + "\n")

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Core: run bandit validation on a given cluster
# ---------------------------------------------------------------------------
def run_bandit_on_cluster(mp, node_rank, master_addr, master_port,
                          gpus_per_node, topo_label):
    """Run AUTO baseline + bandit for each key size."""
    results_dir = Path(VOLUME_PATH) / "bandit_validation" / topo_label
    results_dir.mkdir(parents=True, exist_ok=True)

    num_nodes = WORLD_SIZE // gpus_per_node
    all_results = {}

    for size_label, size_elems in KEY_SIZES:
        if node_rank == 0:
            print(f"\n{'='*60}")
            print(f"  {topo_label} @ {size_label}: AUTO baseline ({BANDIT_ITERS} iters)")
            print(f"{'='*60}", flush=True)

        # Phase A: AUTO baseline
        auto_path = str(results_dir / f"times_auto_{size_label}.txt")
        try:
            mp.spawn(
                bandit_worker_fn,
                args=(gpus_per_node, node_rank, master_addr, master_port,
                      {},  # no tuner
                      size_elems, auto_path, None, BANDIT_ITERS, BANDIT_WARMUP),
                nprocs=gpus_per_node,
                join=True,
            )
        except Exception as e:
            if node_rank == 0:
                print(f"  AUTO ERROR: {str(e)[:300]}", flush=True)

        # Phase B: Bandit
        if node_rank == 0:
            print(f"\n{'='*60}")
            print(f"  {topo_label} @ {size_label}: BANDIT ({BANDIT_ITERS} iters, "
                  f"explore={EXPLORE_ROUNDS}×4={EXPLORE_ROUNDS*4} iters)")
            print(f"{'='*60}", flush=True)

        bandit_path = str(results_dir / f"times_bandit_{size_label}.txt")
        reward_path = str(results_dir / f"rewards_{size_label}.log")

        # Clear old reward file
        if node_rank == 0:
            Path(reward_path).unlink(missing_ok=True)

        bandit_env = {
            "NCCL_TUNER_PLUGIN": "/repo/libnccl_bandit_v2.so",
            "NCCL_TUNER_REWARD_FILE": reward_path,
            "NCCL_TUNER_RANK": str(0 if node_rank == 0 else 1),
            "NCCL_TUNER_EXPLORE_ROUNDS": str(EXPLORE_ROUNDS),
            "NCCL_TUNER_NODES": str(num_nodes),
            "NCCL_TUNER_LOG": "1",
        }

        try:
            mp.spawn(
                bandit_worker_fn,
                args=(gpus_per_node, node_rank, master_addr, master_port,
                      bandit_env,
                      size_elems, bandit_path, reward_path, BANDIT_ITERS, BANDIT_WARMUP),
                nprocs=gpus_per_node,
                join=True,
            )
        except Exception as e:
            if node_rank == 0:
                print(f"  BANDIT ERROR: {str(e)[:300]}", flush=True)

        # Compute results
        if node_rank == 0:
            size_results = {}
            for label, path in [("auto", auto_path), ("bandit", bandit_path)]:
                if Path(path).is_file():
                    vals = [float(x) for x in Path(path).read_text().strip().split("\n") if x.strip()]
                    if vals:
                        import numpy as np
                        arr = np.array(vals)
                        # Use last 100 iterations for fair comparison (bandit fully exploiting)
                        exploit_vals = arr[EXPLORE_ROUNDS * 4:] if label == "bandit" and len(arr) > EXPLORE_ROUNDS * 4 else arr
                        size_results[label] = {
                            "median_all": round(float(np.median(arr)), 3),
                            "median_exploit": round(float(np.median(exploit_vals)), 3),
                            "mean_exploit": round(float(np.mean(exploit_vals)), 3),
                            "std_exploit": round(float(np.std(exploit_vals)), 3),
                            "n_total": len(arr),
                            "n_exploit": len(exploit_vals),
                        }

            if "auto" in size_results and "bandit" in size_results:
                auto_med = size_results["auto"]["median_exploit"]
                bandit_med = size_results["bandit"]["median_exploit"]
                gain = (auto_med - bandit_med) / auto_med * 100 if auto_med > 0 else 0
                size_results["speedup_pct"] = round(gain, 1)

                # Safety gate: bandit must beat AUTO by ≥5% to be considered a win
                SAFETY_THRESHOLD = 5.0
                if gain >= SAFETY_THRESHOLD:
                    size_results["decision"] = "BANDIT"
                    print(f"\n  {size_label}: AUTO={auto_med:.1f}ms BANDIT={bandit_med:.1f}ms "
                          f"→ BANDIT WINS ({gain:+.1f}%, passed {SAFETY_THRESHOLD}% gate)",
                          flush=True)
                elif gain > 0:
                    size_results["decision"] = "AUTO (marginal)"
                    print(f"\n  {size_label}: AUTO={auto_med:.1f}ms BANDIT={bandit_med:.1f}ms "
                          f"→ KEEP AUTO (gain {gain:+.1f}% below {SAFETY_THRESHOLD}% gate)",
                          flush=True)
                else:
                    size_results["decision"] = "AUTO"
                    print(f"\n  {size_label}: AUTO={auto_med:.1f}ms BANDIT={bandit_med:.1f}ms "
                          f"→ KEEP AUTO (bandit worse: {gain:+.1f}%)",
                          flush=True)

            all_results[size_label] = size_results

    # Save combined results
    if node_rank == 0:
        output = {
            "topology": topo_label,
            "bandit_iters": BANDIT_ITERS,
            "explore_rounds": EXPLORE_ROUNDS,
            "explore_iters": EXPLORE_ROUNDS * 4,
            "results": all_results,
        }
        (results_dir / "bandit_results.json").write_text(json.dumps(output, indent=2))
        volume.commit()

        print(f"\n{'='*60}")
        print(f"  BANDIT SUMMARY: {topo_label}")
        print(f"{'='*60}")
        for size_label, sr in all_results.items():
            speedup = sr.get("speedup_pct", 0)
            print(f"  {size_label}: {speedup:+.1f}%")
        print(flush=True)

        return output
    return {}


# ---------------------------------------------------------------------------
# Per-topology Modal functions
# ---------------------------------------------------------------------------
@app.function(
    name="bandit-val-2x4",
    image=bandit_image,
    gpu="A100:4",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
@modal.experimental.clustered(2)
def bandit_2x4():
    import torch.multiprocessing as mp
    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = list(cluster_info.container_ips)[0]
    return run_bandit_on_cluster(
        mp, node_rank=node_rank, master_addr=master_addr, master_port=29500,
        gpus_per_node=4, topo_label="2x4",
    )


@app.function(
    name="bandit-val-4x2",
    image=bandit_image,
    gpu="A100:2",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
@modal.experimental.clustered(4)
def bandit_4x2():
    import torch.multiprocessing as mp
    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = list(cluster_info.container_ips)[0]
    return run_bandit_on_cluster(
        mp, node_rank=node_rank, master_addr=master_addr, master_port=29500,
        gpus_per_node=2, topo_label="4x2",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@app.function(
    name="bandit-val-orchestrator",
    image=modal.Image.debian_slim(python_version="3.11"),
    timeout=21600,
    volumes={VOLUME_PATH: volume},
)
def orchestrator(config: str = "all"):
    """Run bandit validation sequentially across topologies."""
    import time as _time

    configs = {
        "2x4": bandit_2x4,
        "4x2": bandit_4x2,
    }

    if config == "all":
        run_order = ["2x4", "4x2"]
    else:
        run_order = [c.strip() for c in config.split(",") if c.strip() in configs]

    combined = {}
    for topo in run_order:
        print(f"\n{'='*70}")
        print(f"  LAUNCHING BANDIT VALIDATION: {topo}")
        print(f"{'='*70}\n", flush=True)

        for attempt in range(2):
            try:
                print(f"  Attempt {attempt+1}/2...", flush=True)
                result = configs[topo].remote()
                combined[topo] = result
                print(f"  {topo} COMPLETE", flush=True)

                progress_path = Path(VOLUME_PATH) / "bandit_validation" / "combined_bandit.json"
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
    print(f"  ALL BANDIT VALIDATIONS COMPLETE")
    print(f"  Topologies: {list(combined.keys())}")
    print(f"{'='*70}", flush=True)

    return combined


@app.local_entrypoint()
def main(config: str = "all"):
    print(f"Online RL Bandit Tuner Validation")
    print(f"  Topologies: {config}")
    print(f"  Per topology per size: AUTO ({BANDIT_ITERS} iters) + Bandit ({BANDIT_ITERS} iters, {BANDIT_ITERS - EXPLORE_ROUNDS*4} exploit)")
    print(f"  Bandit: {EXPLORE_ROUNDS} explore rounds × 4 arms = {EXPLORE_ROUNDS * 4} explore iters")
    print(f"  Safety gate: bandit must beat AUTO by ≥5% to be accepted")
    print(f"  Key sizes: {', '.join(s[0] for s in KEY_SIZES)}")
    print(f"  Mode: sequential only")
    print()
    orchestrator.remote(config=config)
