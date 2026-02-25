"""
Modal app: Validate the workload-aware tuner v2 against NCCL AUTO.

Emulates run_modal_multinode.py exactly, but with two configs:
  1. auto — no plugin (NCCL AUTO baseline)
  2. tuner_v3 — our tuner plugin loaded via NCCL_TUNER_PLUGIN

2 nodes × 4 GPUs = 8 total.
"""

import json
import os
import time
from pathlib import Path

import modal
import modal.experimental

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Image: same as multinode experiment, plus compile the tuner
validation_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget", "libibverbs-dev", "libibverbs1")
    .run_commands("pip install --upgrade pip")
    .pip_install("torch")
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
    # Compile the tuner plugin during image build
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

N_NODES = 2
GPUS_PER_NODE = 4   # 2×4 = 8 total GPUs across nodes
WORLD_SIZE = N_NODES * GPUS_PER_NODE

MSG_SIZES = [
    ("256KB",  65_536),
    ("1MB",    262_144),
    ("4MB",    1_048_576),
    ("16MB",   4_194_304),
    ("64MB",   16_777_216),
    ("256MB",  67_108_864),
]

# Two configs: AUTO baseline vs our tuner plugin
CONFIGS = [
    ("auto",      {}),
    ("tuner_v3",  {"NCCL_TUNER_PLUGIN": "/repo/libnccl_tuner_v3.so"}),
]

ITERS = 50
WARMUP = 10
COMPUTE_MUL = 4096


# ---------------------------------------------------------------------------
# Worker function: runs on each GPU (identical to run_modal_multinode.py)
# ---------------------------------------------------------------------------
def worker_fn(local_rank, node_rank, master_addr, master_port,
              nccl_env, size_elems, overlap, out_path):
    """Single GPU worker: compute + allreduce iteration proxy."""
    import torch
    import torch.distributed as dist

    global_rank = node_rank * GPUS_PER_NODE + local_rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Set NCCL env vars for this config
    for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_TUNER_PLUGIN", "NCCL_TUNER_REWARD_FILE"):
        os.environ.pop(key, None)
    for k, v in nccl_env.items():
        os.environ[k] = v
    os.environ["NCCL_DEBUG"] = "WARN"

    # Use env:// init with MASTER_ADDR/PORT
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    os.environ["LOCAL_RANK"] = str(local_rank)

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=WORLD_SIZE)

    grad = torch.randn(size_elems, device=device, dtype=torch.float32) / WORLD_SIZE

    if not overlap:
        # Warmup
        for _ in range(WARMUP):
            a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            tmp = grad.clone()
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

        # Timed
        times_ms = []
        for _ in range(ITERS):
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

        # Warmup
        for _ in range(WARMUP):
            with torch.cuda.stream(compute_stream):
                a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                _ = torch.matmul(a, b)
            with torch.cuda.stream(comm_stream):
                tmp = grad.clone()
                dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

        # Timed
        times_ms = []
        for _ in range(ITERS):
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
        out_lines = [f"{t:.3f}" for t in times_ms]
        Path(out_path).write_text("\n".join(out_lines) + "\n")

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main Modal function (multi-node) — same structure as run_modal_multinode.py
# ---------------------------------------------------------------------------
@app.function(
    name="browser-networking-tests",
    image=validation_image,
    gpu=f"A100:{GPUS_PER_NODE}",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
@modal.experimental.clustered(N_NODES)
def run_tuner_validation():
    """Multi-node: AUTO vs tuner across sizes and modes."""
    import torch.multiprocessing as mp

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = cluster_info.container_ips[0]
    master_port = 29500

    results_dir = Path(VOLUME_PATH) / "tuner_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"sequential": {}, "overlap": {}}

    if node_rank == 0:
        print(f"\n{'='*70}")
        print(f"  TUNER VALIDATION — MULTI-NODE")
        print(f"  {N_NODES} nodes × {GPUS_PER_NODE} GPUs = {WORLD_SIZE} total")
        print(f"  Master: {master_addr}:{master_port}")
        print(f"  All container IPs: {cluster_info.container_ips}")
        print(f"  Message sizes: {[s[0] for s in MSG_SIZES]}")
        print(f"  Configs: {[c[0] for c in CONFIGS]}")
        print(f"{'='*70}\n", flush=True)

    for mode_label, overlap in [("sequential", False), ("overlap", True)]:
        if node_rank == 0:
            print(f"\n{'='*60}")
            print(f"  MODE: {mode_label.upper()}")
            print(f"{'='*60}\n", flush=True)

        for size_label, size_elems in MSG_SIZES:
            size_data = {}

            for cfg_label, cfg_env in CONFIGS:
                tag = f"val_{mode_label}_{size_label}_{cfg_label}"
                out_path = str(results_dir / f"times_{tag}.txt")

                if node_rank == 0:
                    print(f"  [{mode_label.upper()}] {cfg_label:15s} @ {size_label} ...",
                          end=" ", flush=True)

                try:
                    mp.spawn(
                        worker_fn,
                        args=(node_rank, master_addr, master_port,
                              cfg_env, size_elems, overlap, out_path),
                        nprocs=GPUS_PER_NODE,
                        join=True,
                    )

                    if node_rank == 0 and Path(out_path).is_file():
                        times = [float(x) for x in
                                 Path(out_path).read_text().strip().split("\n")
                                 if x.strip()]
                        if times:
                            times_sorted = sorted(times)
                            median_t = times_sorted[len(times_sorted) // 2]
                            mean_t = sum(times) / len(times)
                            size_data[cfg_label] = round(median_t, 3)
                            print(f"median={median_t:.3f}ms mean={mean_t:.3f}ms n={len(times)}",
                                  flush=True)
                        else:
                            size_data[cfg_label] = 0
                            print("EMPTY", flush=True)
                    elif node_rank == 0:
                        size_data[cfg_label] = 0
                        print("NO FILE", flush=True)

                except Exception as e:
                    if node_rank == 0:
                        err_str = str(e)[:300]
                        print(f"ERROR: {err_str}", flush=True)
                    size_data[cfg_label] = 0

            if node_rank == 0 and size_data:
                all_results[mode_label][size_label] = size_data

                # Show tuner gain
                auto_t = size_data.get("auto", 0)
                tuner_t = size_data.get("tuner_v3", 0)
                if auto_t > 0 and tuner_t > 0:
                    gain = (auto_t - tuner_t) / auto_t * 100
                    winner = "TUNER" if tuner_t < auto_t else "AUTO"
                    print(f"    => {winner} wins: gain={gain:+.1f}%\n", flush=True)

        if node_rank == 0:
            (results_dir / "validation_results_mn.json").write_text(
                json.dumps(all_results, indent=2))
            volume.commit()

    if node_rank == 0:
        (results_dir / "validation_results_mn.json").write_text(
            json.dumps(all_results, indent=2))
        volume.commit()

        all_files = {}
        for f in results_dir.iterdir():
            if f.is_file():
                all_files[f.name] = f.read_text()

        # Print summary
        print(f"\n{'='*70}")
        print(f"  VALIDATION COMPLETE — {len(all_files)} files saved")
        print(f"{'='*70}")
        for mode in ("sequential", "overlap"):
            print(f"\n  {mode.upper()}:")
            for size, row in all_results.get(mode, {}).items():
                auto_t = row.get("auto", 0)
                tuner_t = row.get("tuner_v3", 0)
                gain = (auto_t - tuner_t) / auto_t * 100 if auto_t > 0 else 0
                winner = "TUNER" if tuner_t < auto_t else "AUTO"
                print(f"    {size:>8s}: AUTO={auto_t:.1f}ms  TUNER={tuner_t:.1f}ms  "
                      f"gain={gain:+.1f}%  [{winner}]")
        print(flush=True)

        return {"results": all_results, "files": all_files}

    return {"results": {}, "files": {}}


@app.local_entrypoint()
def main():
    print(f"Launching tuner validation: {N_NODES} nodes × {GPUS_PER_NODE} GPUs = {WORLD_SIZE} total")
    print(f"Configs: {[c[0] for c in CONFIGS]}")
    print(f"Sizes: {[s[0] for s in MSG_SIZES]}, Modes: 2")
    print(f"Total runs: {len(CONFIGS) * len(MSG_SIZES) * 2}")
    print()

    out = run_tuner_validation.remote()

    results_dir = Path(__file__).parent / "results" / "tuner_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in out.get("files", {}).items():
        (results_dir / filename).write_text(content)
    print(f"\nSaved {len(out.get('files', {}))} files to {results_dir}")

    results = out.get("results", {})
    if results:
        print(json.dumps(results, indent=2))
