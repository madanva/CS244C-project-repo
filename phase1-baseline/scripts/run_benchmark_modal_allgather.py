"""
Modal app: Phase 1 — Multi-node NCCL all_gather benchmark.

Same experiment as run_benchmark_aws_multinode_allgather.py but on Modal:
multi-node, multi-GPU, Ring (Simple/LL/LL128) + AUTO, optional NVLS;
sequential and overlap modes; results to Modal volume and local.

Uses clustered(N_NODES) and multiprocessing spawn per node, like
phase4-tuner/a100-8gpu-new/run_modal_multinode.py.

Run: modal run run_benchmark_modal_allgather.py
Keep your terminal/connection open until the run finishes; if the local client
disconnects, Modal stops the app and no results are returned. For long runs
use: modal run --detach run_benchmark_modal_allgather.py (then fetch results
from the volume "phase1-allgather-modal-storage" in the Modal dashboard).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal
import modal.experimental

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

multinode_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget", "libibverbs-dev", "libibverbs1")
    .run_commands("pip install --upgrade pip")
    .pip_install("torch", "numpy")
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
)

volume = modal.Volume.from_name("phase1-allgather-modal-storage", create_if_missing=True)
VOLUME_PATH = "/results"

app = modal.App("phase1-allgather-multinode")

N_NODES = 2
GPUS_PER_NODE = 4  # 2×4 = 8 total GPUs
WORLD_SIZE = N_NODES * GPUS_PER_NODE

# Message sizes: match phase1 allgather sweep (8 to 256M); subset for reasonable runtime
MSG_SIZES = [
    ("8", 8),
    ("1K", 1024),
    ("64K", 65536),
    ("256KB", 262144),
    ("1MB", 1048576),
    ("4MB", 4194304),
    ("16MB", 16777216),
    ("64MB", 67108864),
    ("256MB", 268435456),
]

# AllGather configs: Ring (Simple, LL, LL128) + AUTO; optional NVLS
CONFIGS = [
    ("ring_simple", {"NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple"}),
    ("ring_ll", {"NCCL_ALGO": "Ring", "NCCL_PROTO": "LL"}),
    ("ring_ll128", {"NCCL_ALGO": "Ring", "NCCL_PROTO": "LL128"}),
    ("auto", {}),
]

ITERS = 50
WARMUP = 10
COMPUTE_MUL = 4096


def worker_fn(
    local_rank: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    nccl_env: dict,
    size_elems: int,
    overlap: bool,
    out_path: str | None,
) -> None:
    """Single GPU worker: all_gather (and optional compute overlap) iteration."""
    import torch
    import torch.distributed as dist

    global_rank = node_rank * GPUS_PER_NODE + local_rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_TUNER_PLUGIN", "NCCL_TUNER_REWARD_FILE"):
        os.environ.pop(key, None)
    for k, v in nccl_env.items():
        os.environ[k] = str(v)
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    os.environ["LOCAL_RANK"] = str(local_rank)

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=WORLD_SIZE)

    t = torch.randn(size_elems, device=device, dtype=torch.float32)
    tensor_list = [torch.empty(size_elems, device=device, dtype=torch.float32) for _ in range(WORLD_SIZE)]

    if not overlap:
        for _ in range(WARMUP):
            a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            dist.all_gather(tensor_list, t)
            torch.cuda.synchronize()
        times_ms = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            dist.all_gather(tensor_list, t)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    else:
        compute_stream = torch.cuda.Stream(device=device)
        comm_stream = torch.cuda.Stream(device=device)
        for _ in range(WARMUP):
            with torch.cuda.stream(compute_stream):
                a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                _ = torch.matmul(a, b)
            with torch.cuda.stream(comm_stream):
                dist.all_gather(tensor_list, t)
            torch.cuda.synchronize()
        times_ms = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.stream(compute_stream):
                a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                _ = torch.matmul(a, b)
            with torch.cuda.stream(comm_stream):
                dist.all_gather(tensor_list, t)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    if global_rank == 0 and out_path:
        Path(out_path).write_text("\n".join(f"{t:.3f}" for t in times_ms) + "\n")

    try:
        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        pass


@app.function(
    name="phase1-allgather-multinode",
    image=multinode_image,
    gpu=f"A100:{GPUS_PER_NODE}",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
@modal.experimental.clustered(N_NODES)
def run_allgather_multinode(include_nvls: bool = False, mode: str = "both"):
    """Multi-node all_gather sweep: sequential + overlap across message sizes and configs."""
    import torch.multiprocessing as mp

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = cluster_info.container_ips[0]
    master_port = 29500

    results_dir = Path(VOLUME_PATH) / "allgather_multinode"
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = list(CONFIGS)
    if include_nvls:
        configs = configs + [("nvls_simple", {"NCCL_ALGO": "NVLS", "NCCL_PROTO": "Simple"})]

    all_results: dict = {"sequential": {}, "overlap": {}}

    if mode == "overlap":
        modes = [("overlap", True)]
    elif mode == "sequential":
        modes = [("sequential", False)]
    else:
        modes = [("sequential", False), ("overlap", True)]

    if node_rank == 0:
        print(f"\n{'='*70}")
        print(f"  PHASE1 ALL_GATHER MULTI-NODE (Modal)")
        print(f"  {N_NODES} nodes × {GPUS_PER_NODE} GPUs = {WORLD_SIZE} total")
        print(f"  Master: {master_addr}:{master_port}")
        print(f"  Container IPs: {cluster_info.container_ips}")
        print(f"  Message sizes: {[s[0] for s in MSG_SIZES]}")
        print(f"  Configs: {[c[0] for c in configs]}")
        print(f"  Mode: {mode}")
        print(f"{'='*70}\n", flush=True)

    for mode_label, overlap in modes:
        if node_rank == 0:
            print(f"\n{'='*60}\n  MODE: {mode_label.upper()}\n{'='*60}\n", flush=True)

        for size_label, size_bytes in MSG_SIZES:
            size_elems = max(1, size_bytes // 4)  # float32, avoid 0
            size_data = {}

            for cfg_label, cfg_env in configs:
                tag = f"allgather_{mode_label}_{size_label}_{cfg_label}"
                out_path = str(results_dir / f"times_{tag}.txt")

                if node_rank == 0:
                    print(f"  [{mode_label.upper()}] {cfg_label:15s} @ {size_label:>8s} ...", end=" ", flush=True)

                try:
                    mp.spawn(
                        worker_fn,
                        args=(node_rank, master_addr, master_port, cfg_env, size_elems, overlap, out_path),
                        nprocs=GPUS_PER_NODE,
                        join=True,
                    )
                    if node_rank == 0 and Path(out_path).is_file():
                        lines = Path(out_path).read_text().strip().split("\n")
                        times = [float(x) for x in lines if x.strip()]
                        if times:
                            times_sorted = sorted(times)
                            median_t = times_sorted[len(times_sorted) // 2]
                            mean_t = sum(times) / len(times)
                            size_data[cfg_label] = round(median_t, 3)
                            print(f"median={median_t:.3f}ms mean={mean_t:.3f}ms n={len(times)}", flush=True)
                        else:
                            size_data[cfg_label] = 0
                            print("EMPTY", flush=True)
                    elif node_rank == 0:
                        size_data[cfg_label] = 0
                        print("NO FILE", flush=True)
                except Exception as e:
                    if node_rank == 0:
                        print(f"ERROR: {str(e)[:300]}", flush=True)
                    size_data[cfg_label] = 0

            if node_rank == 0 and size_data:
                all_results[mode_label][size_label] = size_data
                valid = {k: v for k, v in size_data.items() if v > 0}
                if valid:
                    auto_t = valid.get("auto", 0)
                    best_cfg = min(valid, key=valid.get)
                    best_t = valid[best_cfg]
                    gap = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0
                    print(f"    => Best: {best_cfg} ({best_t:.3f}ms), AUTO gap: {gap:+.1f}%\n", flush=True)

        if node_rank == 0:
            (results_dir / "allgather_multinode_results.json").write_text(json.dumps(all_results, indent=2))
            volume.commit()

    if node_rank == 0:
        (results_dir / "allgather_multinode_results.json").write_text(json.dumps(all_results, indent=2))
        volume.commit()
        all_files = {}
        for f in results_dir.iterdir():
            if f.is_file():
                all_files[f.name] = f.read_text()
        print(f"\n{'='*70}\n  ALL_GATHER EXPERIMENT COMPLETE — {len(all_files)} files saved\n{'='*70}\n", flush=True)
        return {"results": all_results, "files": all_files}
    return {"results": {}, "files": {}}


@app.local_entrypoint()
def main(include_nvls: bool = False, mode: str = "both"):
    """Run the all_gather multinode experiment on Modal and save results locally."""
    print(
        f"Launching Phase1 all_gather: {N_NODES} nodes × {GPUS_PER_NODE} GPUs = {WORLD_SIZE} total"
    )
    print(f"Configs: {len(CONFIGS) + (1 if include_nvls else 0)}, Sizes: {len(MSG_SIZES)}, Mode: {mode}")
    print()

    out = run_allgather_multinode.remote(include_nvls, mode=mode)

    results_dir = Path(__file__).parent / "results" / "modal_allgather"
    results_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in out.get("files", {}).items():
        (results_dir / filename).write_text(content)
    print(f"\nSaved {len(out.get('files', {}))} files to {results_dir}")

    results = out.get("results", {})
    if not results.get("sequential"):
        print("No results (check Modal logs)")
        return

    print(f"\n{'='*90}")
    print(f"  PHASE1 ALL_GATHER: SEQUENTIAL vs OVERLAP ({N_NODES} nodes × {GPUS_PER_NODE} GPUs)")
    print(f"{'='*90}")
    for size_label, _ in MSG_SIZES:
        seq = results["sequential"].get(size_label, {})
        ovl = results["overlap"].get(size_label, {})
        if not seq or not ovl:
            continue
        seq_valid = {k: v for k, v in seq.items() if v > 0}
        ovl_valid = {k: v for k, v in ovl.items() if v > 0}
        if not seq_valid or not ovl_valid:
            continue
        seq_best = min(seq_valid, key=seq_valid.get)
        ovl_best = min(ovl_valid, key=ovl_valid.get)
        seq_auto = seq_valid.get("auto", 0)
        ovl_auto = ovl_valid.get("auto", 0)
        seq_gap = (seq_auto - seq_valid[seq_best]) / seq_auto * 100 if seq_auto > 0 else 0
        ovl_gap = (ovl_auto - ovl_valid[ovl_best]) / ovl_auto * 100 if ovl_auto > 0 else 0
        flipped = " [FLIP]" if seq_best != ovl_best else ""
        print(
            f"  {size_label:>8s} | SEQ: {seq_best:>12s} {seq_valid[seq_best]:.3f}ms gap={seq_gap:+.1f}% | "
            f"OVL: {ovl_best:>12s} {ovl_valid[ovl_best]:.3f}ms gap={ovl_gap:+.1f}%{flipped}"
        )


if __name__ == "__main__":
    main()
