"""
Modal app: Same-cluster comparison — AUTO vs all forced configs.

Runs ALL configs on the SAME cluster allocation to eliminate cluster-to-cluster
variance. Also captures NCCL's internal selection for AUTO (algo, proto,
channels, chunk sizes) via NCCL_DEBUG.

This is the definitive comparison: same hardware, same network, same topology.
"""

import json
import os
import time
from pathlib import Path

import modal
import modal.experimental

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

comparison_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget", "libibverbs-dev", "libibverbs1")
    .run_commands("pip install --upgrade pip")
    .pip_install("torch", "numpy")
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
)

volume = modal.Volume.from_name("browser-networking-tests-storage", create_if_missing=True)
VOLUME_PATH = "/results"

app = modal.App("browser-networking-tests")

N_NODES = 2
GPUS_PER_NODE = 4
WORLD_SIZE = N_NODES * GPUS_PER_NODE

MSG_SIZES = [
    ("256KB",  65_536),
    ("1MB",    262_144),
    ("4MB",    1_048_576),
    ("16MB",   4_194_304),
    ("64MB",   16_777_216),
    ("256MB",  67_108_864),
]

# Only configs that actually work on Modal A100s
CONFIGS = [
    ("auto",        {}),
    ("tree_simple", {"NCCL_ALGO": "Tree",  "NCCL_PROTO": "Simple"}),
    ("tree_ll",     {"NCCL_ALGO": "Tree",  "NCCL_PROTO": "LL"}),
    ("tree_ll128",  {"NCCL_ALGO": "Tree",  "NCCL_PROTO": "LL128"}),
    ("ring_simple", {"NCCL_ALGO": "Ring",  "NCCL_PROTO": "Simple"}),
    ("ring_ll",     {"NCCL_ALGO": "Ring",  "NCCL_PROTO": "LL"}),
    ("ring_ll128",  {"NCCL_ALGO": "Ring",  "NCCL_PROTO": "LL128"}),
]

ITERS = 50
WARMUP = 10
COMPUTE_MUL = 4096


def worker_fn(local_rank, node_rank, master_addr, master_port,
              nccl_env, size_elems, overlap, out_path, debug_log_path):
    """Single GPU worker with optional NCCL debug logging."""
    import torch
    import torch.distributed as dist

    global_rank = node_rank * GPUS_PER_NODE + local_rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Clear all NCCL override env vars
    for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_TUNER_PLUGIN",
                "NCCL_TUNER_REWARD_FILE", "NCCL_TUNER_POLICY",
                "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE"):
        os.environ.pop(key, None)

    # Apply config env vars
    for k, v in nccl_env.items():
        os.environ[k] = v

    # Enable NCCL debug logging on rank 0 if debug_log_path is set
    if global_rank == 0 and debug_log_path:
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "TUNING,COLL"
        os.environ["NCCL_DEBUG_FILE"] = debug_log_path
    else:
        os.environ["NCCL_DEBUG"] = "WARN"

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    os.environ["LOCAL_RANK"] = str(local_rank)

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=WORLD_SIZE)

    grad = torch.randn(size_elems, device=device, dtype=torch.float32) / WORLD_SIZE

    if not overlap:
        # Sequential: compute then communicate
        for _ in range(WARMUP):
            a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            tmp = grad.clone()
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

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
        # Overlap: compute and communicate on separate streams
        compute_stream = torch.cuda.Stream(device=device)
        comm_stream = torch.cuda.Stream(device=device)

        for _ in range(WARMUP):
            with torch.cuda.stream(compute_stream):
                a = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                b = torch.randn(COMPUTE_MUL, COMPUTE_MUL, device=device)
                _ = torch.matmul(a, b)
            with torch.cuda.stream(comm_stream):
                tmp = grad.clone()
                dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
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
                tmp = grad.clone()
                dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    if global_rank == 0 and out_path:
        out_lines = [f"{t:.3f}" for t in times_ms]
        Path(out_path).write_text("\n".join(out_lines) + "\n")

    dist.destroy_process_group()


@app.function(
    name="browser-networking-tests",
    image=comparison_image,
    gpu=f"A100:{GPUS_PER_NODE}",
    timeout=10800,  # 3 hours
    volumes={VOLUME_PATH: volume},
)
@modal.experimental.clustered(N_NODES)
def run_same_cluster_comparison():
    """All configs on the SAME cluster — definitive comparison."""
    import torch.multiprocessing as mp

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = cluster_info.container_ips[0]
    master_port = 29500

    results_dir = Path(VOLUME_PATH) / "same_cluster_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"sequential": {}, "overlap": {}}
    auto_selections = {}

    if node_rank == 0:
        print(f"\n{'='*70}")
        print(f"  SAME-CLUSTER COMPARISON")
        print(f"  All configs run on IDENTICAL hardware/network")
        print(f"  {N_NODES} nodes x {GPUS_PER_NODE} GPUs = {WORLD_SIZE} total")
        print(f"  Master: {master_addr}:{master_port}")
        print(f"  IPs: {cluster_info.container_ips}")
        print(f"  Configs: {len(CONFIGS)}")
        print(f"  Sizes: {len(MSG_SIZES)}, Modes: 2")
        print(f"  Total runs: {len(CONFIGS) * len(MSG_SIZES) * 2}")
        print(f"{'='*70}\n", flush=True)

    for mode_label, overlap in [("sequential", False), ("overlap", True)]:
        if node_rank == 0:
            print(f"\n{'='*60}")
            print(f"  MODE: {mode_label.upper()}")
            print(f"{'='*60}\n", flush=True)

        for size_label, size_elems in MSG_SIZES:
            size_data = {}

            for cfg_label, cfg_env in CONFIGS:
                tag = f"sc_{mode_label}_{size_label}_{cfg_label}"
                out_path = str(results_dir / f"times_{tag}.txt")

                # Enable NCCL debug for AUTO runs
                debug_log_path = None
                if cfg_label == "auto":
                    debug_log_path = str(results_dir / f"nccl_debug_{mode_label}_{size_label}.log")

                if node_rank == 0:
                    debug_str = " [+DEBUG]" if debug_log_path else ""
                    print(f"  [{mode_label.upper()}] {cfg_label:15s} @ {size_label}{debug_str} ...",
                          end=" ", flush=True)

                try:
                    mp.spawn(
                        worker_fn,
                        args=(node_rank, master_addr, master_port,
                              cfg_env, size_elems, overlap, out_path,
                              debug_log_path),
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
                            print(f"median={median_t:.3f}ms  mean={mean_t:.3f}ms  "
                                  f"n={len(times)}", flush=True)

                            # Parse NCCL debug for AUTO
                            if debug_log_path and Path(debug_log_path).is_file():
                                log_content = Path(debug_log_path).read_text()
                                tuning_lines = []
                                for line in log_content.split("\n"):
                                    if "Algo" in line and "proto" in line:
                                        tuning_lines.append(line.strip())
                                if tuning_lines:
                                    unique = list(dict.fromkeys(tuning_lines))
                                    key = f"{mode_label}_{size_label}"
                                    auto_selections[key] = unique
                                    for u in unique:
                                        print(f"    NCCL: {u}")
                        else:
                            size_data[cfg_label] = 0
                            print("EMPTY", flush=True)
                    elif node_rank == 0:
                        size_data[cfg_label] = 0
                        print("NO FILE", flush=True)

                except Exception as e:
                    if node_rank == 0:
                        print(f"ERROR: {str(e)[:200]}", flush=True)
                    size_data[cfg_label] = 0

            if node_rank == 0 and size_data:
                all_results[mode_label][size_label] = size_data

                # Print comparison for this size
                valid = {k: v for k, v in size_data.items() if v > 0}
                if valid:
                    auto_t = valid.get("auto", 0)
                    best_cfg = min(valid, key=valid.get)
                    best_t = valid[best_cfg]
                    gap = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0

                    print(f"    => Best: {best_cfg} ({best_t:.3f}ms), "
                          f"AUTO={auto_t:.3f}ms, gap={gap:+.1f}%")

                    # Show how AUTO compares to the config it supposedly selected
                    if auto_t > 0:
                        for cfg, t in sorted(valid.items(), key=lambda x: x[1]):
                            if cfg == "auto":
                                continue
                            vs_auto = (auto_t - t) / auto_t * 100
                            marker = " <<<" if abs(vs_auto) < 3 else ""
                            print(f"       {cfg:15s}: {t:.3f}ms ({vs_auto:+.1f}% vs AUTO){marker}")
                    print(flush=True)

        # Save after each mode
        if node_rank == 0:
            (results_dir / "same_cluster_results.json").write_text(
                json.dumps(all_results, indent=2))
            (results_dir / "auto_selections.json").write_text(
                json.dumps(auto_selections, indent=2))
            volume.commit()

    # Final save
    if node_rank == 0:
        (results_dir / "same_cluster_results.json").write_text(
            json.dumps(all_results, indent=2))
        (results_dir / "auto_selections.json").write_text(
            json.dumps(auto_selections, indent=2))
        volume.commit()

        # Print final summary
        print(f"\n{'='*100}")
        print(f"  DEFINITIVE SAME-CLUSTER RESULTS")
        print(f"{'='*100}")

        for mode in ("sequential", "overlap"):
            print(f"\n  {mode.upper()}:")
            header = f"  {'Size':<8s}"
            for cfg, _ in CONFIGS:
                header += f"  {cfg:>12s}"
            header += f"  {'Best':>12s}  {'AUTO Gap':>10s}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for size_label, _ in MSG_SIZES:
                if size_label not in all_results[mode]:
                    continue
                row = f"  {size_label:<8s}"
                d = all_results[mode][size_label]
                valid = {}
                for cfg, _ in CONFIGS:
                    v = d.get(cfg, 0)
                    if v > 0:
                        row += f"  {v:>12.3f}"
                        valid[cfg] = v
                    else:
                        row += f"  {'N/A':>12s}"
                if valid:
                    best = min(valid, key=valid.get)
                    auto_t = valid.get("auto", 0)
                    gap = (auto_t - valid[best]) / auto_t * 100 if auto_t > 0 else 0
                    row += f"  {best:>12s}  {gap:>+9.1f}%"
                print(row)

        print(f"\n  AUTO SELECTIONS (from NCCL debug):")
        for key, sels in sorted(auto_selections.items()):
            print(f"    {key}:")
            for s in sels:
                print(f"      {s}")

        all_files = {}
        for f in results_dir.iterdir():
            if f.is_file():
                all_files[f.name] = f.read_text()

        return {"results": all_results, "selections": auto_selections,
                "files": all_files}

    return {"results": {}, "selections": {}, "files": {}}


@app.local_entrypoint()
def main():
    print("SAME-CLUSTER COMPARISON: AUTO vs All Configs")
    print(f"  {N_NODES} nodes x {GPUS_PER_NODE} GPUs = {WORLD_SIZE} total")
    print(f"  Configs: {len(CONFIGS)}")
    for label, env in CONFIGS:
        env_str = " ".join(f"{k}={v}" for k, v in env.items()) if env else "(AUTO)"
        print(f"    {label:15s} -> {env_str}")
    print(f"  Sizes: {len(MSG_SIZES)}, Modes: 2")
    print(f"  Total runs: {len(CONFIGS) * len(MSG_SIZES) * 2}")
    print()

    out = run_same_cluster_comparison.remote()

    results_dir = Path(__file__).parent / "results" / "same_cluster_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in out.get("files", {}).items():
        (results_dir / filename).write_text(content)
    print(f"\nSaved {len(out.get('files', {}))} files to {results_dir}")

    results = out.get("results", {})
    selections = out.get("selections", {})

    if selections:
        print(f"\n{'='*70}")
        print(f"  NCCL AUTO SELECTIONS")
        print(f"{'='*70}")
        for key, sels in sorted(selections.items()):
            print(f"  {key}:")
            for s in sels:
                print(f"    {s}")
