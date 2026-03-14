"""
Modal app: Expanded multi-node NCCL experiment.

Tests ALL algorithm × protocol combinations to produce a comprehensive
comparison figure (expanded Fig 6).

Algorithms: Tree, Ring, CollNet Direct, CollNet Chain, NVLS, NVLS Tree
Protocols:  Simple, LL, LL128 (where applicable)

12 configs × 6 sizes × 2 modes = 144 total runs.

Note: CollNet requires SHARP/InfiniBand hardware collectives, NVLS requires
NVSwitch/NVLink.  These may fail on certain cluster topologies — failures are
recorded as 0 and gracefully skipped in analysis.

App name: browser-networking-tests.  Function name: browser-networking-tests.
"""

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

# ==========================================================================
# EXPANDED CONFIGS: All algo×proto combinations
# ==========================================================================
CONFIGS = [
    # Baseline — NCCL picks automatically
    ("auto",                  {}),

    # Tree algorithm — all 3 protocols
    ("tree_simple",           {"NCCL_ALGO": "Tree",           "NCCL_PROTO": "Simple"}),
    ("tree_ll",               {"NCCL_ALGO": "Tree",           "NCCL_PROTO": "LL"}),
    ("tree_ll128",            {"NCCL_ALGO": "Tree",           "NCCL_PROTO": "LL128"}),

    # Ring algorithm — all 3 protocols
    ("ring_simple",           {"NCCL_ALGO": "Ring",           "NCCL_PROTO": "Simple"}),
    ("ring_ll",               {"NCCL_ALGO": "Ring",           "NCCL_PROTO": "LL"}),
    ("ring_ll128",            {"NCCL_ALGO": "Ring",           "NCCL_PROTO": "LL128"}),

    # CollNet Direct — Simple only (requires SHARP)
    ("collnet_direct_simple", {"NCCL_ALGO": "CollNetDirect",  "NCCL_PROTO": "Simple"}),

    # CollNet Chain — Simple only (requires SHARP)
    ("collnet_chain_simple",  {"NCCL_ALGO": "CollNetChain",   "NCCL_PROTO": "Simple"}),

    # NVLS — Simple only (requires NVSwitch)
    ("nvls_simple",           {"NCCL_ALGO": "NVLS",           "NCCL_PROTO": "Simple"}),

    # NVLS Tree — Simple only (requires NVSwitch + inter-node)
    ("nvls_tree_simple",      {"NCCL_ALGO": "NVLSTree",       "NCCL_PROTO": "Simple"}),

    # PAT (Parallel Allreduce Tree) — all 3 protocols (NCCL 2.19+)
    ("pat_simple",            {"NCCL_ALGO": "PAT",            "NCCL_PROTO": "Simple"}),
    ("pat_ll",                {"NCCL_ALGO": "PAT",            "NCCL_PROTO": "LL"}),
    ("pat_ll128",             {"NCCL_ALGO": "PAT",            "NCCL_PROTO": "LL128"}),
]

ITERS = 50
WARMUP = 10
COMPUTE_MUL = 4096


# ---------------------------------------------------------------------------
# Worker function: runs on each GPU
# ---------------------------------------------------------------------------
def worker_fn(local_rank, node_rank, master_addr, master_port,
              nccl_env, size_elems, overlap, out_path):
    """Single GPU worker: compute + allreduce iteration proxy."""
    import torch
    import torch.distributed as dist

    global_rank = node_rank * GPUS_PER_NODE + local_rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Clear all NCCL override env vars
    for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_TUNER_PLUGIN",
                "NCCL_TUNER_REWARD_FILE", "NCCL_TUNER_POLICY"):
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


# ---------------------------------------------------------------------------
# Main Modal function (multi-node)
# ---------------------------------------------------------------------------
@app.function(
    name="browser-networking-tests",
    image=multinode_image,
    gpu=f"A100:{GPUS_PER_NODE}",
    timeout=14400,   # 4 hours — more configs need more time
    volumes={VOLUME_PATH: volume},
)
@modal.experimental.clustered(N_NODES)
def run_multinode_experiment_expanded():
    """Expanded multi-node sweep: all algo×proto × sizes × modes."""
    import torch.multiprocessing as mp

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = cluster_info.container_ips[0]
    master_port = 29500

    results_dir = Path(VOLUME_PATH) / "multinode_expanded"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"sequential": {}, "overlap": {}}
    failures = []   # Track which configs fail (CollNet/NVLS may not be supported)

    if node_rank == 0:
        print(f"\n{'='*70}")
        print(f"  EXPANDED MULTI-NODE EXPERIMENT")
        print(f"  {N_NODES} nodes x {GPUS_PER_NODE} GPUs = {WORLD_SIZE} total")
        print(f"  Master: {master_addr}:{master_port}")
        print(f"  Configs: {len(CONFIGS)}")
        print(f"  Message sizes: {[s[0] for s in MSG_SIZES]}")
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
                # Skip configs that already failed (hardware not supported)
                if cfg_label in failures:
                    if node_rank == 0:
                        print(f"  [{mode_label.upper()}] {cfg_label:25s} @ {size_label} ... "
                              f"SKIPPED (failed earlier)", flush=True)
                    size_data[cfg_label] = 0
                    continue

                tag = f"mn_exp_{mode_label}_{size_label}_{cfg_label}"
                out_path = str(results_dir / f"times_{tag}.txt")

                if node_rank == 0:
                    print(f"  [{mode_label.upper()}] {cfg_label:25s} @ {size_label} ...",
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
                            print(f"median={median_t:.3f}ms mean={mean_t:.3f}ms "
                                  f"n={len(times)}", flush=True)
                        else:
                            size_data[cfg_label] = 0
                            print("EMPTY", flush=True)
                    elif node_rank == 0:
                        size_data[cfg_label] = 0
                        print("NO FILE", flush=True)

                except Exception as e:
                    err_str = str(e)[:300]
                    if node_rank == 0:
                        print(f"FAILED: {err_str}", flush=True)
                    size_data[cfg_label] = 0
                    # Mark hardware-unsupported configs to skip future sizes
                    if any(kw in err_str.lower() for kw in
                           ["not supported", "invalid algo", "invalid protocol",
                            "no support", "unavailable"]):
                        failures.append(cfg_label)
                        if node_rank == 0:
                            print(f"    => Marking {cfg_label} as unsupported, "
                                  f"will skip for remaining sizes", flush=True)

            if node_rank == 0 and size_data:
                all_results[mode_label][size_label] = size_data

                valid = {k: v for k, v in size_data.items() if v > 0}
                if valid:
                    auto_t = valid.get("auto", 0)
                    best_cfg = min(valid, key=valid.get)
                    best_t = valid[best_cfg]
                    gap = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0
                    print(f"    => Best: {best_cfg} ({best_t:.3f}ms), "
                          f"AUTO gap: {gap:+.1f}%\n", flush=True)

        # Save incremental results after each mode
        if node_rank == 0:
            (results_dir / "multinode_expanded_results.json").write_text(
                json.dumps(all_results, indent=2))
            if failures:
                (results_dir / "unsupported_configs.json").write_text(
                    json.dumps(failures, indent=2))
            volume.commit()

    # Final save
    if node_rank == 0:
        (results_dir / "multinode_expanded_results.json").write_text(
            json.dumps(all_results, indent=2))
        if failures:
            (results_dir / "unsupported_configs.json").write_text(
                json.dumps(failures, indent=2))
        volume.commit()

        all_files = {}
        for f in results_dir.iterdir():
            if f.is_file():
                all_files[f.name] = f.read_text()

        print(f"\n{'='*70}")
        print(f"  EXPANDED EXPERIMENT COMPLETE — {len(all_files)} files saved")
        if failures:
            print(f"  Unsupported configs: {failures}")
        print(f"{'='*70}\n", flush=True)

        return {"results": all_results, "files": all_files,
                "unsupported": failures}

    return {"results": {}, "files": {}, "unsupported": []}


@app.local_entrypoint()
def main():
    print(f"Launching EXPANDED multi-node experiment:")
    print(f"  {N_NODES} nodes x {GPUS_PER_NODE} GPUs = {WORLD_SIZE} total")
    print(f"  Configs: {len(CONFIGS)}")
    for label, env in CONFIGS:
        env_str = " ".join(f"{k}={v}" for k, v in env.items()) if env else "(default)"
        print(f"    {label:25s} -> {env_str}")
    print(f"  Sizes: {len(MSG_SIZES)}, Modes: 2")
    print(f"  Total runs: {len(CONFIGS) * len(MSG_SIZES) * 2}")
    print()

    out = run_multinode_experiment_expanded.remote()

    results_dir = Path(__file__).parent / "results" / "multinode_expanded"
    results_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in out.get("files", {}).items():
        (results_dir / filename).write_text(content)
    print(f"\nSaved {len(out.get('files', {}))} files to {results_dir}")

    unsupported = out.get("unsupported", [])
    if unsupported:
        print(f"\nUnsupported configs (hardware not available): {unsupported}")

    results = out.get("results", {})
    if not results.get("sequential"):
        print("No results (check Modal logs)")
        return

    print(f"\n{'='*100}")
    print(f"  EXPANDED MULTI-NODE: ALL ALGO×PROTO ({N_NODES} nodes x {GPUS_PER_NODE} GPUs)")
    print(f"{'='*100}")

    for size_label, _ in MSG_SIZES:
        seq = results["sequential"].get(size_label, {})
        ovl = results["overlap"].get(size_label, {})
        if not seq or not ovl:
            continue

        seq_valid = {k: v for k, v in seq.items() if v > 0}
        ovl_valid = {k: v for k, v in ovl.items() if v > 0}
        if not seq_valid or not ovl_valid:
            continue

        seq_auto = seq_valid.get("auto", 0)
        seq_best = min(seq_valid, key=seq_valid.get)
        seq_gap = (seq_auto - seq_valid[seq_best]) / seq_auto * 100 if seq_auto > 0 else 0

        ovl_auto = ovl_valid.get("auto", 0)
        ovl_best = min(ovl_valid, key=ovl_valid.get)
        ovl_gap = (ovl_auto - ovl_valid[ovl_best]) / ovl_auto * 100 if ovl_auto > 0 else 0

        flipped = "FLIP" if seq_best != ovl_best else ""
        print(f"  {size_label:<8s} | SEQ: {seq_best:>25s} gap={seq_gap:+.1f}% | "
              f"OVL: {ovl_best:>25s} gap={ovl_gap:+.1f}% | {flipped}")
