"""
Modal app: Multi-node sweep — fixed world_size=8, varying node count.

Tests whether inter-node hops amplify NCCL AUTO's suboptimality.
Configurations:
  1x8 (single node, 0 inter-node hops)
  2x4 (baseline, moderate hops)
  4x2 (high inter-node ratio)
  8x1 (max hops, best-effort)
"""

import json
import os
import time
from pathlib import Path

import modal
import modal.experimental

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sweep_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget", "libibverbs-dev", "libibverbs1")
    .run_commands("pip install --upgrade pip")
    .pip_install("torch", "numpy")
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
)

volume = modal.Volume.from_name(
    "browser-networking-tests-storage", create_if_missing=True
)
VOLUME_PATH = "/results"

app = modal.App("browser-networking-tests")

WORLD_SIZE = 8  # Constant across all configs

MSG_SIZES = [
    ("256KB",  65_536),
    ("1MB",    262_144),
    ("4MB",    1_048_576),
    ("16MB",   4_194_304),
    ("64MB",   16_777_216),
    ("256MB",  67_108_864),
]

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
              nccl_env, size_elems, overlap, out_path, debug_log_path,
              gpus_per_node):
    """Single GPU worker. gpus_per_node passed explicitly for rank calc."""
    import torch
    import torch.distributed as dist

    global_rank = node_rank * gpus_per_node + local_rank
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

    # NCCL debug on rank 0 for AUTO
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

    dist.init_process_group(backend="nccl", rank=global_rank,
                            world_size=WORLD_SIZE)

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


def run_sweep_on_cluster(n_nodes, gpus_per_node, node_rank, master_addr,
                         master_port, container_ips):
    """Core sweep logic shared by all node configurations."""
    import torch.multiprocessing as mp

    tag = f"{n_nodes}x{gpus_per_node}"
    results_dir = Path(VOLUME_PATH) / "node_sweep" / tag
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"sequential": {}, "overlap": {}}
    auto_selections = {}

    if node_rank == 0:
        metadata = {
            "n_nodes": n_nodes,
            "gpus_per_node": gpus_per_node,
            "world_size": WORLD_SIZE,
            "container_ips": list(container_ips),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        print(f"\n{'='*70}")
        print(f"  NODE SWEEP: {tag}")
        print(f"  {n_nodes} nodes x {gpus_per_node} GPUs = {WORLD_SIZE} total")
        print(f"  Master: {master_addr}:{master_port}")
        print(f"  IPs: {container_ips}")
        print(f"  Configs: {len(CONFIGS)}, Sizes: {len(MSG_SIZES)}, Modes: 2")
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
                run_tag = f"ns_{tag}_{mode_label}_{size_label}_{cfg_label}"
                out_path = str(results_dir / f"times_{run_tag}.txt")

                debug_log_path = None
                if cfg_label == "auto":
                    debug_log_path = str(
                        results_dir / f"nccl_debug_{mode_label}_{size_label}.log"
                    )

                if node_rank == 0:
                    debug_str = " [+DEBUG]" if debug_log_path else ""
                    print(f"  [{mode_label.upper()}] {cfg_label:15s} @ "
                          f"{size_label}{debug_str} ...", end=" ", flush=True)

                try:
                    mp.spawn(
                        worker_fn,
                        args=(node_rank, master_addr, master_port,
                              cfg_env, size_elems, overlap, out_path,
                              debug_log_path, gpus_per_node),
                        nprocs=gpus_per_node,
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

                # Print comparison
                valid = {k: v for k, v in size_data.items() if v > 0}
                if valid:
                    auto_t = valid.get("auto", 0)
                    best_cfg = min(valid, key=valid.get)
                    best_t = valid[best_cfg]
                    gap = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0
                    print(f"    => Best: {best_cfg} ({best_t:.3f}ms), "
                          f"AUTO={auto_t:.3f}ms, gap={gap:+.1f}%")
                    print(flush=True)

        # Save after each mode
        if node_rank == 0:
            (results_dir / "results.json").write_text(
                json.dumps(all_results, indent=2))
            (results_dir / "auto_selections.json").write_text(
                json.dumps(auto_selections, indent=2))
            volume.commit()

    # Final save + summary
    if node_rank == 0:
        (results_dir / "results.json").write_text(
            json.dumps(all_results, indent=2))
        (results_dir / "auto_selections.json").write_text(
            json.dumps(auto_selections, indent=2))
        volume.commit()

        print(f"\n{'='*100}")
        print(f"  NODE SWEEP {tag} — FINAL RESULTS")
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
                    gap = ((auto_t - valid[best]) / auto_t * 100
                           if auto_t > 0 else 0)
                    row += f"  {best:>12s}  {gap:>+9.1f}%"
                print(row)

        # Collect all files for return
        all_files = {}
        for f in results_dir.iterdir():
            if f.is_file():
                all_files[f.name] = f.read_text()

        return {"results": all_results, "selections": auto_selections,
                "files": all_files}

    return {"results": {}, "selections": {}, "files": {}}


# ── 1x8: Single node (no clustered decorator) ────────────────────────
@app.function(
    name="node-sweep-1x8",
    image=sweep_image,
    gpu="A100:8",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
def run_1x8():
    """Single node, 8 GPUs. All communication via NVLink."""
    return run_sweep_on_cluster(
        n_nodes=1,
        gpus_per_node=8,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        container_ips=["127.0.0.1"],
    )


# ── 2x4: Two nodes ───────────────────────────────────────────────────
@app.function(
    name="node-sweep-2x4",
    image=sweep_image,
    gpu="A100:4",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
@modal.experimental.clustered(2)
def run_2x4():
    """2 nodes x 4 GPUs."""
    cluster_info = modal.experimental.get_cluster_info()
    return run_sweep_on_cluster(
        n_nodes=2,
        gpus_per_node=4,
        node_rank=cluster_info.rank,
        master_addr=cluster_info.container_ips[0],
        master_port=29500,
        container_ips=cluster_info.container_ips,
    )


# ── 4x2: Four nodes ──────────────────────────────────────────────────
@app.function(
    name="node-sweep-4x2",
    image=sweep_image,
    gpu="A100:2",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
@modal.experimental.clustered(4)
def run_4x2():
    """4 nodes x 2 GPUs. High inter-node ratio."""
    cluster_info = modal.experimental.get_cluster_info()
    return run_sweep_on_cluster(
        n_nodes=4,
        gpus_per_node=2,
        node_rank=cluster_info.rank,
        master_addr=cluster_info.container_ips[0],
        master_port=29500,
        container_ips=cluster_info.container_ips,
    )


# ── 8x1: Eight nodes (best-effort) ──────────────────────────────────
@app.function(
    name="node-sweep-8x1",
    image=sweep_image,
    gpu="A100:1",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
@modal.experimental.clustered(8)
def run_8x1():
    """8 nodes x 1 GPU. Maximum inter-node hops."""
    cluster_info = modal.experimental.get_cluster_info()
    return run_sweep_on_cluster(
        n_nodes=8,
        gpus_per_node=1,
        node_rank=cluster_info.rank,
        master_addr=cluster_info.container_ips[0],
        master_port=29500,
        container_ips=cluster_info.container_ips,
    )


# ── Orchestrator: runs on Modal, calls configs SEQUENTIALLY ──────────
@app.function(
    name="node-sweep-orchestrator",
    image=sweep_image,
    timeout=21600,  # 6 hours for all configs
    volumes={VOLUME_PATH: volume},
)
def run_orchestrator(configs_to_run: list):
    """Runs each node config sequentially on Modal. No GPUs needed here —
    each config spawns its own GPU function via .remote().

    Retries each config up to 2 times on failure. Saves incrementally
    after each config so partial results survive crashes."""

    dispatch = {
        "1x8": run_1x8,
        "2x4": run_2x4,
        "4x2": run_4x2,
        "8x1": run_8x1,
    }

    results_dir = Path(VOLUME_PATH) / "node_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load any existing combined results (in case of resume)
    combined_path = results_dir / "combined_sweep.json"
    combined = {}
    if combined_path.is_file():
        try:
            combined = json.loads(combined_path.read_text())
        except Exception:
            combined = {}

    max_retries = 2

    for cfg in configs_to_run:
        n, g = cfg.split("x")
        print(f"\n{'='*70}")
        print(f"  LAUNCHING {cfg} ({n} nodes x {g} GPUs/node)")
        print(f"{'='*70}\n", flush=True)

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  Attempt {attempt}/{max_retries}...", flush=True)
                out = dispatch[cfg].remote()
                combined[cfg] = {
                    "results": out.get("results", {}),
                    "selections": out.get("selections", {}),
                }
                print(f"  {cfg} DONE ✓", flush=True)
                success = True
                break

            except Exception as e:
                print(f"  {cfg} attempt {attempt} FAILED: {e}", flush=True)
                if attempt < max_retries:
                    print(f"  Retrying {cfg}...", flush=True)
                    time.sleep(10)  # Brief pause before retry

        if not success:
            print(f"  {cfg} FAILED after {max_retries} attempts", flush=True)
            combined[cfg] = {"error": f"Failed after {max_retries} attempts"}

        # Save incrementally after each config
        combined_path.write_text(json.dumps(combined, indent=2))
        volume.commit()
        print(f"  Saved progress ({len([c for c in combined if 'error' not in combined[c]])} "
              f"configs complete)", flush=True)

    # Print gap comparison
    print(f"\n{'='*100}")
    print(f"  GAP COMPARISON ACROSS NODE CONFIGURATIONS")
    print(f"{'='*100}")

    for mode in ("sequential", "overlap"):
        print(f"\n  {mode.upper()} — AUTO Gap (%):")
        header = f"  {'Size':<8s}"
        for cfg in configs_to_run:
            header += f"  {cfg:>10s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for size_label, _ in MSG_SIZES:
            row = f"  {size_label:<8s}"
            for cfg in configs_to_run:
                data = combined.get(cfg, {}).get("results", {})
                size_data = data.get(mode, {}).get(size_label, {})
                valid = {k: v for k, v in size_data.items() if v > 0}
                auto_t = valid.get("auto", 0)
                if valid and auto_t > 0:
                    best_t = min(valid.values())
                    gap = (auto_t - best_t) / auto_t * 100
                    row += f"  {gap:>+9.1f}%"
                else:
                    row += f"  {'N/A':>10s}"
            print(row)

    return combined


# ── Local entrypoint ─────────────────────────────────────────────────
@app.local_entrypoint()
def main(config: str = "all"):
    """Multi-node sweep: fixed world_size=8, varying topology.

    Args:
        config: Which node config to run: 1x8, 2x4, 4x2, 8x1, or all
    """
    valid_configs = ["1x8", "2x4", "4x2", "8x1"]

    if config not in valid_configs and config != "all":
        print(f"Unknown config: {config}. Choose from: 1x8, 2x4, 4x2, 8x1, all")
        return

    configs_to_run = valid_configs if config == "all" else [config]

    print(f"NODE SWEEP: fixed world_size={WORLD_SIZE}")
    print(f"Configs to run: {configs_to_run}")
    print(f"  7 algo/proto combos x 6 sizes x 2 modes = 84 runs per config")
    print(f"  Runs SEQUENTIALLY via orchestrator (stays under GPU limit)")
    print()

    # Single orchestrator call — survives detach
    combined = run_orchestrator.remote(configs_to_run)

    # Save locally
    local_dir = Path(__file__).parent / "results" / "node_sweep"
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "combined_sweep.json").write_text(
        json.dumps(combined, indent=2))
    print(f"\nSaved combined results to {local_dir / 'combined_sweep.json'}")
