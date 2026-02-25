"""
Modal app: Phase 4 — Overlap-aware NCCL tuning experiment on 8x A100.

Paper-level experiment: demonstrates that NCCL AUTO's protocol/algorithm
selection becomes suboptimal under compute-communication overlap, because
it doesn't account for SM contention from concurrent compute kernels.

Three experiment blocks:
  1. Sequential baselines (re-run for same-session consistency)
  2. Overlap baselines (same sizes/configs, with --overlap)
  3. Compute intensity sweep (vary matmul size under overlap)

App name: browser-networking-tests.  Function name: browser-networking-tests.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

rl_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget")
    .run_commands("pip install --upgrade pip")
    .pip_install("torch")
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
)

volume = modal.Volume.from_name("browser-networking-tests-storage", create_if_missing=True)
VOLUME_PATH = "/results"

app = modal.App("browser-networking-tests")

PROXY_SCRIPT = "/repo/phase3-iteration-proxy/a100-8gpu-new/iteration_proxy.py"

# ---------------------------------------------------------------------------
# Message sizes targeting NCCL transition boundaries
# ---------------------------------------------------------------------------
MSG_SIZES = [
    ("32KB",   8_192),
    ("64KB",   16_384),
    ("256KB",  65_536),
    ("1MB",    262_144),
    ("2MB",    524_288),
    ("4MB",    1_048_576),
    ("16MB",   4_194_304),
    ("64MB",   16_777_216),
    ("256MB",  67_108_864),
]

# ---------------------------------------------------------------------------
# NCCL configs to test
# ---------------------------------------------------------------------------
CONFIGS = [
    ("auto",        {}),
    ("tree_simple", {"NCCL_ALGO": "Tree",  "NCCL_PROTO": "Simple"}),
    ("tree_ll128",  {"NCCL_ALGO": "Tree",  "NCCL_PROTO": "LL128"}),
    ("ring_simple", {"NCCL_ALGO": "Ring",  "NCCL_PROTO": "Simple"}),
    ("ring_ll128",  {"NCCL_ALGO": "Ring",  "NCCL_PROTO": "LL128"}),
]

# ---------------------------------------------------------------------------
# Compute intensity levels for the sweep (matmul NxN dimension)
# ---------------------------------------------------------------------------
COMPUTE_MULS = [
    ("light",   1024),   # ~1 ms matmul — comm-dominated
    ("medium",  2048),   # ~4 ms matmul — balanced
    ("default", 4096),   # ~25 ms matmul — default (matches Phase 3)
    ("heavy",   6144),   # ~80 ms matmul — compute-heavy
    ("extreme", 8192),   # ~190 ms matmul — SM-saturated
]

# Focus the compute intensity sweep on sizes where we expect the most
# interesting behavior (transition boundaries + one stable region)
SWEEP_SIZES = [
    ("1MB",    262_144),
    ("4MB",    1_048_576),
    ("16MB",   4_194_304),
    ("64MB",   16_777_216),
]

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
ITERS = 50
WARMUP = 10

# For compute-heavy configs, use fewer iters (each takes much longer)
HEAVY_ITERS = 30
HEAVY_WARMUP = 5


# ---------------------------------------------------------------------------
# Proxy launcher
# ---------------------------------------------------------------------------
def run_proxy(env_extra, iters, warmup, out_file, size_elems,
              overlap=False, compute_mul=4096):
    """Run the iteration proxy; return list of float times in ms."""
    env = {**os.environ, **env_extra}
    for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_TUNER_PLUGIN",
                "NCCL_TUNER_REWARD_FILE"):
        if key not in env_extra:
            env.pop(key, None)

    cmd = [
        "python", "-m", "torch.distributed.run",
        "--nproc_per_node=8", "--standalone",
        PROXY_SCRIPT,
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--size", str(size_elems),
        "--compute-mul", str(compute_mul),
        "--out", str(out_file),
    ]
    if overlap:
        cmd.append("--overlap")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/repo")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"iteration_proxy exited {result.returncode}")

    if Path(out_file).is_file():
        return [float(x) for x in Path(out_file).read_text().strip().split("\n")
                if x.strip()]
    return []


# ---------------------------------------------------------------------------
# Main Modal function
# ---------------------------------------------------------------------------
@app.function(
    name="browser-networking-tests",
    image=rl_image,
    gpu="A100:8",
    timeout=14400,      # 4 hours for full experiment
    volumes={VOLUME_PATH: volume},
)
def run_overlap_experiment():
    """Full paper experiment: sequential vs overlap vs compute intensity."""
    results_dir = Path(VOLUME_PATH) / "overlap_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "sequential": {},
        "overlap": {},
        "compute_sweep": {},
    }

    # ==================================================================
    # BLOCK 1: Sequential baselines (no overlap, default compute-mul)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  BLOCK 1: Sequential baselines (no overlap)")
    print(f"{'='*70}\n", flush=True)

    for size_label, size_elems in MSG_SIZES:
        size_data = {}
        for cfg_label, cfg_env in CONFIGS:
            tag = f"seq_{size_label}_{cfg_label}"
            out_path = str(results_dir / f"times_{tag}.txt")
            print(f"  [SEQ] {cfg_label:15s} @ {size_label}", flush=True)
            times = run_proxy(cfg_env, ITERS, WARMUP, out_path, size_elems,
                              overlap=False)
            mean_t = sum(times) / len(times) if times else 0
            size_data[cfg_label] = round(mean_t, 3)
            print(f"         mean={mean_t:.3f} ms, n={len(times)}", flush=True)
        all_results["sequential"][size_label] = size_data

        # Quick summary
        auto_t = size_data.get("auto", 0)
        best_cfg = min(size_data, key=size_data.get)
        best_t = size_data[best_cfg]
        gap = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0
        print(f"    => Best: {best_cfg} ({best_t:.3f}ms), "
              f"AUTO gap: {gap:+.1f}%\n", flush=True)

    volume.commit()

    # ==================================================================
    # BLOCK 2: Overlap baselines (same sizes, with --overlap)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  BLOCK 2: Overlap baselines (compute + allreduce concurrent)")
    print(f"{'='*70}\n", flush=True)

    for size_label, size_elems in MSG_SIZES:
        size_data = {}
        for cfg_label, cfg_env in CONFIGS:
            tag = f"ovlp_{size_label}_{cfg_label}"
            out_path = str(results_dir / f"times_{tag}.txt")
            print(f"  [OVL] {cfg_label:15s} @ {size_label}", flush=True)
            times = run_proxy(cfg_env, ITERS, WARMUP, out_path, size_elems,
                              overlap=True)
            mean_t = sum(times) / len(times) if times else 0
            size_data[cfg_label] = round(mean_t, 3)
            print(f"         mean={mean_t:.3f} ms, n={len(times)}", flush=True)
        all_results["overlap"][size_label] = size_data

        # Compare with sequential
        seq_auto = all_results["sequential"].get(size_label, {}).get("auto", 0)
        ovl_auto = size_data.get("auto", 0)
        auto_t = size_data.get("auto", 0)
        best_cfg = min(size_data, key=size_data.get)
        best_t = size_data[best_cfg]
        gap = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0
        print(f"    => Best: {best_cfg} ({best_t:.3f}ms), "
              f"AUTO gap: {gap:+.1f}%", flush=True)
        print(f"    => SEQ AUTO: {seq_auto:.3f}ms, OVL AUTO: {ovl_auto:.3f}ms\n",
              flush=True)

    volume.commit()

    # ==================================================================
    # BLOCK 3: Compute intensity sweep (overlap mode, varying matmul size)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  BLOCK 3: Compute intensity sweep (overlap mode)")
    print(f"{'='*70}\n", flush=True)

    for size_label, size_elems in SWEEP_SIZES:
        all_results["compute_sweep"][size_label] = {}

        for cmul_label, cmul_val in COMPUTE_MULS:
            # Use fewer iters for heavy/extreme compute
            iters = HEAVY_ITERS if cmul_val >= 6144 else ITERS
            warmup = HEAVY_WARMUP if cmul_val >= 6144 else WARMUP

            cmul_data = {}
            print(f"\n  [SWEEP] size={size_label}, compute={cmul_label} "
                  f"(matmul {cmul_val}x{cmul_val})", flush=True)

            for cfg_label, cfg_env in CONFIGS:
                tag = f"sweep_{size_label}_{cmul_label}_{cfg_label}"
                out_path = str(results_dir / f"times_{tag}.txt")
                print(f"    {cfg_label:15s}", end="", flush=True)
                times = run_proxy(cfg_env, iters, warmup, out_path, size_elems,
                                  overlap=True, compute_mul=cmul_val)
                mean_t = sum(times) / len(times) if times else 0
                cmul_data[cfg_label] = round(mean_t, 3)
                print(f"  mean={mean_t:.3f} ms", flush=True)

            all_results["compute_sweep"][size_label][cmul_label] = cmul_data

            # Summary for this (size, compute) point
            auto_t = cmul_data.get("auto", 0)
            best_cfg = min(cmul_data, key=cmul_data.get)
            best_t = cmul_data[best_cfg]
            gap = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0
            print(f"    => Best: {best_cfg} ({best_t:.3f}ms), "
                  f"AUTO gap: {gap:+.1f}%", flush=True)

        volume.commit()

    # ==================================================================
    # Save master results
    # ==================================================================
    (results_dir / "overlap_experiment_results.json").write_text(
        json.dumps(all_results, indent=2))
    volume.commit()

    # Collect all files for local download
    all_files = {}
    for f in results_dir.iterdir():
        if f.is_file():
            all_files[f.name] = f.read_text()

    return {"results": all_results, "files": all_files}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    out = run_overlap_experiment.remote()

    results_dir = Path(__file__).parent / "results" / "overlap_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save all files locally
    for filename, content in out["files"].items():
        (results_dir / filename).write_text(content)
    print(f"Saved {len(out['files'])} result files to {results_dir}")

    results = out["results"]

    # ------------------------------------------------------------------
    # Print Block 1 & 2 comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  SEQUENTIAL vs OVERLAP comparison")
    print(f"{'='*100}")
    print(f"\n{'Size':<8s} | {'--- Sequential ---':^40s} | {'--- Overlap ---':^40s}")
    print(f"{'':8s} | {'AUTO':>8s} {'Best':>12s} {'Gap':>6s} | "
          f"{'AUTO':>8s} {'Best':>12s} {'Gap':>6s} | {'Winner flipped?'}")
    print("-" * 105)

    for size_label, _ in MSG_SIZES:
        seq = results["sequential"].get(size_label, {})
        ovl = results["overlap"].get(size_label, {})

        seq_auto = seq.get("auto", 0)
        seq_best_cfg = min(seq, key=seq.get) if seq else "?"
        seq_best_t = seq.get(seq_best_cfg, 0)
        seq_gap = (seq_auto - seq_best_t) / seq_auto * 100 if seq_auto > 0 else 0

        ovl_auto = ovl.get("auto", 0)
        ovl_best_cfg = min(ovl, key=ovl.get) if ovl else "?"
        ovl_best_t = ovl.get(ovl_best_cfg, 0)
        ovl_gap = (ovl_auto - ovl_best_t) / ovl_auto * 100 if ovl_auto > 0 else 0

        flipped = "YES <<<" if seq_best_cfg != ovl_best_cfg else "no"

        print(f"{size_label:<8s} | {seq_auto:>8.3f} {seq_best_cfg:>12s} {seq_gap:>+5.1f}% | "
              f"{ovl_auto:>8.3f} {ovl_best_cfg:>12s} {ovl_gap:>+5.1f}% | {flipped}")

    # ------------------------------------------------------------------
    # Print Block 3 compute intensity sweep
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  COMPUTE INTENSITY SWEEP (overlap mode)")
    print(f"{'='*100}")

    for size_label in results["compute_sweep"]:
        print(f"\n  Message size: {size_label}")
        print(f"  {'Compute':>10s} | {'AUTO':>8s} {'tree_s':>8s} {'tree_l':>8s} "
              f"{'ring_s':>8s} {'ring_l':>8s} | {'Best':>12s} {'Gap':>6s}")
        print(f"  {'-'*85}")

        for cmul_label, cmul_data in results["compute_sweep"][size_label].items():
            auto = cmul_data.get("auto", 0)
            ts = cmul_data.get("tree_simple", 0)
            tl = cmul_data.get("tree_ll128", 0)
            rs = cmul_data.get("ring_simple", 0)
            rl = cmul_data.get("ring_ll128", 0)
            best_cfg = min(cmul_data, key=cmul_data.get)
            best_t = cmul_data[best_cfg]
            gap = (auto - best_t) / auto * 100 if auto > 0 else 0
            print(f"  {cmul_label:>10s} | {auto:>8.3f} {ts:>8.3f} {tl:>8.3f} "
                  f"{rs:>8.3f} {rl:>8.3f} | {best_cfg:>12s} {gap:>+5.1f}%")

    # Auto-run analysis
    print("\nGenerating plots...")
    analyze_script = Path(__file__).parent / "analyze_overlap.py"
    if analyze_script.exists():
        result = subprocess.run(
            [sys.executable, str(analyze_script)],
            cwd=str(Path(__file__).parent),
        )
        if result.returncode != 0:
            print("Warning: analyze_overlap.py failed", file=sys.stderr)
        else:
            print("Done — all plots saved to results/overlap_experiment/")
    else:
        print("Note: analyze_overlap.py not found, skipping plot generation")
