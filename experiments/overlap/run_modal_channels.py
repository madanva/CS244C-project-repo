"""
Modal app: Phase 4 — NCCL channel count (CTA) sweep under overlap on 8x A100.

Key experiment: NCCL uses multiple CUDA thread blocks (CTAs) per collective,
each occupying one SM. Under compute-communication overlap, these CTAs compete
with the compute kernel for SM resources. By sweeping NCCL_MAX_CTAS we can:
  1. Find the Pareto-optimal CTA count for overlap workloads
  2. Show that NCCL's default CTA count is suboptimal under overlap
  3. Demonstrate the SM contention mechanism quantitatively

Experiment blocks:
  A. CTA sweep under OVERLAP at key message sizes (the main result)
  B. CTA sweep SEQUENTIAL (same sizes) as control
  C. CTA × compute intensity interaction at one key size

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
# CTA counts to sweep
# ---------------------------------------------------------------------------
CTA_COUNTS = [1, 2, 4, 8, 12, 16, 24, 32]

# ---------------------------------------------------------------------------
# Message sizes (subset — focus on where we saw the biggest overlap effects)
# ---------------------------------------------------------------------------
MSG_SIZES = [
    ("1MB",   262_144),
    ("4MB",   1_048_576),
    ("16MB",  4_194_304),
    ("64MB",  16_777_216),
]

# ---------------------------------------------------------------------------
# NCCL configs
# ---------------------------------------------------------------------------
CONFIGS = [
    ("auto",        {}),
    ("tree_simple", {"NCCL_ALGO": "Tree",  "NCCL_PROTO": "Simple"}),
    ("tree_ll128",  {"NCCL_ALGO": "Tree",  "NCCL_PROTO": "LL128"}),
    ("ring_simple", {"NCCL_ALGO": "Ring",  "NCCL_PROTO": "Simple"}),
    ("ring_ll128",  {"NCCL_ALGO": "Ring",  "NCCL_PROTO": "LL128"}),
]

# Compute intensity levels for Block C
COMPUTE_MULS_C = [
    ("medium",  2048),
    ("default", 4096),
    ("heavy",   6144),
]

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
ITERS = 50
WARMUP = 10
HEAVY_ITERS = 30
HEAVY_WARMUP = 5


# ---------------------------------------------------------------------------
# Proxy launcher
# ---------------------------------------------------------------------------
def run_proxy(env_extra, iters, warmup, out_file, size_elems,
              overlap=False, compute_mul=4096, max_ctas=None):
    """Run the iteration proxy; return list of float times in ms."""
    env = {**os.environ, **env_extra}

    # Clean NCCL env vars not explicitly set
    for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_TUNER_PLUGIN",
                "NCCL_TUNER_REWARD_FILE", "NCCL_MIN_CTAS", "NCCL_MAX_CTAS"):
        if key not in env_extra:
            env.pop(key, None)

    # Set CTA limits if specified
    if max_ctas is not None:
        env["NCCL_MIN_CTAS"] = str(max_ctas)
        env["NCCL_MAX_CTAS"] = str(max_ctas)

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
    timeout=14400,      # 4 hours
    volumes={VOLUME_PATH: volume},
    retries=modal.Retries(max_retries=3, initial_delay=5.0),
)
def run_channel_experiment():
    """CTA sweep: sequential vs overlap, plus CTA × compute interaction."""
    results_dir = Path(VOLUME_PATH) / "channel_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if preempted
    checkpoint_file = results_dir / "channel_experiment_results.json"
    if checkpoint_file.exists():
        all_results = json.loads(checkpoint_file.read_text())
        print("Resumed from checkpoint!", flush=True)
    else:
        all_results = {
            "overlap_cta_sweep": {},
            "sequential_cta_sweep": {},
            "cta_compute_interaction": {},
        }

    # ==================================================================
    # BLOCK A: CTA sweep under OVERLAP (the main result)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  BLOCK A: CTA sweep under OVERLAP")
    print(f"{'='*70}\n", flush=True)

    for size_label, size_elems in MSG_SIZES:
        if size_label not in all_results["overlap_cta_sweep"]:
            all_results["overlap_cta_sweep"][size_label] = {}

        for cta in CTA_COUNTS:
            # Skip if already completed
            if str(cta) in all_results["overlap_cta_sweep"].get(size_label, {}):
                print(f"\n  [OVL CTA={cta:2d}] @ {size_label} — SKIPPED (cached)",
                      flush=True)
                continue

            cta_data = {}
            print(f"\n  [OVL CTA={cta:2d}] @ {size_label}", flush=True)

            for cfg_label, cfg_env in CONFIGS:
                tag = f"ovlcta_{size_label}_{cta}cta_{cfg_label}"
                out_path = str(results_dir / f"times_{tag}.txt")
                print(f"    {cfg_label:15s}", end="", flush=True)
                times = run_proxy(cfg_env, ITERS, WARMUP, out_path, size_elems,
                                  overlap=True, max_ctas=cta)
                mean_t = sum(times) / len(times) if times else 0
                cta_data[cfg_label] = round(mean_t, 3)
                print(f"  mean={mean_t:.3f} ms", flush=True)

            all_results["overlap_cta_sweep"][size_label][str(cta)] = cta_data

            # Summary
            auto_t = cta_data.get("auto", 0)
            best_cfg = min(cta_data, key=cta_data.get)
            best_t = cta_data[best_cfg]
            gap = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0
            print(f"    => Best: {best_cfg} ({best_t:.3f}ms), "
                  f"AUTO gap: {gap:+.1f}%", flush=True)

        # Checkpoint after each size
        checkpoint_file.write_text(json.dumps(all_results, indent=2))
        volume.commit()

    # ==================================================================
    # BLOCK B: CTA sweep SEQUENTIAL (control)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  BLOCK B: CTA sweep SEQUENTIAL (control)")
    print(f"{'='*70}\n", flush=True)

    for size_label, size_elems in MSG_SIZES:
        if size_label not in all_results["sequential_cta_sweep"]:
            all_results["sequential_cta_sweep"][size_label] = {}

        for cta in CTA_COUNTS:
            # Skip if already completed
            if str(cta) in all_results["sequential_cta_sweep"].get(size_label, {}):
                print(f"\n  [SEQ CTA={cta:2d}] @ {size_label} — SKIPPED (cached)",
                      flush=True)
                continue

            cta_data = {}
            print(f"\n  [SEQ CTA={cta:2d}] @ {size_label}", flush=True)

            for cfg_label, cfg_env in CONFIGS:
                tag = f"seqcta_{size_label}_{cta}cta_{cfg_label}"
                out_path = str(results_dir / f"times_{tag}.txt")
                print(f"    {cfg_label:15s}", end="", flush=True)
                times = run_proxy(cfg_env, ITERS, WARMUP, out_path, size_elems,
                                  overlap=False, max_ctas=cta)
                mean_t = sum(times) / len(times) if times else 0
                cta_data[cfg_label] = round(mean_t, 3)
                print(f"  mean={mean_t:.3f} ms", flush=True)

            all_results["sequential_cta_sweep"][size_label][str(cta)] = cta_data

        # Checkpoint after each size
        checkpoint_file.write_text(json.dumps(all_results, indent=2))
        volume.commit()

    # ==================================================================
    # BLOCK C: CTA × compute intensity interaction (4MB, overlap)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  BLOCK C: CTA × compute intensity interaction (4MB, overlap)")
    print(f"{'='*70}\n", flush=True)

    target_size = 1_048_576  # 4MB
    target_label = "4MB"

    for cmul_label, cmul_val in COMPUTE_MULS_C:
        if cmul_label not in all_results["cta_compute_interaction"]:
            all_results["cta_compute_interaction"][cmul_label] = {}
        iters = HEAVY_ITERS if cmul_val >= 6144 else ITERS
        warmup = HEAVY_WARMUP if cmul_val >= 6144 else WARMUP

        for cta in CTA_COUNTS:
            # Skip if already completed
            if str(cta) in all_results["cta_compute_interaction"].get(cmul_label, {}):
                print(f"\n  [CTA×COMPUTE] cta={cta}, compute={cmul_label} — SKIPPED",
                      flush=True)
                continue

            cta_data = {}
            print(f"\n  [CTA×COMPUTE] cta={cta}, compute={cmul_label} "
                  f"({cmul_val}x{cmul_val}) @ {target_label}", flush=True)

            for cfg_label, cfg_env in CONFIGS:
                tag = f"ctacomp_{cmul_label}_{cta}cta_{cfg_label}"
                out_path = str(results_dir / f"times_{tag}.txt")
                print(f"    {cfg_label:15s}", end="", flush=True)
                times = run_proxy(cfg_env, iters, warmup, out_path, target_size,
                                  overlap=True, compute_mul=cmul_val,
                                  max_ctas=cta)
                mean_t = sum(times) / len(times) if times else 0
                cta_data[cfg_label] = round(mean_t, 3)
                print(f"  mean={mean_t:.3f} ms", flush=True)

            all_results["cta_compute_interaction"][cmul_label][str(cta)] = cta_data

        # Checkpoint after each compute level
        checkpoint_file.write_text(json.dumps(all_results, indent=2))
        volume.commit()

    # ==================================================================
    # Save master results
    # ==================================================================
    (results_dir / "channel_experiment_results.json").write_text(
        json.dumps(all_results, indent=2))
    volume.commit()

    # Collect all files
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
    out = run_channel_experiment.remote()

    results_dir = Path(__file__).parent / "results" / "channel_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in out["files"].items():
        (results_dir / filename).write_text(content)
    print(f"Saved {len(out['files'])} result files to {results_dir}")

    results = out["results"]

    # ------------------------------------------------------------------
    # Block A: Overlap CTA sweep table
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  BLOCK A: CTA SWEEP UNDER OVERLAP")
    print(f"{'='*100}")

    for size_label in results["overlap_cta_sweep"]:
        print(f"\n  Message size: {size_label}")
        print(f"  {'CTAs':>6s} | {'AUTO':>8s} {'tree_s':>8s} {'tree_l':>8s} "
              f"{'ring_s':>8s} {'ring_l':>8s} | {'Best':>12s} {'Gap':>6s}")
        print(f"  {'-'*80}")

        for cta, cta_data in results["overlap_cta_sweep"][size_label].items():
            auto = cta_data.get("auto", 0)
            ts = cta_data.get("tree_simple", 0)
            tl = cta_data.get("tree_ll128", 0)
            rs = cta_data.get("ring_simple", 0)
            rl = cta_data.get("ring_ll128", 0)
            best = min(cta_data, key=cta_data.get)
            best_t = cta_data[best]
            gap = (auto - best_t) / auto * 100 if auto > 0 else 0
            print(f"  {cta:>6s} | {auto:>8.3f} {ts:>8.3f} {tl:>8.3f} "
                  f"{rs:>8.3f} {rl:>8.3f} | {best:>12s} {gap:>+5.1f}%")

    # ------------------------------------------------------------------
    # Find optimal CTA count per size (overlap)
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  OPTIMAL CTA COUNT (overlap vs sequential)")
    print(f"{'='*100}")
    print(f"\n  {'Size':<8s} | {'OVL optimal CTAs':>18s} {'OVL best ms':>12s} | "
          f"{'SEQ optimal CTAs':>18s} {'SEQ best ms':>12s} | {'OVL default':>12s} {'SEQ default':>12s}")
    print(f"  {'-'*100}")

    for size_label in results["overlap_cta_sweep"]:
        # Find best (cta, config) combo under overlap
        ovl_best_cta, ovl_best_cfg, ovl_best_t = "?", "?", 999
        for cta, data in results["overlap_cta_sweep"][size_label].items():
            for cfg, t in data.items():
                if t < ovl_best_t:
                    ovl_best_t, ovl_best_cta, ovl_best_cfg = t, cta, cfg

        # Find best under sequential
        seq_best_cta, seq_best_cfg, seq_best_t = "?", "?", 999
        for cta, data in results["sequential_cta_sweep"].get(size_label, {}).items():
            for cfg, t in data.items():
                if t < seq_best_t:
                    seq_best_t, seq_best_cta, seq_best_cfg = t, cta, cfg

        # Default CTA (closest to NCCL default ~8-16)
        ovl_default = results["overlap_cta_sweep"][size_label].get("8", {}).get("auto", 0)
        seq_default = results["sequential_cta_sweep"].get(size_label, {}).get("8", {}).get("auto", 0)

        print(f"  {size_label:<8s} | {ovl_best_cta:>3s} CTAs/{ovl_best_cfg:<12s} {ovl_best_t:>8.3f} ms | "
              f"{seq_best_cta:>3s} CTAs/{seq_best_cfg:<12s} {seq_best_t:>8.3f} ms | "
              f"{ovl_default:>8.3f} ms   {seq_default:>8.3f} ms")

    # Auto-run analysis
    print("\nGenerating plots...")
    analyze_script = Path(__file__).parent / "analyze_channels.py"
    if analyze_script.exists():
        result = subprocess.run(
            [sys.executable, str(analyze_script)],
            cwd=str(Path(__file__).parent),
        )
        if result.returncode != 0:
            print("Warning: analyze_channels.py failed", file=sys.stderr)
        else:
            print("Done — all plots saved to results/channel_experiment/")
    else:
        print("Note: analyze_channels.py not found, skipping plot generation")
