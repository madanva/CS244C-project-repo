"""
Modal app: Phase 4 — message-size transition boundary experiment on 8x A100.

Sweeps NCCL AllReduce across message sizes that target known transition
boundaries (LL->LL128, LL128->Simple, tree->ring).  At each size, runs all
5 fixed (algo, protocol) configs plus a UCB1 bandit to discover where NCCL
AUTO is suboptimal.

The iteration proxy uses sequential compute->allreduce (no overlap) so the
full allreduce latency is visible in iteration time at every message size.

App name: browser-networking-tests.  Function name: browser-networking-tests.
"""

import json
import math
import os
import random
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
# Message sizes targeting known NCCL transition boundaries.
# Each entry: (label, num_float32_elements).
# bytes = elements * 4.
# ---------------------------------------------------------------------------
MSG_SIZES = [
    ("32KB",   8_192),         #   32 KB — deep in LL territory
    ("64KB",   16_384),        #   64 KB — LL -> LL128 transition
    ("256KB",  65_536),        #  256 KB — LL128 mid-range
    ("1MB",    262_144),       #    1 MB — LL128 -> Simple transition
    ("2MB",    524_288),       #    2 MB — transition zone
    ("4MB",    1_048_576),     #    4 MB — Simple territory (previous test)
    ("16MB",   4_194_304),     #   16 MB — tree -> ring region
    ("64MB",   16_777_216),    #   64 MB — ring territory
    ("256MB",  67_108_864),    #  256 MB — large ring / upper transition
]

# ---------------------------------------------------------------------------
# Fixed configs (the bandit arms)
# ---------------------------------------------------------------------------
ARMS = [
    ("auto",        {}),
    ("tree_simple", {"NCCL_ALGO": "Tree",  "NCCL_PROTO": "Simple"}),
    ("tree_ll128",  {"NCCL_ALGO": "Tree",  "NCCL_PROTO": "LL128"}),
    ("ring_simple", {"NCCL_ALGO": "Ring",  "NCCL_PROTO": "Simple"}),
    ("ring_ll128",  {"NCCL_ALGO": "Ring",  "NCCL_PROTO": "LL128"}),
]

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
BASELINE_ITERS = 50
BASELINE_WARMUP = 10

EPISODE_ITERS = 15
EPISODE_WARMUP = 5
NUM_EPISODES = 15          # per message size

UCB_C = 1.414


# ---------------------------------------------------------------------------
# Proxy launcher
# ---------------------------------------------------------------------------
def run_proxy(env_extra, iters, warmup, out_file, size_elems=None):
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
        "--out", str(out_file),
    ]
    if size_elems is not None:
        cmd.extend(["--size", str(size_elems)])

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/repo")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"iteration_proxy exited {result.returncode}")

    if Path(out_file).is_file():
        return [float(x) for x in Path(out_file).read_text().strip().split("\n")
                if x.strip()]
    return []


# ---------------------------------------------------------------------------
# UCB1 selection
# ---------------------------------------------------------------------------
def select_ucb1(arm_stats, total_pulls, c=UCB_C):
    n = len(arm_stats)
    for i in range(n):
        if arm_stats[i]["count"] == 0:
            return i
    ln_t = math.log(total_pulls)
    best_idx, best_score = 0, float("inf")
    for i in range(n):
        ni = arm_stats[i]["count"]
        mean_i = arm_stats[i]["sum"] / ni
        score = mean_i - c * math.sqrt(ln_t / ni)
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Run UCB1 bandit at a single message size
# ---------------------------------------------------------------------------
def run_ucb1_at_size(size_label, size_elems, results_dir):
    """Return (episode_log, arm_stats_summary, all_times)."""
    arm_stats = [{"count": 0, "sum": 0.0} for _ in ARMS]
    episode_log = []
    all_times = []
    total_pulls = 0

    for ep in range(NUM_EPISODES):
        arm_idx = select_ucb1(arm_stats, total_pulls)
        arm_label, arm_env = ARMS[arm_idx]

        out_file = str(results_dir / f"ucb1_{size_label}_ep{ep:02d}.txt")
        times = run_proxy(arm_env, EPISODE_ITERS, EPISODE_WARMUP, out_file,
                          size_elems=size_elems)
        mean_t = sum(times) / len(times) if times else 999.0

        arm_stats[arm_idx]["count"] += 1
        arm_stats[arm_idx]["sum"] += mean_t
        total_pulls += 1
        all_times.extend(times)

        episode_log.append({
            "episode": ep,
            "arm_idx": arm_idx,
            "arm_label": arm_label,
            "mean_ms": round(mean_t, 3),
        })

        arm_str = "  ".join(
            f"{ARMS[i][0]}={arm_stats[i]['sum']/arm_stats[i]['count']:.2f}({arm_stats[i]['count']})"
            if arm_stats[i]["count"] > 0 else f"{ARMS[i][0]}=?(0)"
            for i in range(len(ARMS))
        )
        print(f"  [ucb1@{size_label}] Ep {ep:2d}: arm={arm_label:<15s} "
              f"mean={mean_t:.2f} ms  |  {arm_str}", flush=True)

    stats_summary = []
    for i, (label, _) in enumerate(ARMS):
        s = arm_stats[i]
        mean = s["sum"] / s["count"] if s["count"] > 0 else None
        stats_summary.append({
            "arm": label,
            "count": s["count"],
            "mean_ms": round(mean, 3) if mean else None,
        })
    return episode_log, stats_summary, all_times


# ---------------------------------------------------------------------------
# Main Modal function
# ---------------------------------------------------------------------------
@app.function(
    name="browser-networking-tests",
    image=rl_image,
    gpu="A100:8",
    timeout=10800,      # 3 hours for full sweep
    volumes={VOLUME_PATH: volume},
)
def run_phase4_experiment():
    """Sweep message sizes: baselines + UCB1 bandit at each size."""
    results_dir = Path(VOLUME_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Master summary: {size_label: {config: mean_ms}}
    sweep_summary = {}

    for size_label, size_elems in MSG_SIZES:
        size_bytes = size_elems * 4
        print(f"\n{'#'*70}\n# Size: {size_label} ({size_bytes:,} bytes, "
              f"{size_elems:,} float32 elements)\n{'#'*70}", flush=True)

        size_data = {}

        # ---- Baselines: all 5 fixed configs ----
        for arm_label, arm_env in ARMS:
            tag = f"{size_label}_{arm_label}"
            print(f"\n  Baseline: {arm_label} @ {size_label}", flush=True)
            out_path = str(results_dir / f"iteration_times_{tag}.txt")
            times = run_proxy(arm_env, BASELINE_ITERS, BASELINE_WARMUP,
                              out_path, size_elems=size_elems)
            mean_t = sum(times) / len(times) if times else 0
            print(f"    mean={mean_t:.3f} ms, n={len(times)}", flush=True)
            size_data[arm_label] = round(mean_t, 3)

        # ---- UCB1 bandit ----
        print(f"\n  UCB1 bandit @ {size_label} ({NUM_EPISODES} episodes)",
              flush=True)
        ep_log, arm_stats, bandit_times = run_ucb1_at_size(
            size_label, size_elems, results_dir)

        # Converged mean (2nd half of bandit iterations)
        if bandit_times:
            half = bandit_times[len(bandit_times) // 2:]
            bandit_conv = round(sum(half) / len(half), 3) if half else 0
        else:
            bandit_conv = 0
        size_data["ucb1_converged"] = bandit_conv

        # Find bandit's preferred arm
        best_arm = max(arm_stats, key=lambda s: s["count"])
        size_data["ucb1_preferred_arm"] = best_arm["arm"]
        size_data["ucb1_preferred_mean"] = best_arm["mean_ms"]

        # Save per-size bandit data
        (results_dir / f"ucb1_{size_label}_episode_log.json").write_text(
            json.dumps(ep_log, indent=2))
        (results_dir / f"ucb1_{size_label}_arm_stats.json").write_text(
            json.dumps(arm_stats, indent=2))

        sweep_summary[size_label] = size_data

        # Print per-size summary
        auto_mean = size_data.get("auto", 0)
        best_fixed_label = min(
            [l for l, _ in ARMS],
            key=lambda l: size_data.get(l, 999))
        best_fixed_mean = size_data[best_fixed_label]
        gap_pct = ((auto_mean - best_fixed_mean) / auto_mean * 100
                   if auto_mean > 0 else 0)
        print(f"\n  --- {size_label} Summary ---", flush=True)
        for l, _ in ARMS:
            m = size_data.get(l, 0)
            marker = " <-- AUTO" if l == "auto" else ""
            marker = " <-- BEST" if l == best_fixed_label else marker
            print(f"    {l:<15s}: {m:.3f} ms{marker}", flush=True)
        print(f"    UCB1 converged:  {bandit_conv:.3f} ms "
              f"(preferred: {best_arm['arm']})", flush=True)
        print(f"    AUTO vs best:    {gap_pct:+.1f}%", flush=True)

    # Save master sweep summary
    (results_dir / "phase4_sweep_summary.json").write_text(
        json.dumps(sweep_summary, indent=2))

    volume.commit()

    # Collect all result files so the local entrypoint can save them
    all_files = {}
    for f in results_dir.iterdir():
        if f.is_file():
            all_files[f.name] = f.read_text()

    return {"sweep_summary": sweep_summary, "files": all_files}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    out = run_phase4_experiment.remote()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Save ALL result files locally (iteration times, bandit logs, summary)
    for filename, content in out["files"].items():
        (results_dir / filename).write_text(content)
    print(f"Saved {len(out['files'])} result files to {results_dir}")

    # Print summary table
    summary = out["sweep_summary"]
    print(f"\n{'Size':<8s} {'AUTO':>9s} {'tree_s':>9s} {'tree_l':>9s} "
          f"{'ring_s':>9s} {'ring_l':>9s} {'UCB1':>9s} {'Best':>12s} "
          f"{'Gap':>7s}")
    print("-" * 90)
    for size_label, data in summary.items():
        auto = data.get("auto", 0)
        ts = data.get("tree_simple", 0)
        tl = data.get("tree_ll128", 0)
        rs = data.get("ring_simple", 0)
        rl = data.get("ring_ll128", 0)
        ucb = data.get("ucb1_converged", 0)
        configs = {"auto": auto, "tree_simple": ts, "tree_ll128": tl,
                   "ring_simple": rs, "ring_ll128": rl}
        best_label = min(configs, key=configs.get)
        best_val = configs[best_label]
        gap = (auto - best_val) / auto * 100 if auto > 0 else 0
        print(f"{size_label:<8s} {auto:>9.3f} {ts:>9.3f} {tl:>9.3f} "
              f"{rs:>9.3f} {rl:>9.3f} {ucb:>9.3f} {best_label:>12s} "
              f"{gap:>+6.1f}%")

    # Auto-run analysis and plot generation
    print("\nGenerating plots...")
    analyze_script = Path(__file__).parent / "analyze_and_plot.py"
    result = subprocess.run(
        [sys.executable, str(analyze_script)],
        cwd=str(Path(__file__).parent),
    )
    if result.returncode != 0:
        print("Warning: analyze_and_plot.py failed", file=sys.stderr)
    else:
        print("Done — all plots saved to results/")
