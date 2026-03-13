"""
Modal app: Phase 2 contention on 8x A100.
Runs GPU stress (low/medium/high) on all 8 GPUs while running NCCL AllReduce.
Job name: browser-networking-test.
Same idea as L40S run_nccl_with_contention.sh but for 8 GPUs on Modal.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

# Repo root (CS244C-Research)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Image: CUDA 12 devel + NCCL + repo (same pattern as phase1 and fine_tuned_model)
contention_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget", "build-essential")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb",
        "dpkg -i /tmp/cuda-keyring.deb",
        "apt-get update",
        "apt-get install -y libnccl2 libnccl-dev",
    )
    .add_local_dir(REPO_ROOT, remote_path="/repo")
)

volume = modal.Volume.from_name("cs244c-nccl-results", create_if_missing=True)
VOLUME_PATH = "/results"

app = modal.App("browser-networking-tests")


def _base_env(cuda_home: str, nccl_home: str) -> dict:
    return {
        **os.environ,
        "CUDA_HOME": cuda_home,
        "NCCL_HOME": nccl_home,
        "LD_LIBRARY_PATH": ":".join([
            f"{nccl_home}/lib",
            f"{nccl_home}/lib/x86_64-linux-gnu",
            os.environ.get("LD_LIBRARY_PATH", ""),
        ]).strip(":"),
    }


@app.function(
    name="browser-networking-test",
    image=contention_image,
    gpu="A100:8",
    timeout=3600,
    volumes={VOLUME_PATH: volume},
)
def run_contention_all_levels():
    """Build nccl-tests and gpu_stress_benchmark, then run contention for low, medium, high."""
    nccl_tests = Path("/repo/nccl-tests")
    phase2_a100 = Path("/repo/phase2-contention/a100-8gpu-new")
    stress_src = phase2_a100 / "gpu_stress_benchmark.cu"
    if not nccl_tests.is_dir():
        raise RuntimeError("nccl-tests not found. Run: git submodule update --init --recursive")
    if not stress_src.is_file():
        raise RuntimeError(f"gpu_stress_benchmark.cu not found at {stress_src}")

    cuda_home = "/usr/local/cuda"
    nccl_home = "/usr"
    base_env = _base_env(cuda_home, nccl_home)

    # Build nccl-tests
    subprocess.run(
        ["make", "MPI=0", f"CUDA_HOME={cuda_home}", f"NCCL_HOME={nccl_home}", "-j"],
        cwd=nccl_tests,
        check=True,
        env=base_env,
    )
    nccl_binary = nccl_tests / "build" / "all_reduce_perf"
    if not nccl_binary.exists():
        raise RuntimeError(f"Build failed: {nccl_binary} not found")

    # Build GPU stress benchmark (nvcc + cublas)
    stress_bin = Path("/tmp/gpu_stress_benchmark")
    subprocess.run(
        [
            "nvcc", "-lcublas", str(stress_src),
            "-o", str(stress_bin),
        ],
        check=True,
        env=base_env,
    )

    results = {}
    for level in ("low", "medium", "high"):
        print(f"--- Contention level: {level} ---")
        # Start one stress process per GPU (each sees one GPU as device 0)
        stress_procs = []
        for gpu_id in range(8):
            env = {**base_env, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            p = subprocess.Popen(
                [str(stress_bin), level],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            stress_procs.append(p)
        time.sleep(2)

        # Run NCCL AllReduce (same flags as L40S, -g 8)
        result = subprocess.run(
            [
                str(nccl_binary),
                "-b", "8", "-e", "128M", "-f", "2", "-g", "8",
            ],
            cwd=nccl_tests,
            capture_output=True,
            text=True,
            env=base_env,
        )
        for p in stress_procs:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"all_reduce_perf exited {result.returncode} for level {level}")
        results[level] = result.stdout
        out_name = f"results_8gpu_allreduce_contended_{level}.txt"
        (Path(VOLUME_PATH) / out_name).write_text(result.stdout)
        volume.commit()
        print(f"Saved {out_name}")

    return results


@app.local_entrypoint()
def main():
    """Run contention for all levels and write result files to results/."""
    results = run_contention_all_levels.remote()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    for level, stdout in results.items():
        out_path = results_dir / f"results_8gpu_allreduce_contended_{level}.txt"
        out_path.write_text(stdout)
        print(f"Wrote {out_path}")
    print("Done.")
