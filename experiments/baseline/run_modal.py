"""
Modal app: run NCCL all-reduce benchmark on 8x A100 (phase1-baseline style).
Job name: browser-networking-test.
Produces output in the same format as L40S results for plot_nccl_bw.py.
"""

import subprocess
import sys
import argparse
from pathlib import Path

import modal

# Repo root (CS244C-Research) for mounting nccl-tests
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Image: CUDA 12 devel + NCCL (for building and running nccl-tests).
# Add repo at build/startup so nccl-tests submodule is available (same pattern as fine_tuned_model).
nccl_image = (
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

# Volume for persisting benchmark output
volume = modal.Volume.from_name("cs244c-nccl-results", create_if_missing=True)
VOLUME_PATH = "/results"

app = modal.App("browser-networking-tests")



def run_nccl_allreduce_ngpu(num_gpus):
    """Build nccl-tests and run all_reduce_perf with specified number of GPUs (8B–128MB, factor 2)."""
    nccl_tests = Path("/repo/nccl-tests")
    if not nccl_tests.is_dir():
        raise RuntimeError(
            "nccl-tests not found at /repo/nccl-tests. Ensure the submodule is checked out: "
            "git submodule update --init --recursive"
        )

    cuda_home = "/usr/local/cuda"
    nccl_home = "/usr"

    # Build
    subprocess.run(
        ["make", "MPI=0", f"CUDA_HOME={cuda_home}", f"NCCL_HOME={nccl_home}", "-j"],
        cwd=nccl_tests,
        check=True,
        env={**__import__("os").environ, "CUDA_HOME": cuda_home, "NCCL_HOME": nccl_home},
    )

    binary = nccl_tests / "build" / "all_reduce_perf"
    if not binary.exists():
        raise RuntimeError(f"Build failed: {binary} not found")

    # Same flags as L40S (see run-baseline-tutorial.md) but -g <num_gpus>
    cmd = [
        str(binary),
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", str(num_gpus),
    ]
    env = {
        **__import__("os").environ,
        "CUDA_HOME": cuda_home,
        "NCCL_HOME": nccl_home,
        "LD_LIBRARY_PATH": ":".join([
            f"{nccl_home}/lib",
            f"{nccl_home}/lib/x86_64-linux-gnu",
            __import__("os").environ.get("LD_LIBRARY_PATH", ""),
        ]).strip(":"),
    }
    result = subprocess.run(
        cmd,
        cwd=nccl_tests,
        capture_output=True,
        text=True,
        env=env,
    )
    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print(stderr, file=sys.stderr)
        raise RuntimeError(f"all_reduce_perf exited {result.returncode}")

    # Write to volume
    out_file = f"results_{num_gpus}gpu_allreduce.txt"
    vol_file = Path(VOLUME_PATH) / out_file
    vol_file.parent.mkdir(parents=True, exist_ok=True)
    vol_file.write_text(stdout)
    volume.commit()

    # Print so user can copy to results/results_{num_gpus}gpu_allreduce.txt locally
    print(f"=== NCCL output (save to results/results_{num_gpus}gpu_allreduce.txt) ===")
    print(stdout)
    print("=== end ===")
    return stdout

def get_modal_gpu_string(arch, num_gpus):
    return f"{arch}:{num_gpus}" if num_gpus > 0 else f"{arch}:1"

def make_modal_function(arch, num_gpus):
    return app.function(
        name="browser-networking-test",
        image=nccl_image,
        gpu=get_modal_gpu_string(arch, num_gpus),
        timeout=3600,
        volumes={VOLUME_PATH: volume},
    )(lambda: run_nccl_allreduce_ngpu(num_gpus))


@app.local_entrypoint()
def main():
    """Run the benchmark and optionally write result to local results/."""
    parser = argparse.ArgumentParser(description='Run NCCL all-reduce benchmark on Modal with configurable GPU architecture and count.')
    parser.add_argument('--arch', type=str, default='A100', help='GPU architecture to use (e.g., A100, L40S)')
    parser.add_argument('--gpus', type=int, default=8, help='Number of GPUs to use (default: 8)')
    args = parser.parse_args()
    arch = args.arch.upper()
    num_gpus = args.gpus

    modal_func = make_modal_function(arch, num_gpus)
    out = modal_func.remote()
    results_dir = Path(__file__).parent / "results" / f"{arch.lower()}-{num_gpus}gpu-results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "results.txt"
    out_path.write_text(out)
    print(f"Wrote results to: {out_path}")
