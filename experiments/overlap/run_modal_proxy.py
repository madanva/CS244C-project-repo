"""
Modal app: Phase 3 iteration-level proxy on 8x A100.
Runs the training-step proxy under different NCCL configs (AUTO, Simple, LL128)
and records iteration times. Job name: browser-networking-test.
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Image: CUDA 12 + PyTorch (NCCL via PyTorch)
proxy_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget")
    .run_commands("pip install --upgrade pip")
    .pip_install("torch")
    .add_local_dir(REPO_ROOT, remote_path="/repo")
)

volume = modal.Volume.from_name("cs244c-nccl-results", create_if_missing=True)
VOLUME_PATH = "/results"

app = modal.App("browser-networking-tests")

PROXY_SCRIPT = "/repo/phase3-iteration-proxy/a100-8gpu-new/iteration_proxy.py"
CONFIGS = [
    ("auto", {}),
    ("simple", {"NCCL_PROTO": "Simple"}),
    ("ll128", {"NCCL_PROTO": "LL128"}),
]


@app.function(
    name="browser-networking-test",
    image=proxy_image,
    gpu="A100:8",
    timeout=1800,
    volumes={VOLUME_PATH: volume},
)
def run_iteration_proxy_all_configs():
    """Run iteration proxy for AUTO, Simple, and LL128; save iteration times per config."""
    results_dir = Path(VOLUME_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)
    all_times = {}

    for config_name, env_add in CONFIGS:
        print(f"--- Config: {config_name} ---", flush=True)
        env = {**os.environ, **env_add}
        out_file = results_dir / f"iteration_times_{config_name}.txt"
        cmd = [
            "python", "-m", "torch.distributed.run",
            "--nproc_per_node=8",
            "--standalone",
            PROXY_SCRIPT,
            "--iters", "50",
            "--warmup", "5",
            "--out", str(out_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/repo")
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"iteration_proxy exited {result.returncode} for config {config_name}")
        print(result.stdout, flush=True)
        if out_file.is_file():
            all_times[config_name] = out_file.read_text().strip().split("\n")
        volume.commit()

    return all_times


@app.local_entrypoint()
def main():
    """Run proxy for all configs and write iteration time files to results/."""
    out = run_iteration_proxy_all_configs.remote()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    for config_name, lines in out.items():
        path = results_dir / f"iteration_times_{config_name}.txt"
        path.write_text("\n".join(lines) + "\n")
        print(f"Wrote {path}")
    print("Done.")
