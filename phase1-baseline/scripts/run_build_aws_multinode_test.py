#!/usr/bin/env python3
"""
Reuse one existing AWS node by IP: copy source (optional), run build, download build/.
No launch/terminate — use the same node every time for fast build troubleshooting.

Usage:
  1. Start a node once (e.g. run_build_aws_multinode.py --no-terminate, or leave one running).
  2. Run this script with that node's public IP:
       python run_build_aws_multinode_test.py --ip 54.1.2.3 --private-key ~/.ssh/key.pem
  3. Iterate on the build; re-run this script as needed (no instance startup/shutdown).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Only what we need from run_aws_multinode (no boto3)
from run_aws_multinode import (
    BUILD_CACHE_DIR,
    REPO_ROOT,
    get_nccl_mpi_build_cmd,
    scp_to_node,
    ssh_run_cmd,
    wait_for_ssh,
)


def scp_from_node(
    host: str,
    remote_path: str,
    local_parent: Path,
    private_key_path: str,
    user: str = "ubuntu",
) -> None:
    """Copy remote_path from host into local_parent (creates local_parent/<basename(remote_path)>)."""
    local_parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp", "-r", "-i", private_key_path,
        "-o", "StrictHostKeyChecking=no",
        f"{user}@{host}:{remote_path}",
        str(local_parent),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build nccl-tests with MPI on an existing node (no launch/terminate)."
    )
    parser.add_argument("--ip", required=True, help="Public IP of the existing EC2 node")
    parser.add_argument("--private-key", required=True, help="Path to .pem for SSH/SCP")
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip copying nccl-tests source (source already on node from previous run)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading build/ to local (only run build on node)",
    )
    args = parser.parse_args()

    private_key_path = Path(args.private_key).expanduser().resolve()
    if not private_key_path.is_file():
        sys.exit(f"Private key not found: {private_key_path}")

    ip = args.ip.strip()
    remote_repo = "/home/ubuntu/repo"

    print("Checking SSH...")
    wait_for_ssh(ip, str(private_key_path), timeout_sec=30)

    if not args.skip_copy:
        nccl_tests_src = REPO_ROOT / "nccl-tests"
        if not nccl_tests_src.is_dir():
            sys.exit("nccl-tests not found. Run: git submodule update --init --recursive")
        print("Copying nccl-tests source to node...")
        ssh_run_cmd(ip, ["mkdir", "-p", remote_repo], str(private_key_path))
        scp_to_node(nccl_tests_src, f"{remote_repo}/", ip, str(private_key_path))
    else:
        print("Skipping copy (--skip-copy).")

    print("Building nccl-tests with MPI=1...")
    ssh_run_cmd(ip, get_nccl_mpi_build_cmd(), str(private_key_path))

    if not args.skip_download:
        print("Downloading build/ to local build_cache (nccl-tests-mpi)...")
        if BUILD_CACHE_DIR.exists():
            shutil.rmtree(BUILD_CACHE_DIR)
        BUILD_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
        scp_from_node(
            ip,
            "/home/ubuntu/repo/nccl-tests/build",
            BUILD_CACHE_DIR.parent,
            str(private_key_path),
        )
        if not (BUILD_CACHE_DIR / "all_reduce_perf_mpi").is_file():
            sys.exit("Downloaded build missing all_reduce_perf_mpi")
        print(f"Cached build saved to: {BUILD_CACHE_DIR} (MPI=1)")
    else:
        print("Skipping download (--skip-download).")


if __name__ == "__main__":
    main()
