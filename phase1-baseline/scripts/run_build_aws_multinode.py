#!/usr/bin/env python3
"""
Spin up one AWS EC2 node (same hardware as run_aws_multinode default), build nccl-tests
with MPI=1, download the build/ directory to the local build_cache, then terminate.

Does not assume any node is already running: launches a fresh instance each run (unless
--no-terminate). Uses the same build logic as run_build_aws_multinode_test.py (NCCL/MPI
discovery, LIBRARY_PATH, apt install of OpenMPI and libnccl if missing) so the build
succeeds on a clean DLAMI.

Cached build is saved under build_cache/nccl-tests-mpi/build/ (MPI=1). Run this once
(or when nccl-tests or instance type changes); run_aws_multinode.py will then use the
cached build and skip building on each node.

Example:
  python run_build_aws_multinode.py --key-name my-key --private-key ~/.ssh/my-key.pem
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from botocore.exceptions import ClientError
except ImportError:
    ClientError = None  # type: ignore[misc, assignment]

# Import shared helpers and constants from run_aws_multinode
from run_aws_multinode import (
    BUILD_CACHE_DIR,
    REPO_ROOT,
    SCRIPT_DIR,
    delete_orphaned_volumes,
    ensure_security_group,
    get_latest_dlami_gpu_ubuntu,
    get_nccl_mpi_build_cmd,
    get_volume_ids_for_instances,
    launch_instances,
    scp_to_node,
    ssh_run_cmd,
    wait_for_instances,
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
        description="Build nccl-tests with MPI on one AWS node and save to local build_cache."
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--ami", default="", help="AMI ID; default: latest Deep Learning GPU Ubuntu 22.04")
    parser.add_argument(
        "--instance-type",
        default="g5.xlarge",
        help="Instance type, must match run_aws_multinode (default: g5.xlarge)",
    )
    parser.add_argument("--key-name", required=True, help="EC2 key pair name")
    parser.add_argument("--private-key", required=True, help="Path to .pem for SSH")
    parser.add_argument("--vpc-id", default="", help="VPC ID for security group")
    parser.add_argument("--no-terminate", action="store_true", help="Leave instance running")
    args = parser.parse_args()

    private_key_path = Path(args.private_key).expanduser().resolve()
    if not private_key_path.is_file():
        sys.exit(f"Private key not found: {private_key_path}")

    try:
        import boto3
    except ImportError:
        sys.exit(
            "boto3 not found. Use: mamba env create -f phase1-baseline/scripts/environment.yml && mamba activate aws-multinode"
        )

    ec2 = boto3.client("ec2", region_name=args.region)
    instance_ids: list[str] = []

    try:
        ami = args.ami
        if not ami:
            print("Looking up latest Deep Learning GPU AMI (Ubuntu 22.04)...")
            ami = get_latest_dlami_gpu_ubuntu(ec2, args.region)
            print(f"Using AMI: {ami}")

        vpc_id = args.vpc_id or None
        security_group_id = ensure_security_group(ec2, vpc_id, args.region)

        instance_ids = launch_instances(
            ec2,
            ami=ami,
            instance_type=args.instance_type,
            num_nodes=1,
            key_name=args.key_name,
            region=args.region,
            security_group_id=security_group_id,
            placement_group=None,
        )

        print("Waiting for instance to be running and have IPs...")
        instances = wait_for_instances(ec2, instance_ids)
        ip = instances[0]["PublicIpAddress"]
        print(f"Instance IP: {ip}")

        print("Waiting for SSH...")
        wait_for_ssh(ip, str(private_key_path))

        remote_repo = "/home/ubuntu/repo"
        nccl_tests_src = REPO_ROOT / "nccl-tests"
        if not nccl_tests_src.is_dir():
            sys.exit("nccl-tests not found. Run: git submodule update --init --recursive")

        print("Copying nccl-tests source to node...")
        ssh_run_cmd(ip, ["mkdir", "-p", remote_repo], str(private_key_path))
        scp_to_node(nccl_tests_src, f"{remote_repo}/", ip, str(private_key_path))

        # Same build as run_build_aws_multinode_test: discovers NCCL/MPI, installs via apt if missing
        print("Building nccl-tests with MPI=1 (discover NCCL/MPI, install if needed)...")
        ssh_run_cmd(ip, get_nccl_mpi_build_cmd(), str(private_key_path))

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

    finally:
        if instance_ids and not args.no_terminate:
            volume_ids = get_volume_ids_for_instances(ec2, instance_ids)
            print("Terminating instance...")
            try:
                ec2.terminate_instances(InstanceIds=instance_ids)
                print("Done. Instance terminated.")
                if volume_ids:
                    time.sleep(5)
                    delete_orphaned_volumes(ec2, volume_ids)
            except Exception as e:
                if ClientError and isinstance(e, ClientError):
                    if e.response.get("Error", {}).get("Code") == "InvalidInstanceID.NotFound":
                        print("Instance already gone; skipping terminate.")
                    else:
                        print(f"Terminate failed: {e}", file=sys.stderr)
                else:
                    print(f"Terminate failed: {e}", file=sys.stderr)
        elif instance_ids and args.no_terminate:
            print("Left instance running (--no-terminate):", instance_ids)


if __name__ == "__main__":
    main()
