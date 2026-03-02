#!/usr/bin/env python3
"""
Run NCCL all-reduce benchmark on 2 (or N) AWS EC2 GPU nodes, then terminate instances.

Works with A100 (p4d), A10G (g5), or T4 (g4dn). Use smaller instance types to minimize
cost and vCPU quota (e.g. g5.xlarge = 1 GPU, 4 vCPUs per node).

Flow:
  1. Launch N EC2 instances with a Deep Learning AMI.
  2. Wait for instances to be running and SSH-ready.
  3. Copy cached build to nodes (from --build-dir or default build_cache). This script never builds on nodes.
  4. Run all_reduce_perf for every (algorithm, protocol) combination (excluding LL/LL128 with
     CollNet/NVLS) plus one AUTO run; save each run to a separate .txt in the results folder.
  5. Terminate all instances to minimize cost.

Requirements:
  - AWS CLI configured (aws configure) or env vars AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.
  - An EC2 key pair in the target region; pass --key-name and --private-key (path to .pem).
  - boto3: use the aws-multinode mamba env (mamba env create -f phase1-baseline/scripts/environment.yml, then mamba activate aws-multinode).

Examples:
  # Use cached build from run_build_aws_multinode.py, then benchmark (2 nodes):
  python run_benchmark_aws_multinode.py --key-name my-key --private-key ~/.ssh/my-key.pem --build-dir phase1-baseline/scripts/build_cache/nccl-tests-mpi/build --num-nodes 2

  # Multinode with PyTorch backend (like Modal; avoids Open MPI/PMIx issues):
  python run_benchmark_aws_multinode.py --key-name my-key --private-key ~/.ssh/my-key.pem --build-dir build_cache/nccl-tests-mpi/build --num-nodes 2 --multinode-backend torch --auto-only

  # Sequential + overlap (writes _sequential and _overlap .txt files for ingest):
  python run_benchmark_aws_multinode.py --key-name my-key --private-key ~/.ssh/my-key.pem --num-nodes 2 --multinode-backend torch --mode both

  # Overlap-only, leave instances running (no terminate; terminate manually to avoid cost):
  python run_benchmark_aws_multinode.py --key-name cs244c-aws-multinode-experiments-2-28 --private-key ~/.ssh/cs244c-aws-multinode-experiments-2-28.pem --num-nodes 2 --multinode-backend torch --mode overlap --no-terminate

  # Use existing VMs (no launch/terminate; like run_build_aws_multinode_test.py):
  python run_benchmark_aws_multinode.py --existing-ips 54.1.2.3,54.4.5.6 --private-key ~/.ssh/my-key.pem --build-dir build_cache/nccl-tests-mpi/build --multinode-backend torch --auto-only

  # Using default cache (build_cache/nccl-tests-mpi/build):
  python run_benchmark_aws_multinode.py --key-name my-key --private-key ~/.ssh/my-key.pem --num-nodes 2

  # 2 nodes × 1 T4 each (2 GPUs, 8 vCPUs):
  python run_aws_multinode.py --key-name my-key --private-key ~/.ssh/my-key.pem --instance-type g4dn.xlarge --num-nodes 2

  # A100 (requires higher vCPU quota):
  python run_aws_multinode.py --key-name my-key --private-key ~/.ssh/my-key.pem --instance-type p4d.24xlarge --num-nodes 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

try:
    from botocore.exceptions import ClientError
except ImportError:
    ClientError = None  # type: ignore[misc, assignment]

SCRIPT_DIR = Path(__file__).resolve().parent

# Cached nccl-tests build (MPI=1) from run_build_aws_multinode.py or run_build_aws_multinode_test.py.
# Path: build_cache/nccl-tests-mpi/build/ (name indicates MPI-enabled); copied to nodes as .../nccl-tests/build.
BUILD_CACHE_DIR = SCRIPT_DIR / "build_cache" / "nccl-tests-mpi" / "build"

# GPUs per node for common AWS instance types (for hostfile slots and mpirun -N)
GPUS_PER_INSTANCE_TYPE = {
    # A100
    "p4d.24xlarge": 8,
    "p4de.24xlarge": 8,
    "p5.48xlarge": 8,
    # A10G (g5)
    "g5.xlarge": 1,
    "g5.2xlarge": 1,
    "g5.4xlarge": 1,
    "g5.8xlarge": 1,
    "g5.12xlarge": 4,
    "g5.16xlarge": 1,
    "g5.24xlarge": 4,
    "g5.48xlarge": 8,
    # T4 (g4dn)
    "g4dn.xlarge": 1,
    "g4dn.2xlarge": 1,
    "g4dn.4xlarge": 1,
    "g4dn.8xlarge": 1,
    "g4dn.12xlarge": 4,
    "g4dn.16xlarge": 1,
}

# NCCL algorithms and protocols for all_reduce_perf (env vars NCCL_ALGO, NCCL_PROTO).
# LL/LL128 are not used with CollNet/NVLS (see NCCL docs).
NCCL_ALGORITHMS = ["Ring", "Tree", "CollnetChain", "CollnetDirect", "NVLS", "NVLSTree", "PAT"]
NCCL_PROTOCOLS = ["Simple", "LL", "LL128"]
# Algorithms for which LL/LL128 should be skipped (CollNet/NVLS family).
ALGOS_NO_LL = frozenset({"CollnetChain", "CollnetDirect", "NVLS", "NVLSTree"})

# PyTorch all-reduce benchmark script (Modal-style: MASTER_ADDR/PORT, RANK, WORLD_SIZE).
# Runs on each process; reads env and prints nccl-tests-like table from rank 0.
# Set OVERLAP_MODE=1 to overlap compute (matmul) with all_reduce; 0 or unset = sequential.
TORCH_ALLREDUCE_BENCH_SCRIPT = r'''
import os
import sys
import time
import torch
import torch.distributed as dist

def main():
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        overlap = os.environ.get("OVERLAP_MODE", "0") == "1"
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                                init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        # Sizes like nccl-tests: 8 to 128M, factor 2
        min_b, max_b = 8, 128 * 1024 * 1024
        sizes = []
        b = min_b
        while b <= max_b:
            sizes.append(b)
            b *= 2
        iters, warmup = 20, 1
        compute_mul = 4096  # matmul size for overlap (like Modal)
        lines = []
        if rank == 0:
            mode_label = "overlap" if overlap else "sequential"
            lines.append("# PyTorch NCCL all_reduce (env rendezvous, like Modal)")
            lines.append(f"# world_size={world_size} minBytes={min_b} maxBytes={max_b} mode={mode_label}")
            lines.append("# size(B)     time(us)   algbw(GB/s)  busbw(GB/s)")
        for size in sizes:
            n = size // 4  # float32
            t = torch.randn(n, device=device, dtype=torch.float32) / world_size
            if not overlap:
                for _ in range(warmup):
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(iters):
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
            else:
                compute_stream = torch.cuda.Stream(device=device)
                comm_stream = torch.cuda.Stream(device=device)
                for _ in range(warmup):
                    with torch.cuda.stream(compute_stream):
                        a = torch.randn(compute_mul, compute_mul, device=device)
                        b = torch.randn(compute_mul, compute_mul, device=device)
                        torch.matmul(a, b)
                    with torch.cuda.stream(comm_stream):
                        tmp = t.clone()
                        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(iters):
                    with torch.cuda.stream(compute_stream):
                        a = torch.randn(compute_mul, compute_mul, device=device)
                        b = torch.randn(compute_mul, compute_mul, device=device)
                        torch.matmul(a, b)
                    with torch.cuda.stream(comm_stream):
                        tmp = t.clone()
                        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
                    torch.cuda.synchronize()
            elapsed_us = (time.perf_counter() - start) / iters * 1e6
            algbw = size * 2.0 / (elapsed_us * 1e3)  # approximate
            busbw = size * 2.0 * (world_size - 1) / world_size / (elapsed_us * 1e3)
            if rank == 0:
                lines.append(f"{size:12d} {elapsed_us:10.2f} {algbw:12.2f} {busbw:12.2f}")
        dist.destroy_process_group()
        if rank == 0:
            print("\n".join(lines), flush=True)
    except Exception as e:
        import traceback
        r = os.environ.get("RANK", "?")
        print(f"Rank {r} failed: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''


def get_benchmark_configs() -> list[tuple[str, dict[str, str] | None]]:
    """Return (tag, env_additions) for each run. env_additions is None for AUTO (no overrides)."""
    configs: list[tuple[str, dict[str, str] | None]] = []
    for algo in NCCL_ALGORITHMS:
        for proto in NCCL_PROTOCOLS:
            if algo in ALGOS_NO_LL and proto in ("LL", "LL128"):
                continue
            tag = f"{algo}_{proto}".replace(" ", "").lower()
            configs.append((tag, {"NCCL_ALGO": algo, "NCCL_PROTO": proto}))
    configs.append(("auto", None))  # NCCL's automatic algorithm/protocol selection
    return configs


def get_latest_dlami_gpu_ubuntu(ec2, region: str) -> str:
    """Find the latest AWS Deep Learning Base GPU AMI (Ubuntu 22.04) in the region."""
    # DLAMI names vary; try common patterns (Amazon-owned)
    for name_pattern in [
        "*Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*",
        "*Deep Learning Base GPU AMI (Ubuntu 22.04)*",
        "*Deep Learning OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*",
    ]:
        r = ec2.describe_images(
            Owners=["amazon"],
            Filters=[
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
                {"Name": "name", "Values": [name_pattern]},
            ],
        )
        images = r.get("Images", [])
        if images:
            latest = max(images, key=lambda x: x["CreationDate"])
            return latest["ImageId"]
    raise RuntimeError(
        f"No Deep Learning GPU Ubuntu 22.04 AMI found in {region}. "
        "Specify an AMI explicitly with --ami (e.g. from EC2 Console → AMIs → search 'Deep Learning')."
    )


def get_ami_root_device(ec2, ami: str) -> str:
    """Return the root device name for the AMI (e.g. /dev/sda1 or /dev/xvda)."""
    r = ec2.describe_images(ImageIds=[ami])
    if not r.get("Images"):
        return "/dev/sda1"
    root = r["Images"][0].get("RootDeviceName") or "/dev/sda1"
    return root


def launch_instances(
    ec2,
    *,
    ami: str,
    instance_type: str,
    num_nodes: int,
    key_name: str,
    region: str,
    security_group_id: str | None,
    placement_group: str | None,
) -> list[dict]:
    """Launch num_nodes EC2 instances; return list of instance dicts with Id, PrivateIpAddress, etc."""
    placement = {}
    if placement_group:
        placement["GroupName"] = placement_group

    root_device = get_ami_root_device(ec2, ami)
    # Root volume: ensure DeleteOnTermination so we don't leave EBS volumes (and cost) behind.
    run_args = {
        "ImageId": ami,
        "InstanceType": instance_type,
        "MinCount": num_nodes,
        "MaxCount": num_nodes,
        "KeyName": key_name,
        "BlockDeviceMappings": [
            {"DeviceName": root_device, "Ebs": {"DeleteOnTermination": True}},
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": "cs244c-nccl-multinode"},
                    {"Key": "Purpose", "Value": "NCCL benchmark"},
                ],
            }
        ],
    }
    if placement:
        run_args["Placement"] = placement
    if security_group_id:
        run_args["SecurityGroupIds"] = [security_group_id]

    r = ec2.run_instances(**run_args)
    ids = [inst["InstanceId"] for inst in r["Instances"]]
    print(f"Launched instances: {ids}")
    return ids


def ensure_security_group(ec2, vpc_id: str | None, region: str) -> str:
    """Create or reuse a security group that allows SSH and internal traffic for MPI."""
    name = "cs244c-nccl-multinode-sg"
    try:
        existing = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [name]}]
        )
        if existing["SecurityGroups"]:
            sg_id = existing["SecurityGroups"][0]["GroupId"]
            print(f"Using existing security group: {sg_id}")
            return sg_id
    except Exception:
        pass

    if not vpc_id:
        vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
        if not vpcs["Vpcs"]:
            raise RuntimeError("No default VPC found; specify --vpc-id")
        vpc_id = vpcs["Vpcs"][0]["VpcId"]

    r = ec2.create_security_group(
        GroupName=name,
        Description="SSH + internal for NCCL multi-node",
        VpcId=vpc_id,
    )
    sg_id = r["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {"FromPort": 22, "ToPort": 22, "IpProtocol": "tcp", "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}]},
            {"FromPort": 1, "ToPort": 65535, "IpProtocol": "tcp", "UserIdGroupPairs": [{"GroupId": sg_id}]},
            {"FromPort": 1, "ToPort": 65535, "IpProtocol": "udp", "UserIdGroupPairs": [{"GroupId": sg_id}]},
        ],
    )
    print(f"Created security group: {sg_id}")
    return sg_id


def get_volume_ids_for_instances(ec2, instance_ids: list[str]) -> list[str]:
    """Return EBS volume IDs attached to the given instances (so we can delete them if orphaned)."""
    if not instance_ids:
        return []
    try:
        r = ec2.describe_instances(InstanceIds=instance_ids)
    except Exception:
        return []
    volume_ids: list[str] = []
    for res in r.get("Reservations", []):
        for inst in res.get("Instances", []):
            for bdm in inst.get("BlockDeviceMappings", []):
                if "Ebs" in bdm and "VolumeId" in bdm["Ebs"]:
                    volume_ids.append(bdm["Ebs"]["VolumeId"])
    return volume_ids


def delete_orphaned_volumes(ec2, volume_ids: list[str]) -> None:
    """Delete volumes that are in 'available' state (left behind after instance termination)."""
    if not volume_ids:
        return
    try:
        r = ec2.describe_volumes(VolumeIds=volume_ids)
    except Exception:
        return
    for vol in r.get("Volumes", []):
        if vol.get("State") == "available":
            try:
                ec2.delete_volume(VolumeId=vol["VolumeId"])
                print(f"Deleted orphaned volume: {vol['VolumeId']}")
            except Exception as e:
                if ClientError and isinstance(e, ClientError):
                    if e.response.get("Error", {}).get("Code") != "InvalidVolume.NotFound":
                        print(f"Failed to delete volume {vol['VolumeId']}: {e}", file=sys.stderr)
                else:
                    print(f"Failed to delete volume {vol['VolumeId']}: {e}", file=sys.stderr)


def wait_for_instances(ec2, instance_ids: list[str], timeout_sec: int = 600) -> list[dict]:
    """Wait until all instances are running and have private IPs; return instance info."""
    start = time.time()
    not_found_retries = 0
    max_not_found_retries = 6  # ~1 min for eventual consistency after launch
    while time.time() - start < timeout_sec:
        try:
            r = ec2.describe_instances(InstanceIds=instance_ids)
        except Exception as e:
            if ClientError and isinstance(e, ClientError):
                err_code = e.response.get("Error", {}).get("Code", "")
                if err_code == "InvalidInstanceID.NotFound":
                    not_found_retries += 1
                    if not_found_retries > max_not_found_retries:
                        raise RuntimeError(
                            f"Instance IDs not found after {max_not_found_retries} retries (instances may have been "
                            f"terminated or wrong region): {instance_ids}"
                        ) from e
                    time.sleep(10)
                    continue
            raise
        instances = []
        for res in r["Reservations"]:
            instances.extend(res["Instances"])
        if len(instances) != len(instance_ids):
            time.sleep(5)
            continue
        if all(i["State"]["Name"] == "running" for i in instances) and all(
            i.get("PrivateIpAddress") for i in instances
        ) and all(i.get("PublicIpAddress") for i in instances):
            return instances
        time.sleep(10)
    raise RuntimeError(
        f"Instances not ready within {timeout_sec}s (need running + private + public IPs): {instance_ids}"
    )


def wait_for_ssh(host: str, private_key_path: str, user: str = "ubuntu", timeout_sec: int = 300) -> None:
    """Poll until SSH to host succeeds."""
    start = time.time()
    ssh_cmd = [
        "ssh",
        "-i", private_key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        "-o", "BatchMode=yes",
        f"{user}@{host}",
        "echo", "ok",
    ]
    while time.time() - start < timeout_sec:
        try:
            subprocess.run(ssh_cmd, check=True, capture_output=True)
            return
        except subprocess.CalledProcessError:
            time.sleep(5)
    raise RuntimeError(f"SSH to {host} did not become ready within {timeout_sec}s")


def scp_to_node(
    local_path: Path,
    remote_dir: str,
    host: str,
    private_key_path: str,
    user: str = "ubuntu",
) -> None:
    """Copy local_path (file or dir) to host:remote_dir."""
    local = str(local_path)
    remote = f"{user}@{host}:{remote_dir}"
    cmd = [
        "scp", "-r", "-i", private_key_path,
        "-o", "StrictHostKeyChecking=no",
        local, remote,
    ]
    subprocess.run(cmd, check=True)


def ssh_run_cmd(
    host: str,
    command: list[str] | str,
    private_key_path: str,
    user: str = "ubuntu",
    check: bool = True,
) -> subprocess.CompletedProcess:
    args = ["ssh", "-i", private_key_path, "-o", "StrictHostKeyChecking=no", f"{user}@{host}"]
    if isinstance(command, str):
        args += ["bash", "-lc", command]
    else:
        args += command
    result = subprocess.run(args, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"SSH command failed with exit code {result.returncode}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NCCL all-reduce on 2 (or N) AWS A100 nodes, then terminate instances."
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--ami", default="", help="AMI ID; if not set, look up latest Deep Learning GPU Ubuntu 22.04")
    parser.add_argument("--instance-type", default="g5.xlarge", help="Instance type (default: g5.xlarge = 1 A10G, 4 vCPUs; use p4d.24xlarge for 8x A100)")
    parser.add_argument("--num-nodes", type=int, default=2, help="Number of nodes (default: 2)")
    parser.add_argument(
        "--existing-ips",
        metavar="IP1,IP2,...",
        default=None,
        help="Use existing VMs: comma-separated public IPs (skips launch/terminate; like run_build_aws_multinode_test.py)",
    )
    parser.add_argument(
        "--private-ips",
        metavar="IP1,IP2,...",
        default=None,
        help="With --existing-ips: comma-separated private IPs for hostfile (if omitted, discovered via SSH)",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="With --existing-ips: GPUs per node (if omitted, discovered via nvidia-smi on first node)",
    )
    parser.add_argument("--key-name", default=None, help="EC2 key pair name (required when launching; not needed with --existing-ips)")
    parser.add_argument("--private-key", required=True, help="Path to the private key file (e.g. .pem) for SSH/SCP")
    parser.add_argument("--vpc-id", default="", help="VPC ID for security group; default uses default VPC")
    parser.add_argument("--placement-group", default="", help="Optional placement group for low latency")
    parser.add_argument("--no-terminate", action="store_true", help="Do not terminate instances after run (for debugging)")
    parser.add_argument("--results-dir", default=None, help="Directory to save results (default: phase1-baseline/scripts/results/aws-multinode)")
    parser.add_argument(
        "--build-dir",
        default=None,
        metavar="DIR",
        help="Path to cached nccl-tests build (must contain all_reduce_perf_mpi). Default: script's build_cache/nccl-tests-mpi/build",
    )
    parser.add_argument("--auto-only", action="store_true", help="Run only the AUTO benchmark (sanity check; skip algo/proto combinations)")
    parser.add_argument(
        "--multinode-backend",
        choices=("mpi", "torch"),
        default="mpi",
        help="Multinode launch: mpi=Open MPI mpirun (default); torch=PyTorch env rendezvous (like Modal, avoids PMIx)",
    )
    parser.add_argument(
        "--mode",
        choices=("sequential", "overlap", "both"),
        default="sequential",
        help="Torch backend only: sequential (no overlap), overlap (compute+comm overlapped), or both (default: sequential). MPI backend always runs sequential.",
    )
    args = parser.parse_args()

    private_key_path = Path(args.private_key).expanduser().resolve()
    if not private_key_path.is_file():
        sys.exit(f"Private key not found: {private_key_path}")

    use_existing = bool(args.existing_ips)
    if use_existing:
        if not args.key_name:
            pass  # key-name not needed when using existing IPs
        public_ips = [x.strip() for x in args.existing_ips.split(",") if x.strip()]
        if not public_ips:
            sys.exit("--existing-ips must have at least one IP")
        num_nodes = len(public_ips)
    else:
        if not args.key_name:
            sys.exit("--key-name is required when launching instances")
        try:
            import boto3
        except ImportError:
            sys.exit(
                "boto3 not found. Use the aws-multinode mamba env: "
                "mamba env create -f phase1-baseline/scripts/environment.yml && mamba activate aws-multinode"
            )

    ec2 = None
    instance_ids: list[str] = []
    security_group_id: str | None = None
    if not use_existing:
        ec2 = __import__("boto3").client("ec2", region_name=args.region)

    try:
        if use_existing:
            # Use existing VMs: no launch/terminate (like run_build_aws_multinode_test.py)
            print(f"Using existing IPs (no launch/terminate): {public_ips}")
            for ip in public_ips:
                print(f"Waiting for SSH on {ip}...")
                wait_for_ssh(ip, str(private_key_path), timeout_sec=30)
            if args.private_ips:
                private_ips = [x.strip() for x in args.private_ips.split(",") if x.strip()]
                if len(private_ips) != num_nodes:
                    sys.exit(f"--private-ips must have {num_nodes} entries (one per --existing-ips)")
            else:
                print("Discovering private IPs via SSH...")
                private_ips = []
                for ip in public_ips:
                    r = ssh_run_cmd(ip, ["hostname -I | awk '{print $1}'"], str(private_key_path), check=False)
                    first = (r.stdout or "").strip().split()
                    private_ips.append(first[0] if first else ip)
            if args.gpus_per_node is not None:
                gpus_per_node = args.gpus_per_node
            else:
                r = ssh_run_cmd(public_ips[0], ["nvidia-smi --query-gpu=name --format=csv,noheader"], str(private_key_path), check=False)
                lines = [l for l in (r.stdout or "").strip().split("\n") if l.strip()]
                gpus_per_node = len(lines) if r.returncode == 0 and lines else 1
                print(f"Discovered {gpus_per_node} GPU(s) per node")
            total_gpus = num_nodes * gpus_per_node
            print(f"Instance public IPs (for SSH): {public_ips}")
            print(f"Instance private IPs (for hostfile): {private_ips}")
            results_dir = Path(args.results_dir or SCRIPT_DIR / "results" / "aws-multinode")
            results_dir.mkdir(parents=True, exist_ok=True)
            cluster_specs_dir = results_dir / "cluster_specs"
            cluster_specs_dir.mkdir(parents=True, exist_ok=True)
            instance_type_for_topology = "existing"
        else:
            # Resolve AMI and launch
            ami = args.ami
            if not ami:
                print("Looking up latest Deep Learning GPU AMI (Ubuntu 22.04)...")
                ami = get_latest_dlami_gpu_ubuntu(ec2, args.region)
                print(f"Using AMI: {ami}")
            vpc_id = args.vpc_id or None
            security_group_id = ensure_security_group(ec2, vpc_id, args.region)
            placement_group = args.placement_group or None
            instance_ids = launch_instances(
                ec2,
                ami=ami,
                instance_type=args.instance_type,
                num_nodes=args.num_nodes,
                key_name=args.key_name,
                region=args.region,
                security_group_id=security_group_id,
                placement_group=placement_group,
            )
            print("Waiting for instances to be running and have IPs...")
            instances = wait_for_instances(ec2, instance_ids)
            nodes = sorted(instances, key=lambda i: i["PrivateIpAddress"])
            public_ips = [n["PublicIpAddress"] for n in nodes]
            private_ips = [n["PrivateIpAddress"] for n in nodes]
            num_nodes = args.num_nodes
            print(f"Instance public IPs (for SSH): {public_ips}")
            print(f"Instance private IPs (for hostfile): {private_ips}")
            for ip in public_ips:
                print(f"Waiting for SSH on {ip}...")
                wait_for_ssh(ip, str(private_key_path))
            results_dir = Path(args.results_dir or SCRIPT_DIR / "results" / "aws-multinode")
            results_dir.mkdir(parents=True, exist_ok=True)
            cluster_specs_dir = results_dir / "cluster_specs"
            cluster_specs_dir.mkdir(parents=True, exist_ok=True)
            gpus_per_node = GPUS_PER_INSTANCE_TYPE.get(args.instance_type, 8)
            total_gpus = num_nodes * gpus_per_node
            instance_type_for_topology = args.instance_type

        # Collect topology and GPU specs from each node (nvidia-smi) into cluster_specs/
        print("Collecting cluster topology and GPU specs...")
        hostfile_lines = [f"{ip} slots={gpus_per_node}" for ip in private_ips]
        hostfile_content = "\n".join(hostfile_lines) + "\n"
        topology_lines = [
            "=== Cluster topology ===",
            f"instance_type={instance_type_for_topology}",
            f"num_nodes={num_nodes}",
            f"gpus_per_node={gpus_per_node}",
            f"total_gpus={total_gpus}",
            "",
            "hostfile:",
            hostfile_content,
            "node_ips (private): " + ", ".join(private_ips),
            "",
        ]
        for i, ip in enumerate(public_ips):
            for label, cmd in [
                ("nvidia_smi", "nvidia-smi"),
                ("nvidia_smi_query", "nvidia-smi -q"),
                ("nvidia_smi_topo", "nvidia-smi topo -m"),
            ]:
                result = ssh_run_cmd(ip, [cmd], str(private_key_path), check=False)
                out = result.stdout or ""
                if result.returncode != 0:
                    out = f"(exit {result.returncode})\n{result.stderr or ''}\n{out}"
                fname = cluster_specs_dir / f"node{i}_{ip}_{label}.txt"
                fname.write_text(out)
            # One-line summary for topology file
            result = ssh_run_cmd(ip, ["nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"], str(private_key_path), check=False)
            summary = (result.stdout or "").strip() or "(nvidia-smi query failed)"
            topology_lines.append(f"node{i} ({ip}): {summary}")
        (cluster_specs_dir / "cluster_topology.txt").write_text("\n".join(topology_lines) + "\n")
        print(f"Cluster topology and specs written to: {cluster_specs_dir}")

        # Use cached build only when using MPI backend. Torch backend does not need nccl-tests binary.
        remote_repo = "/home/ubuntu/repo"
        remote_nccl = f"{remote_repo}/nccl-tests"
        if args.multinode_backend == "mpi":
            build_dir = (Path(args.build_dir).expanduser().resolve() if args.build_dir else BUILD_CACHE_DIR)
            if not build_dir.is_dir() or not (build_dir / "all_reduce_perf_mpi").is_file():
                sys.exit(
                    f"No valid cached build: '{build_dir}' is missing or does not contain all_reduce_perf_mpi. "
                    "Run run_build_aws_multinode.py (or run_build_aws_multinode_test.py) and pass --build-dir to this script."
                )
            print(f"Using cached MPI build from {build_dir}")
            remote_build = f"{remote_nccl}/build"
            local_binary = build_dir / "all_reduce_perf_mpi"
            for ip in public_ips:
                ssh_run_cmd(ip, [f"mkdir -p {remote_build}"], str(private_key_path))
                scp_to_node(local_binary, f"{remote_build}/", ip, str(private_key_path))
        else:
            # Torch backend: ensure nodes have repo dir for consistency (no binary needed)
            for ip in public_ips:
                ssh_run_cmd(ip, [f"mkdir -p {remote_repo}"], str(private_key_path))

        node0_ip = public_ips[0]
        master_addr = private_ips[0]  # VPC private IP for rendezvous (like Modal's container_ips[0])
        master_port = 29500
        configs = [("auto", None)] if args.auto_only else get_benchmark_configs()

        if args.multinode_backend == "torch":
            # Modal-style: PyTorch env rendezvous (MASTER_ADDR, RANK, WORLD_SIZE). No Open MPI / PMIx.
            run_seq = args.mode in ("sequential", "both")
            run_ovl = args.mode in ("overlap", "both")
            print("Using PyTorch multinode backend (MASTER_ADDR/RANK/WORLD_SIZE, like Modal)...")
            print(f"Mode: {args.mode} (sequential={run_seq}, overlap={run_ovl})")
            print("Installing PyTorch on all nodes...")
            for ip in public_ips:
                ssh_run_cmd(ip, ["pip install --quiet torch"], str(private_key_path), check=False)
            # Write benchmark script to each node
            script_path = SCRIPT_DIR / ".torch_allreduce_bench.py"
            script_path.write_text(TORCH_ALLREDUCE_BENCH_SCRIPT.strip())
            for ip in public_ips:
                scp_to_node(script_path, "/tmp/", ip, str(private_key_path))
            script_path.unlink(missing_ok=True)
            remote_script = "/tmp/.torch_allreduce_bench.py"

            def run_one_config(tag: str, env_exports: str, overlap_mode: bool, out_suffix: str) -> None:
                overlap_env = "export OVERLAP_MODE=1; " if overlap_mode else "export OVERLAP_MODE=0; "
                node0_cmd_parts = []
                for i in range(gpus_per_node):
                    node0_cmd_parts.append(
                        f"({env_exports}{overlap_env}export PYTHONUNBUFFERED=1 RANK={i} WORLD_SIZE={total_gpus} MASTER_ADDR={master_addr} "
                        f"MASTER_PORT={master_port} LOCAL_RANK={i}; python3 -u {remote_script})"
                    )
                node0_cmd = " & ".join(node0_cmd_parts) + "; wait"

                def node_launch_cmd(node_idx: int) -> str:
                    base_rank = node_idx * gpus_per_node
                    parts = []
                    for i in range(gpus_per_node):
                        r = base_rank + i
                        parts.append(
                            f"({env_exports}{overlap_env}export PYTHONUNBUFFERED=1 RANK={r} WORLD_SIZE={total_gpus} MASTER_ADDR={master_addr} "
                            f"MASTER_PORT={master_port} LOCAL_RANK={i}; python3 -u {remote_script})"
                        )
                    return " & ".join(parts) + "; wait"

                def run_ssh(ip: str, cmd: str):
                    return ssh_run_cmd(ip, [cmd], str(private_key_path), check=False)

                with concurrent.futures.ThreadPoolExecutor(max_workers=num_nodes) as ex:
                    futures = [ex.submit(run_ssh, public_ips[0], node0_cmd)]
                    for ni in range(1, num_nodes):
                        futures.append(ex.submit(run_ssh, public_ips[ni], node_launch_cmd(ni)))
                    results = [f.result() for f in futures]
                stdout = results[0].stdout or ""
                stderr = results[0].stderr or ""
                if results[0].returncode != 0:
                    print(f"    (non-zero exit {results[0].returncode}; saving output anyway)", file=sys.stderr)
                if stderr:
                    stdout = stdout or "(no stdout)"
                    stdout = f"(stderr)\n{stderr}\n(stdout)\n{stdout}"
                if not stdout.strip():
                    stdout = f"(no output from rank 0; exit={results[0].returncode})\n(stderr)\n{stderr}"
                out_file = results_dir / f"results_{total_gpus}gpu_allreduce_{tag}{out_suffix}.txt"
                out_file.write_text(stdout)
                print(f"  [{tag}{out_suffix}] -> {out_file.name}")

            print(f"Running PyTorch NCCL all_reduce for {len(configs)} configs (mode={args.mode})...")
            for tag, env_additions in configs:
                if env_additions is None:
                    env_exports = "unset NCCL_ALGO NCCL_PROTO 2>/dev/null; "
                else:
                    env_exports = " ".join(f"export {k}={v}; " for k, v in env_additions.items())
                if run_seq:
                    run_one_config(tag, env_exports, overlap_mode=False, out_suffix="_sequential" if run_ovl else "")
                if run_ovl:
                    run_one_config(tag, env_exports, overlap_mode=True, out_suffix="_overlap")
            print(f"All results saved under: {results_dir}")
        else:
            # MPI backend: hostfile, Open MPI, mpirun
            ssh_run_cmd(node0_ip, [f"echo '{hostfile_content}' > /tmp/hostfile"], str(private_key_path))
            ssh_run_cmd(node0_ip, ["mkdir -p ~/.ssh && chmod 700 ~/.ssh"], str(private_key_path))
            remote_key = "/home/ubuntu/.ssh/cs244c_key"
            subprocess.run(
                ["scp", "-i", str(private_key_path), "-o", "StrictHostKeyChecking=no", str(private_key_path), f"ubuntu@{node0_ip}:{remote_key}"],
                check=True,
                capture_output=True,
            )
            ssh_run_cmd(node0_ip, [f"chmod 600 {remote_key}"], str(private_key_path))
            print("Ensuring OpenMPI (mpirun/orted) is installed on all nodes...")
            for ip in public_ips:
                ssh_run_cmd(ip, ["sudo apt-get update -qq && sudo apt-get install -y openmpi-bin"], str(private_key_path), check=False)
            mpirun_bin = "/usr/bin/mpirun"
            vpc_cidr = "172.31.0.0/16"
            mpirun_prefix = (
                "cd /home/ubuntu/repo/nccl-tests && "
                "export CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr && "
                "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH && "
                f"{mpirun_bin} -np {total_gpus} -N {gpus_per_node} --hostfile /tmp/hostfile "
                "-x PMIX_MCA_gds=hash "
                f'--mca plm_rsh_args "-i {remote_key} -o StrictHostKeyChecking=no" '
                f"--mca btl tcp,self --mca btl_tcp_if_include {vpc_cidr} "
                f"--mca oob tcp --mca oob_tcp_if_include {vpc_cidr} "
            )
            bench_args = "./build/all_reduce_perf_mpi -b 8 -e 128M -f 2 -g 1"
            print(f"Running NCCL all_reduce_perf for {len(configs)} configs{' (AUTO only)' if args.auto_only else ' (algo/proto + AUTO)'}...")
            for tag, env_additions in configs:
                if env_additions is None:
                    env_exports = "unset NCCL_ALGO NCCL_PROTO 2>/dev/null; "
                else:
                    env_exports = " ".join(f"export {k}={v}; " for k, v in env_additions.items())
                run_cmd = env_exports + mpirun_prefix + bench_args
                print(f"  [{tag}] ...")
                result = ssh_run_cmd(node0_ip, [run_cmd], str(private_key_path), check=False)
                stdout = result.stdout or ""
                if result.returncode != 0:
                    print(f"    (non-zero exit {result.returncode}; saving output anyway)", file=sys.stderr)
                    if result.stderr:
                        stdout = f"(stderr)\n{result.stderr}\n(stdout)\n{stdout}"
                out_file = results_dir / f"results_{total_gpus}gpu_allreduce_{tag}.txt"
                out_file.write_text(stdout)
                print(f"    -> {out_file.name}")
            print(f"All results saved under: {results_dir}")

    finally:
        if instance_ids and not args.no_terminate:
            # Collect volume IDs before terminating so we can delete any that don't auto-delete.
            volume_ids = get_volume_ids_for_instances(ec2, instance_ids)
            print("Terminating instances...")
            try:
                ec2.terminate_instances(InstanceIds=instance_ids)
                print("Done. Instances terminated to avoid further cost.")
                # Delete any EBS volumes that were left behind (e.g. DeleteOnTermination=False on AMI).
                if volume_ids:
                    time.sleep(5)
                    delete_orphaned_volumes(ec2, volume_ids)
            except Exception as e:
                if ClientError and isinstance(e, ClientError):
                    if e.response.get("Error", {}).get("Code") == "InvalidInstanceID.NotFound":
                        print("Instances already gone (InvalidInstanceID.NotFound); skipping terminate.")
                    else:
                        print(f"Terminate failed: {e}", file=sys.stderr)
                else:
                    print(f"Terminate failed: {e}", file=sys.stderr)
        elif instance_ids and args.no_terminate:
            print("Left instances running (--no-terminate). Terminate manually to avoid cost:")
            print("  aws ec2 terminate-instances --instance-ids", " ".join(instance_ids), "--region", args.region)


if __name__ == "__main__":
    main()
