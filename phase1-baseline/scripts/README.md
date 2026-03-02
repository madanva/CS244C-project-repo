
# Table of Contents

1. [Running NCCL All-Reduce Benchmark with Modal](#running-nccl-all-reduce-benchmark-with-modal)
2. [Running NCCL All-Reduce Benchmark on AWS](#running-nccl-all-reduce-benchmark-on-aws)
3. [Running NCCL All-Reduce Benchmark on FarmShare](#running-nccl-all-reduce-benchmark-on-farmshare)
4. [Plotting Results](#plotting-results)
5. [Summary](#summary)

# Running NCCL All-Reduce Benchmark with Modal

The `run_modal.py` script lets you run the NCCL all-reduce benchmark on any supported GPU architecture and GPU count using Modal Labs cloud infrastructure.

## Usage

From this directory, run:

```bash
modal run run_modal.py --arch <ARCH> --gpus <NUM_GPUS>
```

- `<ARCH>`: GPU architecture (e.g., `A100`, `L40S`)
- `<NUM_GPUS>`: Number of GPUs to use (e.g., `2`, `4`, `8`)

Example:

```bash
modal run run_modal.py --arch A100 --gpus 8
```

## What It Does

- Builds the `nccl-tests` benchmark suite inside a container with CUDA and NCCL.
- Runs the `all_reduce_perf` binary with the specified GPU architecture and count.
- Captures the benchmark output and saves it to a file named `results_<arch>_<num_gpus>gpu_allreduce.txt` in the `results/` directory.
- Prints the output for easy access and analysis.

# Running NCCL All-Reduce Benchmark on AWS

`run_benchmark_aws_multinode.py` runs the NCCL all-reduce benchmark on N EC2 GPU nodes (launched by the script or via `--existing-ips`), then terminates any instances it launched.

## AWS setup (one-time)

1. **Credentials:** `aws configure` (Access Key ID, Secret Access Key, region e.g. `us-east-1`), or set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_DEFAULT_REGION`.
2. **EC2 key pair:** AWS Console → EC2 → Key Pairs → Create key pair (.pem). Save the .pem (e.g. `~/.ssh/my-key.pem`) and run `chmod 600 ~/.ssh/my-key.pem`.
3. **Env:** `mamba env create -f phase1-baseline/scripts/environment.yml && mamba activate aws-multinode` (for boto3). From repo root: `git submodule update --init --recursive`.

Verify: `aws sts get-caller-identity`.

## Usage

From `phase1-baseline/scripts`:

```bash
python run_benchmark_aws_multinode.py --key-name YOUR_KEY --private-key ~/.ssh/YOUR_KEY.pem [options]
```

**All parameters**

| Argument | Default | Description |
|----------|---------|-------------|
| `--region` | `us-east-1` | AWS region |
| `--ami` | (lookup) | AMI ID; omit to use latest Deep Learning GPU Ubuntu 22.04 |
| `--instance-type` | `g5.xlarge` | Instance type (e.g. `g5.xlarge` 1 GPU, `p4d.24xlarge` 8× A100) |
| `--num-nodes` | `2` | Number of nodes. With `--existing-ips`: target total; extras launched if fewer IPs given |
| `--existing-ips` | — | Comma-separated public IPs to reuse. If fewer than `--num-nodes`, script launches the rest |
| `--private-ips` | (discover) | With `--existing-ips`: comma-separated private IPs (else discovered via SSH) |
| `--gpus-per-node` | (discover) | With `--existing-ips`: GPUs per node (else from nvidia-smi on first node) |
| `--key-name` | — | EC2 key pair name (required when launching; required with `--existing-ips` when launching extras) |
| `--private-key` | **required** | Path to .pem for SSH/SCP |
| `--vpc-id` | default VPC | VPC for security group |
| `--placement-group` | — | Optional placement group |
| `--no-terminate` | off | Leave launched instances running (terminate manually to avoid cost) |
| `--results-dir` | `results/aws-multinode` | Where to save result .txt files |
| `--build-dir` | `build_cache/nccl-tests-mpi/build` | Path to cached nccl-tests build (must contain `all_reduce_perf_mpi`) |
| `--auto-only` | off | Run only AUTO benchmark |
| `--multinode-backend` | `mpi` | `mpi` (Open MPI) or `torch` (PyTorch rendezvous) |
| `--mode` | `sequential` | Torch only: `sequential`, `overlap`, or `both` |
| `--skip-unsupported-algos` | off | Skip CollNet/NVLS/PAT (use on 2-node 1-GPU e.g. g5.xlarge to avoid failures) |

**Examples**

Launch 3 nodes, torch backend, both modes, skip unsupported algos, leave up:

```bash
python run_benchmark_aws_multinode.py --key-name my-key --private-key ~/.ssh/my-key.pem --num-nodes 3 --multinode-backend torch --mode both --skip-unsupported-algos --no-terminate
```

Use 3 existing IPs (no launch/terminate):

```bash
python run_benchmark_aws_multinode.py --existing-ips "IP1,IP2,IP3" --private-key ~/.ssh/my-key.pem --multinode-backend torch --mode both
```

Use 3 existing IPs but run on 5 nodes (script launches 2 more):

```bash
python run_benchmark_aws_multinode.py --existing-ips "IP1,IP2,IP3" --private-ips "P1,P2,P3" --num-nodes 5 --key-name my-key --private-key ~/.ssh/my-key.pem --multinode-backend torch
```

Results: `results_<N>gpu_allreduce_<config>_sequential.txt` and `_overlap.txt` (when using torch with overlap/both).

# Running NCCL All-Reduce Benchmark on FarmShare

Before running the NCCL All-Reduce benchmarks on FarmShare, you must:

- Install micromamba 
- Create and activate an environment, the `nccl-env` environment: This environment should contain CUDA, NCCL, and any other dependencies required to build and run `nccl-tests`.

Example commands:
```bash
# Create the nccl-env environment
micromamba create -y -n nccl-env cuda nccl make gcc
# (Add any other dependencies as needed)

# Activate the environment (for interactive use)
micromamba activate nccl-env
```

The `run_nccl_farmshare.sh` script will automatically activate the environment for you when running benchmarks. **This script EXPECTS the name `nccl-env`.**

## Usage

From this directory, run:

```bash
./run_nccl_farmshare.sh <NUM_GPUS> <ALGORITHMS> [PROTOCOL]
```

- `<NUM_GPUS>`: Number of GPUs to use (e.g., `2`, `4`)
- `<ALGORITHMS>`: Comma-separated list of NCCL algorithms (can be `ring,tree`) or `auto` for automatic selection. If you do `ring,tree`, it will run both algorithms and save results separately.
- `[PROTOCOL]`: Optional. NCCL protocol to use (`ll128`, `ll`, `simple`). Select only one

Examples:

```bash
./run_nccl_farmshare.sh 2 auto
./run_nccl_farmshare.sh 4 ring,tree ll128
./run_nccl_farmshare.sh 4 ring ll
```

> [!NOTE] farmshare only allows up to 4 GPUs

## What It Does

- Requests the specified number of L40S GPUs on FarmShare interactively using `srun`.
- Loads the CUDA and NCCL environments using micromamba.
- Builds the `nccl-tests` benchmark suite if needed.
- Runs the `all_reduce_perf` binary with the selected GPU count.
- Captures the benchmark output and saves it to a timestamped file in a results folder named after the GPU type and count (e.g., `l40s_2gpu_results/2026-02-20_15-30-00.txt`).
- Prints the output location for easy access and analysis.

## Notes

- Make sure your micromamba environment and NCCL libraries are set up as described in the script.
- Results are organized by GPU type and count for easy comparison and plotting.

## Troubleshooting

*If it's complaining about formatting or "srun: error: Invalid Trackable RESource (TRES) specification"*

Ensure that you have passed the number of gpus as an argument when running the script, e.g., `./run_nccl_farmshare.sh 2`. The script expects a single argument specifying the number of GPUs to request from FarmShare.

*If you are getting issues of where to run the script `run_nccl_farmshare.sh`*

It expects to be run within the `scripts/` directory of the `phase1-baseline/` directory. Make sure you are in the correct directory before executing the script.


# Plotting Results

Use the plotting scripts to visualize bandwidth and latency:

**Bandwidth plot:**
```bash
python plot_nccl_bw.py ../results/results_a100_8gpu_allreduce.txt --arch "A100 8-GPU"
```
Output: Plots are written to `bandwidth_graphs/` in this directory.

**Latency plot:**
```bash
python plot_nccl_latency.py ../results/results_a100_8gpu_allreduce.txt --arch "A100 8-GPU"
```
Output: Plots are written to `latency_graphs/` in this directory.

**Multi-line plotting:**
To compare results from multiple folders in a single plot:
```bash
python plot_nccl_multi.py ../folder1 ../folder2 --output_dir multi_graphs --arch "l40s-2gpu"
```
Output: Plots are written to `multi_graphs/` in this directory, with filenames and plot titles reflecting the input folders. It will generate a multi-line plot for latency and bandwidth.

Note that in the folders you designate, it will grab the .txt file corresponding to the nccl-test output.

