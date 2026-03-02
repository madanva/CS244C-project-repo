
# Table of Contents

1. [Running NCCL All-Reduce Benchmark with Modal](#running-nccl-all-reduce-benchmark-with-modal)
2. [Running NCCL All-Reduce Benchmark on FarmShare](#running-nccl-all-reduce-benchmark-on-farmshare)
3. [Plotting Results](#plotting-results)
4. [Summary](#summary)

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
