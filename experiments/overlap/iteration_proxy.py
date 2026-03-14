"""
Phase 3 training-step proxy: compute phase → all-reduce phase.
Measures end-to-end iteration time (not just communication bandwidth).
Run with: torchrun --nproc_per_node=8 iteration_proxy.py [--iters N] [--size S]
Output: iteration times (ms) to stdout and to a file (rank 0 only).
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 iteration proxy")
    p.add_argument("--iters", type=int, default=50, help="Number of timed iterations")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    p.add_argument("--size", type=int, default=2**20, help="All-reduce tensor size (elements, float32)")
    p.add_argument("--compute-mul", type=int, default=4096, help="Compute matmul size (NxN)")
    p.add_argument(
        "--overlap",
        action="store_true",
        help="Enable compute–communication overlap using separate CUDA streams",
    )
    p.add_argument("--out", type=str, default="", help="Output file for iteration times (rank 0)")
    return p.parse_args()


def compute_phase(device: torch.device, n: int, dtype=torch.float32):
    """Simulate per-iteration compute: matmul on GPU (single stream version)."""
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    return c


def allreduce_phase(tensor: torch.Tensor):
    """All-reduce the tensor across all ranks (single stream version)."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(device)

    # Per-iteration buffer for all-reduce (same size on all ranks)
    elem = args.size
    grad = torch.randn(elem, device=device, dtype=torch.float32) / world_size

    reward_file = os.environ.get("NCCL_TUNER_REWARD_FILE", "")

    if not args.overlap:
        # Warmup without overlap
        for _ in range(args.warmup):
            compute_phase(device, args.compute_mul)
            allreduce_phase(grad.clone())

        # Timed iterations, single stream (no overlap)
        times_ms = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            compute_phase(device, args.compute_mul)
            allreduce_phase(grad.clone())
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            iter_ms = (t1 - t0) * 1000.0
            times_ms.append(iter_ms)

            if reward_file and rank == 0:
                try:
                    n_bytes = elem * 4  # float32 elements
                    with open(reward_file, "a") as f:
                        f.write(f"allreduce,{n_bytes},{1},{world_size},{iter_ms:.3f}\n")
                except OSError:
                    pass
    else:
        # Overlap mode: use separate CUDA streams for compute and all-reduce
        compute_stream = torch.cuda.Stream(device=device)
        comm_stream = torch.cuda.Stream(device=device)

        # Warmup with overlap pattern (no timing)
        for _ in range(args.warmup):
            with torch.cuda.stream(compute_stream):
                a = torch.randn(args.compute_mul, args.compute_mul, device=device, dtype=torch.float32)
                b = torch.randn(args.compute_mul, args.compute_mul, device=device, dtype=torch.float32)
                _ = torch.matmul(a, b)
            with torch.cuda.stream(comm_stream):
                tmp = grad.clone()
                dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

        # Timed iterations with overlap: schedule compute and all-reduce
        times_ms = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.cuda.stream(compute_stream):
                a = torch.randn(args.compute_mul, args.compute_mul, device=device, dtype=torch.float32)
                b = torch.randn(args.compute_mul, args.compute_mul, device=device, dtype=torch.float32)
                _ = torch.matmul(a, b)

            with torch.cuda.stream(comm_stream):
                tmp = grad.clone()
                dist.all_reduce(tmp, op=dist.ReduceOp.SUM)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            iter_ms = (t1 - t0) * 1000.0
            times_ms.append(iter_ms)

            if reward_file and rank == 0:
                try:
                    n_bytes = elem * 4  # float32 elements
                    with open(reward_file, "a") as f:
                        f.write(f"allreduce,{n_bytes},{1},{world_size},{iter_ms:.3f}\n")
                except OSError:
                    pass

    if rank == 0:
        out_lines = [f"{t:.3f}" for t in times_ms]
        summary = (
            f"config={os.environ.get('NCCL_PROTO', 'AUTO')} "
            f"mean_ms={sum(times_ms)/len(times_ms):.2f} "
            f"p95_ms={sorted(times_ms)[int(len(times_ms)*0.95)]:.2f} "
            f"iters={args.iters}"
        )
        print(summary, flush=True)
        for line in out_lines:
            print(line, flush=True)
        if args.out:
            with open(args.out, "w") as f:
                f.write("\n".join(out_lines) + "\n")
            print(f"Wrote {args.out}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
