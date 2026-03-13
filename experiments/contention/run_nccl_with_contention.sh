
# Source user bashrc to ensure environment variables are set

source ~/.bashrc
#!/bin/bash
# Run GPU stress benchmark and NCCL all_reduce_perf for contention analysis

set -e

# Sanity checks for NCCL environment and binaries
export NCCL_HOME="$CONDA_PREFIX"

# Check NCCL_HOME
if [ -z "$NCCL_HOME" ]; then
    echo "Error: NCCL_HOME is not set. Please activate your micromamba/conda environment for NCCL."
    exit 1
fi
if [ ! -d "$NCCL_HOME" ]; then
    echo "Error: NCCL_HOME directory '$NCCL_HOME' does not exist."
    exit 1
fi
if [ ! -d "$NCCL_HOME/lib" ]; then
    echo "Error: NCCL_HOME/lib directory '$NCCL_HOME/lib' does not exist."
    exit 1
fi

# Check all_reduce_perf binary
ALLREDUCE_BIN="../../nccl-tests/build/all_reduce_perf"
if [ ! -x "$ALLREDUCE_BIN" ]; then
    echo "Error: NCCL all_reduce_perf binary '$ALLREDUCE_BIN' does not exist or is not executable."
    exit 1
fi

# Set LD_LIBRARY_PATH to include NCCL_HOME/lib for dynamic linker
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"

# Check for libnccl.so.2 presence
if [ ! -f "$NCCL_HOME/lib/libnccl.so.2" ]; then
    echo "Warning: libnccl.so.2 not found in $NCCL_HOME/lib. NCCL may not be installed correctly or the library path is wrong."
fi

# Compile the CUDA benchmark if not already present
if [ ! -f gpu_stress_benchmark ]; then
    echo "Compiling gpu_stress_benchmark.cu..."
    nvcc -lcublas gpu_stress_benchmark.cu -o gpu_stress_benchmark
fi

# Usage: bash run_nccl_with_contention.sh <num_gpus> <algorithms> [protocol]
if [ $# -lt 2 ]; then
    echo "Usage: bash run_nccl_with_contention.sh <num_gpus> <algorithms> [protocol]"
    echo "  <num_gpus>: Number of GPUs to use (e.g., 2, 4)"
    echo "  <algorithms>: Comma-separated list (e.g., ring,tree,auto)"
    echo "  [protocol]: Optional. NCCL protocol to use (ll128, ll, simple)"
    exit 1
fi
NUM_GPUS="$1"
ALGORITHMS="$2"
PROTO="$3"

CONTENTION_LEVELS=(low medium high)

IFS=',' read -ra ALGO_LIST <<< "$ALGORITHMS"

# Get GPU name and tag
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_TAG=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | tr -d ' ')

# Set NCCL_PROTO if provided
if [ -n "$PROTO" ]; then
    case "$PROTO" in
        ll128|ll|simple)
            export NCCL_PROTO="$PROTO"
            echo "Using NCCL_PROTO=$PROTO"
            ;;
        *)
            echo "Invalid protocol: $PROTO. Valid options are ll128, ll, simple."
            exit 1
            ;;
    esac
fi

for CONTENTION in "${CONTENTION_LEVELS[@]}"; do
    for ALGO in "${ALGO_LIST[@]}"; do
        if [ "$ALGO" = "auto" ]; then
            export NCCL_ALGO=""
            RESULT_SUBDIR="auto"
        else
            export NCCL_ALGO="$ALGO"
            RESULT_SUBDIR="$ALGO"
        fi
        if [ -n "$PROTO" ]; then
            RESULT_SUBDIR="${RESULT_SUBDIR}_${PROTO}"
        fi
        OUTDIR="contention_results/${RESULT_SUBDIR}/${CONTENTION}"
        mkdir -p "$OUTDIR"

        echo "Starting gpu_stress_benchmark with $CONTENTION contention on $NUM_GPUS GPUs..."
        ./gpu_stress_benchmark $CONTENTION $NUM_GPUS &
        BENCH_PID=$!
        sleep 2

        echo "Running NCCL all_reduce_perf with ALGO=$ALGO, PROTO=$PROTO, contention=$CONTENTION on $NUM_GPUS GPUs..."
        # Compose output file name based on algo and protocol
        FILE_TAG="$ALGO"
        if [ -n "$PROTO" ]; then
            FILE_TAG="${FILE_TAG}_${PROTO}"
        fi
        OUTFILE="$OUTDIR/results_${GPU_TAG}-${NUM_GPUS}gpu_allreduce_${FILE_TAG}_contended_${CONTENTION}.txt"
        $ALLREDUCE_BIN -b 8 -e 128M -f 2 -g $NUM_GPUS | tee "$OUTFILE"

        echo "Killing GPU stress benchmark..."
        kill $BENCH_PID || true

        echo "Contention experiment complete. Results saved to $OUTFILE."
    done
done