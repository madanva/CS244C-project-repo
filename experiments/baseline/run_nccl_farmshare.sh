############################################################
# Usage:
#   bash run_nccl_farmshare.sh <num_gpus> <algorithms> [protocol]
#
#   <num_gpus>: Number of GPUs to use (e.g., 2, 4)
#   <algorithms>: Comma-separated list of NCCL algorithms (either ring,tree)
#                 or 'auto' for automatic algorithm selection by NCCL.
#   [protocol]: Optional. NCCL protocol to use (ll128, ll, simple)
#
# Examples:
#   bash run_nccl_farmshare.sh 2 auto
#   bash run_nccl_farmshare.sh 4 ring,tree ll128
#   bash run_nccl_farmshare.sh 4 ring ll
############################################################

#!/bin/bash

# Supported NCCL_ALGO values and their meaning:
#
# Ring (2.5+): Classic ring-based collective algorithm. Data is passed around GPUs in a ring, maximizing bandwidth for large messages.
# Tree (2.5+): Uses a tree structure for communication, reducing latency for small/medium messages.

# prior to running this script, you should've already queried the gpu resources you want
# this can be done with the command:
# srun --partition=gpu --gres=gpu:$NUM_GPUS --nodes=1 --pty bash
#
# This script also assumes you have some sort of micromamba environment setup for nccl
# The thing that is necessary for this is for $NCCL_HOME to be set which, isn't set unless 
# you source the environment since $NCCL_HOME should equal your $CONDA_PREFIX
#

set -e

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


# Usage: bash run_nccl_farmshare.sh <num_gpus> <algorithms> [protocol]
if [ $# -lt 2 ]; then
    echo "Usage: bash run_nccl_farmshare.sh <num_gpus> <algorithms> [protocol]"
    echo "  <num_gpus>: Number of GPUs to use (e.g., 2, 4)"
    echo "  <algorithms>: Comma-separated list (e.g., ring,tree,collnet,collnetchain,collnetdirect,nvls,nvlstree,pat) or 'auto' for automatic selection."
    echo "  [protocol]: Optional. NCCL protocol to use (ll128, ll, simple)"
    exit 1
fi
NUM_GPUS="$1"
ALGORITHMS="$2"
PROTO="$3"

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

# Capture directory where script was launched
BASE_DIR="$(pwd)"

IFS=',' read -ra ALGO_LIST <<< "$ALGORITHMS"

# Ensure output directory exists
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_TAG=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | tr -d ' ')

export NCCL_DEBUG=WARN
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"

if [ "$ALGORITHMS" = "auto" ]; then
    echo "Running NCCL all_reduce_perf with automatic algorithm selection on $NUM_GPUS GPUs..."
    unset NCCL_ALGO
    # Compose output folder name based on 'auto' and protocol
    if [ -n "$PROTO" ]; then
        RESULT_SUBDIR="auto_${PROTO}"
    else
        RESULT_SUBDIR="auto"
    fi
    RESULT_DIR="${BASE_DIR}/../${GPU_TAG}_${NUM_GPUS}gpu_results/${RESULT_SUBDIR}"
    mkdir -p "$RESULT_DIR"
    TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    if ../../nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g $NUM_GPUS | tee "$RESULT_DIR/${TIMESTAMP}.txt"; then
        echo "Results saved to $RESULT_DIR/${TIMESTAMP}.txt"
    else
        echo "Warning: NCCL test failed for automatic selection. See $RESULT_DIR/${TIMESTAMP}.txt for details."
    fi
else
    for ALGO in "${ALGO_LIST[@]}"; do
        if [ "$ALGO" = "auto" ]; then
            echo "Skipping 'auto' in algorithm list; use as a single argument for automatic selection."
            continue
        fi
        echo "Running NCCL all_reduce_perf with NCCL_ALGO: $ALGO on $NUM_GPUS GPUs..."

        export NCCL_ALGO="$ALGO"

        # Compose output folder name based on algorithm and protocol
        if [ -n "$PROTO" ]; then
            RESULT_SUBDIR="${ALGO}_${PROTO}"
        else
            RESULT_SUBDIR="${ALGO}"
        fi
        RESULT_DIR="${BASE_DIR}/../${GPU_TAG}_${NUM_GPUS}gpu_results/${RESULT_SUBDIR}"
        mkdir -p "$RESULT_DIR"
        TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
        if ../../nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g $NUM_GPUS | tee "$RESULT_DIR/${TIMESTAMP}.txt"; then
            echo "Results saved to $RESULT_DIR/${TIMESTAMP}.txt"
        else
            echo "Warning: NCCL test failed for algorithm '$ALGO'. See $RESULT_DIR/${TIMESTAMP}.txt for details."
        fi
    done
fi