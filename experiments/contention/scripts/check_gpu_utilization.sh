#!/bin/bash
# Monitor GPU utilization while running gpu_stress_benchmark

set -e


# Remove old binary if it exists
if [ -f gpu_stress_benchmark ]; then
    rm gpu_stress_benchmark
fi

# Always recompile the binary
echo "Compiling gpu_stress_benchmark.cu..."
nvcc -lcublas gpu_stress_benchmark.cu -o gpu_stress_benchmark


UTILIZATION="high"
if [ $# -ge 1 ]; then
    UTILIZATION="$1"
fi

echo "Starting gpu_stress_benchmark with $UTILIZATION utilization..."
./gpu_stress_benchmark $UTILIZATION &
BENCH_PID=$!



# Ensure output directory exists
OUTDIR="gpu_utilization_logs"
mkdir -p "$OUTDIR"


# Monitor GPU utilization for 30 seconds and log to CSV
echo "Monitoring GPU utilization for 30 seconds..."
CSV_FILE="$OUTDIR/gpu_utilization_log_${UTILIZATION}.csv"
echo "timestamp,gpu_utilization,memory_used" > "$CSV_FILE"
for i in {1..30}; do
    TIMESTAMP=$(date +%s)
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | \
        awk -v ts="$TIMESTAMP" -F',' '{print ts "," $1 "," $2}' >> "$CSV_FILE"
    sleep 1
done

# Kill the benchmark
echo "Stopping gpu_stress_benchmark..."
kill $BENCH_PID || true

echo "Done."
