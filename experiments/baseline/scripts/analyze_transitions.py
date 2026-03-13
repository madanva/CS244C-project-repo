#!/usr/bin/env python3
"""
Analyze NCCL all-reduce performance transitions from benchmark output.
Identifies where algorithm/protocol switches likely occur based on bandwidth changes.
"""

import re
import sys

def parse_benchmark_output(filename):
    """Parse nccl-tests output and extract size vs bandwidth data."""
    data = []
    current_size = None
    
    with open(filename, 'r') as f:
        for line in f:
            # Match size line: "     1048576        262144     float"
            size_match = re.search(r'^\s+(\d+)\s+\d+\s+float\s*$', line)
            if size_match:
                current_size = int(size_match.group(1))
                continue
            
            # Match data line: "    sum      -1     4.56  229.85    0.00  ..."
            if current_size is not None:
                data_match = re.search(r'^\s+sum\s+\-1\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
                if data_match:
                    time_us = float(data_match.group(1))
                    algbw_gbps = float(data_match.group(2))
                    data.append((current_size, time_us, algbw_gbps))
                    current_size = None  # Reset for next size
    
    return data

def find_transitions(data, threshold_pct=20):
    """Find significant bandwidth transitions (likely algo/proto switches)."""
    transitions = []
    for i in range(1, len(data)):
        prev_size, prev_time, prev_bw = data[i-1]
        curr_size, curr_time, curr_bw = data[i]
        
        if prev_bw > 0:
            pct_change = abs((curr_bw - prev_bw) / prev_bw) * 100
            if pct_change > threshold_pct:
                transitions.append({
                    'from_size': prev_size,
                    'to_size': curr_size,
                    'from_bw': prev_bw,
                    'to_bw': curr_bw,
                    'change_pct': pct_change,
                    'direction': 'up' if curr_bw > prev_bw else 'down'
                })
    return transitions

def format_size(bytes_val):
    """Format bytes to human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val}B"
    elif bytes_val < 1024*1024:
        return f"{bytes_val/1024:.1f}KB"
    elif bytes_val < 1024*1024*1024:
        return f"{bytes_val/(1024*1024):.1f}MB"
    else:
        return f"{bytes_val/(1024*1024*1024):.2f}GB"

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_transitions.py <benchmark_output_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    data = parse_benchmark_output(filename)
    
    if not data:
        print("No data found in file")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"NCCL All-Reduce Performance Analysis")
    print(f"{'='*80}\n")
    
    print(f"Total data points: {len(data)}\n")
    
    # Show full data
    print(f"{'Size':<12} {'Time (us)':<12} {'Bandwidth (GB/s)':<20}")
    print(f"{'-'*50}")
    for size, time_us, bw in data:
        print(f"{format_size(size):<12} {time_us:<12.2f} {bw:<20.2f}")
    
    # Find transitions
    print(f"\n{'='*80}")
    print(f"Significant Bandwidth Transitions (>20% change)")
    print(f"{'='*80}\n")
    
    transitions = find_transitions(data, threshold_pct=20)
    
    if not transitions:
        print("No significant transitions found")
    else:
        for t in transitions:
            print(f"Transition at {format_size(t['from_size'])} → {format_size(t['to_size'])}")
            print(f"  Bandwidth: {t['from_bw']:.2f} → {t['to_bw']:.2f} GB/s ({t['direction']}, {t['change_pct']:.1f}% change)")
            print()
    
    # Identify performance regions
    print(f"{'='*80}")
    print(f"Performance Regions")
    print(f"{'='*80}\n")
    
    if data:
        # Small messages (< 1MB)
        small_msgs = [(s, bw) for s, t, bw in data if s < 1024*1024]
        if small_msgs:
            avg_bw = sum(bw for s, bw in small_msgs) / len(small_msgs)
            max_bw = max(bw for s, bw in small_msgs)
            print(f"Small messages (< 1MB): avg {avg_bw:.2f} GB/s, max {max_bw:.2f} GB/s")
        
        # Medium messages (1MB - 100MB)
        med_msgs = [(s, bw) for s, t, bw in data if 1024*1024 <= s < 100*1024*1024]
        if med_msgs:
            avg_bw = sum(bw for s, bw in med_msgs) / len(med_msgs)
            max_bw = max(bw for s, bw in med_msgs)
            print(f"Medium messages (1MB - 100MB): avg {avg_bw:.2f} GB/s, max {max_bw:.2f} GB/s")
        
        # Large messages (>= 100MB)
        large_msgs = [(s, bw) for s, t, bw in data if s >= 100*1024*1024]
        if large_msgs:
            avg_bw = sum(bw for s, bw in large_msgs) / len(large_msgs)
            max_bw = max(bw for s, bw in large_msgs)
            print(f"Large messages (>= 100MB): avg {avg_bw:.2f} GB/s, max {max_bw:.2f} GB/s")
        
        print(f"\nPeak bandwidth: {max(bw for s, t, bw in data):.2f} GB/s at {format_size(max(data, key=lambda x: x[2])[0])}")

if __name__ == "__main__":
    main()
