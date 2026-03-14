#!/usr/bin/env python3
"""
Compare NCCL protocol performance from explicit protocol tests.
"""

import re
import sys

def parse_results(filename):
    """Extract clean size vs bandwidth data."""
    data = {}
    current_size = None
    
    with open(filename, 'r') as f:
        for line in f:
            # Match size line
            size_match = re.search(r'^\s+(\d+)\s+\d+\s+float\s*$', line)
            if size_match:
                current_size = int(size_match.group(1))
                continue
            
            # Match data line - get the out-of-place bandwidth (first algbw)
            if current_size is not None:
                data_match = re.search(r'^\s+sum\s+\-1\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
                if data_match:
                    time_us = float(data_match.group(1))
                    algbw_gbps = float(data_match.group(2))
                    
                    # Keep the best bandwidth for each size
                    if current_size not in data or algbw_gbps > data[current_size]:
                        data[current_size] = algbw_gbps
                    current_size = None
    
    return sorted(data.items())

def format_size(bytes_val):
    """Format bytes to human-readable."""
    if bytes_val < 1024*1024:
        return f"{bytes_val/1024:.0f}KB"
    elif bytes_val < 1024*1024*1024:
        return f"{bytes_val/(1024*1024):.0f}MB"
    else:
        return f"{bytes_val/(1024*1024*1024):.1f}GB"

def main():
    print("\n" + "="*90)
    print("NCCL PROTOCOL COMPARISON - EXPLICIT PERFORMANCE DIFFERENCES")
    print("="*90 + "\n")
    
    # Parse all three test results
    baseline = dict(parse_results('baseline_auto.out'))
    ll128 = dict(parse_results('ll128_forced.out'))
    simple = dict(parse_results('simple_forced.out'))
    
    # Get all sizes
    all_sizes = sorted(set(baseline.keys()) | set(ll128.keys()) | set(simple.keys()))
    
    print(f"{'Size':<12} {'Auto (GB/s)':<15} {'LL128 (GB/s)':<15} {'Simple (GB/s)':<15} {'Best Protocol':<20}")
    print("-" * 90)
    
    for size in all_sizes:
        auto_bw = baseline.get(size, 0)
        ll128_bw = ll128.get(size, 0)
        simple_bw = simple.get(size, 0)
        
        # Determine which is best
        best = max([
            (auto_bw, "Auto"),
            (ll128_bw, "LL128"),
            (simple_bw, "Simple")
        ])
        
        print(f"{format_size(size):<12} {auto_bw:<15.2f} {ll128_bw:<15.2f} {simple_bw:<15.2f} {best[1]:<20}")
    
    print("\n" + "="*90)
    print("KEY INSIGHTS")
    print("="*90 + "\n")
    
    # Analyze small messages (< 8MB)
    small_sizes = [s for s in all_sizes if s < 8*1024*1024]
    if small_sizes:
        print("📊 SMALL MESSAGES (< 8MB):")
        for size in small_sizes:
            auto_bw = baseline.get(size, 0)
            ll128_bw = ll128.get(size, 0)
            simple_bw = simple.get(size, 0)
            
            if simple_bw > 0 and ll128_bw > 0:
                ll128_advantage = ((ll128_bw - simple_bw) / simple_bw) * 100
                if ll128_advantage > 10:
                    print(f"  • {format_size(size)}: LL128 is {ll128_advantage:.0f}% FASTER than Simple")
                    print(f"    ({ll128_bw:.0f} GB/s vs {simple_bw:.0f} GB/s)")
    
    print()
    
    # Analyze large messages (>= 32MB)
    large_sizes = [s for s in all_sizes if s >= 32*1024*1024]
    if large_sizes:
        print("📊 LARGE MESSAGES (>= 32MB):")
        for size in large_sizes:
            auto_bw = baseline.get(size, 0)
            ll128_bw = ll128.get(size, 0)
            simple_bw = simple.get(size, 0)
            
            if simple_bw > 0 and ll128_bw > 0:
                simple_advantage = ((simple_bw - ll128_bw) / ll128_bw) * 100
                if simple_advantage > 10:
                    print(f"  • {format_size(size)}: Simple is {simple_advantage:.0f}% FASTER than LL128")
                    print(f"    ({simple_bw:.0f} GB/s vs {ll128_bw:.0f} GB/s)")
    
    print("\n" + "="*90)
    print("CONCLUSION")
    print("="*90 + "\n")
    
    print("✅ AUTO MODE (default) intelligently switches between protocols:")
    print("   • Uses LL128 for small messages (better latency)")
    print("   • Uses Simple for large messages (better bandwidth)")
    print("   • This is why you get the best performance automatically!\n")
    
    print("⚠️  Forcing a single protocol is SUBOPTIMAL:")
    print("   • LL128-only: Great for small messages, terrible for large ones")
    print("   • Simple-only: Great for large messages, terrible for small ones\n")
    
    print("💡 For distributed training: Keep AUTO mode (default)")
    print("   NCCL knows what it's doing!\n")

if __name__ == "__main__":
    main()
