"""
Generate the workload-aware tuner policy from all experimental data.

Reads:
  - results/overlap_experiment/overlap_experiment_results.json  (Exp 2)
  - results/channel_experiment/channel_experiment_results.json   (Exp 3)

Outputs:
  - results/tuner_policy.json           — human-readable policy table
  - results/tuner_policy_table.h        — C header for the tuner plugin
  - results/tuner_policy_analysis.txt   — detailed analysis of policy derivation
"""

import json
from pathlib import Path

RESULTS_ROOT = Path(__file__).parent / "results"
OVERLAP_FILE = RESULTS_ROOT / "overlap_experiment" / "overlap_experiment_results.json"
CHANNEL_FILE = RESULTS_ROOT / "channel_experiment" / "channel_experiment_results.json"

# NCCL constants (must match tuner.h)
ALGO_MAP = {
    "tree": 0,   # NCCL_ALGO_TREE
    "ring": 1,   # NCCL_ALGO_RING
}
PROTO_MAP = {
    "simple": 2,  # NCCL_PROTO_SIMPLE
    "ll128": 1,   # NCCL_PROTO_LL128
    "ll": 0,      # NCCL_PROTO_LL
}

# Config name → (algo_id, proto_id)
CONFIG_TO_IDS = {
    "auto":        (-1, -1),  # let NCCL decide
    "tree_simple": (0, 2),
    "tree_ll128":  (0, 1),
    "ring_simple": (1, 2),
    "ring_ll128":  (1, 1),
}

# Size label → bytes
SIZE_BYTES = {
    "32KB":  32 * 1024,
    "64KB":  64 * 1024,
    "256KB": 256 * 1024,
    "1MB":   1024 * 1024,
    "2MB":   2 * 1024 * 1024,
    "4MB":   4 * 1024 * 1024,
    "16MB":  16 * 1024 * 1024,
    "64MB":  64 * 1024 * 1024,
    "256MB": 256 * 1024 * 1024,
}


def find_best(data_dict):
    """Find best config name and its time from a {config: time_ms} dict."""
    if not data_dict:
        return "auto", 999.0
    best_cfg = min(data_dict, key=data_dict.get)
    return best_cfg, data_dict[best_cfg]


def find_auto_gap(data_dict):
    """Compute (auto_time - best_time) / auto_time * 100."""
    auto_t = data_dict.get("auto", 0)
    best_cfg, best_t = find_best(data_dict)
    if auto_t <= 0:
        return 0.0
    return (auto_t - best_t) / auto_t * 100


def find_best_cta(cta_sweep_for_size):
    """Find best (cta, config, time) across all CTA counts for a given size."""
    best_cta, best_cfg, best_t = "8", "auto", 999.0
    for cta, data in cta_sweep_for_size.items():
        for cfg, t in data.items():
            if t < best_t:
                best_t, best_cta, best_cfg = t, cta, cfg
    return best_cta, best_cfg, best_t


def main():
    print("=" * 80)
    print("  WORKLOAD-AWARE TUNER POLICY GENERATOR")
    print("=" * 80)

    # Load experiment data
    overlap_data = {}
    if OVERLAP_FILE.exists():
        overlap_data = json.loads(OVERLAP_FILE.read_text())
        print(f"\nLoaded overlap experiment: {OVERLAP_FILE}")
    else:
        print(f"\nWARNING: {OVERLAP_FILE} not found")

    channel_data = {}
    if CHANNEL_FILE.exists():
        channel_data = json.loads(CHANNEL_FILE.read_text())
        print(f"Loaded channel experiment: {CHANNEL_FILE}")
    else:
        print(f"WARNING: {CHANNEL_FILE} not found")

    analysis_lines = []

    # ================================================================
    # PHASE 1: Derive optimal (algo, proto) per (size, mode)
    # ================================================================
    print(f"\n{'='*80}")
    print("  PHASE 1: Optimal (algo, proto) per (size, overlap_mode)")
    print(f"{'='*80}")

    policy = {"sequential": {}, "overlap": {}}

    # From overlap experiment
    for mode in ["sequential", "overlap"]:
        mode_data = overlap_data.get(mode, {})
        print(f"\n  {mode.upper()}:")
        print(f"  {'Size':<10s} {'Best Config':<15s} {'Time (ms)':>10s} {'AUTO':>8s} {'Gap%':>6s}")
        print(f"  {'-'*55}")

        for size_label in sorted(mode_data.keys(), key=lambda s: SIZE_BYTES.get(s, 0)):
            data = mode_data[size_label]
            best_cfg, best_t = find_best(data)
            auto_t = data.get("auto", 0)
            gap = find_auto_gap(data)

            policy[mode][size_label] = {
                "best_config": best_cfg,
                "best_time_ms": best_t,
                "auto_time_ms": auto_t,
                "auto_gap_pct": round(gap, 2),
                "algo": CONFIG_TO_IDS[best_cfg][0],
                "proto": CONFIG_TO_IDS[best_cfg][1],
            }

            print(f"  {size_label:<10s} {best_cfg:<15s} {best_t:>10.3f} {auto_t:>8.3f} {gap:>+5.1f}%")

            analysis_lines.append(
                f"{mode},{size_label},{best_cfg},{best_t:.3f},{auto_t:.3f},{gap:+.1f}%"
            )

    # ================================================================
    # PHASE 2: Derive optimal CTA count per (size, mode)
    # ================================================================
    print(f"\n{'='*80}")
    print("  PHASE 2: Optimal CTA count per (size, overlap_mode)")
    print(f"{'='*80}")

    cta_policy = {"overlap": {}, "sequential": {}}

    for mode_key, data_key in [("overlap", "overlap_cta_sweep"),
                                ("sequential", "sequential_cta_sweep")]:
        sweep = channel_data.get(data_key, {})
        if not sweep:
            print(f"\n  {mode_key.upper()}: No data")
            continue

        print(f"\n  {mode_key.upper()}:")
        print(f"  {'Size':<10s} {'Opt CTAs':>10s} {'Best Config':<15s} {'Time (ms)':>10s} "
              f"{'Default(8)':>10s} {'Improvement':>12s}")
        print(f"  {'-'*75}")

        for size_label in sorted(sweep.keys(), key=lambda s: SIZE_BYTES.get(s, 0)):
            best_cta, best_cfg, best_t = find_best_cta(sweep[size_label])
            default_8 = sweep[size_label].get("8", {})
            default_auto = default_8.get("auto", 0)

            improvement = (default_auto - best_t) / default_auto * 100 if default_auto > 0 else 0

            cta_policy[mode_key][size_label] = {
                "optimal_ctas": int(best_cta),
                "best_config": best_cfg,
                "best_time_ms": best_t,
                "default_8_auto_ms": default_auto,
                "improvement_pct": round(improvement, 2),
            }

            print(f"  {size_label:<10s} {best_cta:>10s} {best_cfg:<15s} {best_t:>10.3f} "
                  f"{default_auto:>10.3f} {improvement:>+11.1f}%")

    # ================================================================
    # PHASE 3: Compute intensity effect
    # ================================================================
    print(f"\n{'='*80}")
    print("  PHASE 3: Best config under different compute intensities")
    print(f"{'='*80}")

    compute_policy = {}
    compute_sweep = overlap_data.get("compute_sweep", {})
    if compute_sweep:
        for size_label in sorted(compute_sweep.keys(), key=lambda s: SIZE_BYTES.get(s, 0)):
            compute_policy[size_label] = {}
            print(f"\n  Size: {size_label}")
            for intensity, data in compute_sweep[size_label].items():
                best_cfg, best_t = find_best(data)
                auto_t = data.get("auto", 0)
                gap = find_auto_gap(data)
                compute_policy[size_label][intensity] = {
                    "best_config": best_cfg,
                    "best_time_ms": best_t,
                    "auto_gap_pct": round(gap, 2),
                }
                print(f"    {intensity:<10s} {best_cfg:<15s} {best_t:>8.3f}ms  gap={gap:+.1f}%")

    # ================================================================
    # PHASE 3.5: CTA × Compute intensity interaction (Block C)
    # ================================================================
    print(f"\n{'='*80}")
    print("  PHASE 3.5: CTA × Compute intensity interaction (4MB, overlap)")
    print(f"{'='*80}")

    cta_compute = channel_data.get("cta_compute_interaction", {})
    cta_compute_policy = {}
    if cta_compute:
        for intensity in sorted(cta_compute.keys()):
            cta_data = cta_compute[intensity]
            best_cta, best_cfg, best_t = find_best_cta(cta_data)
            default_8 = cta_data.get("8", {})
            default_auto = default_8.get("auto", 0)
            improvement = (default_auto - best_t) / default_auto * 100 if default_auto > 0 else 0

            cta_compute_policy[intensity] = {
                "optimal_ctas": int(best_cta),
                "best_config": best_cfg,
                "best_time_ms": best_t,
                "default_8_auto_ms": default_auto,
                "improvement_pct": round(improvement, 2),
            }

            print(f"\n  Compute: {intensity} (4MB overlap)")
            print(f"  {'CTAs':<6s} {'AUTO':>8s} {'tree_s':>8s} {'tree_l':>8s} {'ring_s':>8s} {'ring_l':>8s} | {'Best':>12s} {'Gap':>6s}")
            print(f"  {'-'*75}")
            for cta in sorted(cta_data.keys(), key=int):
                data = cta_data[cta]
                best_c, best_v = find_best(data)
                auto_v = data.get("auto", 0)
                gap = (auto_v - best_v) / auto_v * 100 if auto_v > 0 else 0
                print(f"  {cta:<6s} {auto_v:>8.3f} {data.get('tree_simple',0):>8.3f} "
                      f"{data.get('tree_ll128',0):>8.3f} {data.get('ring_simple',0):>8.3f} "
                      f"{data.get('ring_ll128',0):>8.3f} | {best_c:>12s} {gap:>+5.1f}%")

            print(f"\n  -> Optimal: {best_cta} CTAs, {best_cfg}, {best_t:.3f}ms "
                  f"(vs default CTA=8/AUTO {default_auto:.3f}ms, {improvement:+.1f}%)")

        # Key finding: does optimal CTA shift with compute intensity?
        print(f"\n  KEY FINDING — Optimal CTA by compute intensity:")
        for intensity, info in cta_compute_policy.items():
            print(f"    {intensity:<10s}: CTA={info['optimal_ctas']:>2d}, "
                  f"config={info['best_config']:<15s}, imp={info['improvement_pct']:+.1f}%")
    else:
        print("  No Block C data available.")

    # ================================================================
    # PHASE 4: Winner flips analysis
    # ================================================================
    print(f"\n{'='*80}")
    print("  PHASE 4: Winner flips (sequential → overlap)")
    print(f"{'='*80}")

    flips = []
    sizes_both = set(policy["sequential"].keys()) & set(policy["overlap"].keys())
    print(f"\n  {'Size':<10s} {'SEQ Best':<15s} {'OVL Best':<15s} {'Flip?':>6s}")
    print(f"  {'-'*50}")
    for size in sorted(sizes_both, key=lambda s: SIZE_BYTES.get(s, 0)):
        seq_cfg = policy["sequential"][size]["best_config"]
        ovl_cfg = policy["overlap"][size]["best_config"]
        flipped = seq_cfg != ovl_cfg
        flip_str = "FLIP" if flipped else ""
        if flipped:
            flips.append(size)
        print(f"  {size:<10s} {seq_cfg:<15s} {ovl_cfg:<15s} {flip_str:>6s}")

    print(f"\n  Total flips: {len(flips)}/{len(sizes_both)} "
          f"({len(flips)/len(sizes_both)*100:.0f}%)")
    print(f"  Flipped sizes: {', '.join(flips)}")

    # ================================================================
    # PHASE 5: Generate C header
    # ================================================================
    print(f"\n{'='*80}")
    print("  PHASE 5: Generating C lookup table header")
    print(f"{'='*80}")

    # Build the C lookup table
    # We use size bands matching the bandit plugin:
    # Band 0: < 1KB,  Band 1: 1KB-16KB,  Band 2: 16KB-256KB
    # Band 3: 256KB-1MB,  Band 4: 1MB-8MB,  Band 5: >= 8MB

    # Map our sizes to bands
    size_to_band = {}
    for s, b in SIZE_BYTES.items():
        if b < 1024:
            size_to_band[s] = 0
        elif b < 16 * 1024:
            size_to_band[s] = 1
        elif b < 256 * 1024:
            size_to_band[s] = 2
        elif b < 1024 * 1024:
            size_to_band[s] = 3
        elif b < 8 * 1024 * 1024:
            size_to_band[s] = 4
        else:
            size_to_band[s] = 5

    # For each band, find the best overall config across sizes in that band
    # Priority: use overlap experiment data, fall back to channel data
    NUM_BANDS = 6
    band_policy = {
        "sequential": [None] * NUM_BANDS,
        "overlap": [None] * NUM_BANDS,
    }

    for mode in ["sequential", "overlap"]:
        band_candidates = {b: {} for b in range(NUM_BANDS)}

        for size_label, info in policy[mode].items():
            band = size_to_band.get(size_label)
            if band is not None:
                cfg = info["best_config"]
                t = info["best_time_ms"]
                if cfg not in band_candidates[band] or t < band_candidates[band].get(cfg, 999):
                    band_candidates[band][cfg] = t

        for band in range(NUM_BANDS):
            if band_candidates[band]:
                best = min(band_candidates[band], key=band_candidates[band].get)
                band_policy[mode][band] = CONFIG_TO_IDS.get(best, (-1, -1))
            else:
                band_policy[mode][band] = (-1, -1)  # let NCCL decide

    # CTA recommendations
    cta_recs = {"overlap": {}, "sequential": {}}
    for mode in ["overlap", "sequential"]:
        for size_label, info in cta_policy.get(mode, {}).items():
            band = size_to_band.get(size_label)
            if band is not None:
                cta_recs[mode][band] = info["optimal_ctas"]

    # Generate C header
    header_lines = [
        "/*",
        " * Auto-generated workload-aware NCCL tuner policy table.",
        " * Generated by generate_tuner_policy.py from experimental data.",
        " *",
        " * Platform: 8x A100 (NVLink, single-node)",
        " * Collective: AllReduce",
        " *",
        " * Size bands:",
        " *   0: < 1 KB",
        " *   1: 1 KB - 16 KB",
        " *   2: 16 KB - 256 KB",
        " *   3: 256 KB - 1 MB",
        " *   4: 1 MB - 8 MB",
        " *   5: >= 8 MB",
        " */",
        "",
        "#ifndef TUNER_POLICY_TABLE_H",
        "#define TUNER_POLICY_TABLE_H",
        "",
        "#define POLICY_NUM_BANDS 6",
        "#define POLICY_NUM_MODES 2  // 0=sequential, 1=overlap",
        "",
        "typedef struct {",
        "    int algo;      // NCCL algo index (-1 = let NCCL decide)",
        "    int proto;     // NCCL proto index (-1 = let NCCL decide)",
        "    int nChannels; // recommended channel count (0 = let NCCL decide)",
        "} PolicyEntry;",
        "",
        "// policy_table[overlap_mode][size_band]",
        "static const PolicyEntry policy_table[POLICY_NUM_MODES][POLICY_NUM_BANDS] = {",
    ]

    mode_names = ["sequential", "overlap"]
    for mode_idx, mode in enumerate(mode_names):
        header_lines.append(f"    // Mode {mode_idx}: {mode}")
        header_lines.append("    {")
        for band in range(NUM_BANDS):
            algo, proto = band_policy[mode][band]
            nchan = cta_recs.get(mode, {}).get(band, 0)
            comment = f"band {band}"
            # Find which sizes map to this band
            sizes_in_band = [s for s, b in size_to_band.items() if b == band]
            if sizes_in_band:
                comment += f" ({', '.join(sorted(sizes_in_band, key=lambda x: SIZE_BYTES.get(x,0)))})"

            # Map algo/proto to readable names
            algo_name = {-1: "AUTO", 0: "TREE", 1: "RING"}.get(algo, "?")
            proto_name = {-1: "AUTO", 0: "LL", 1: "LL128", 2: "SIMPLE"}.get(proto, "?")
            comment += f" -> {algo_name}+{proto_name}"

            comma = "," if band < NUM_BANDS - 1 else ""
            header_lines.append(
                f"        {{ {algo:2d}, {proto:2d}, {nchan:2d} }}{comma}  "
                f"/* {comment} */"
            )
        comma = "," if mode_idx < 1 else ""
        header_lines.append(f"    }}{comma}")

    header_lines.extend([
        "};",
        "",
        "#endif // TUNER_POLICY_TABLE_H",
        "",
    ])

    header_path = RESULTS_ROOT / "tuner_policy_table.h"
    header_path.write_text("\n".join(header_lines))
    print(f"  Saved C header: {header_path}")

    # ================================================================
    # Save full policy JSON
    # ================================================================
    full_policy = {
        "platform": "8x A100, NVLink, single-node",
        "collective": "AllReduce",
        "algo_proto_policy": policy,
        "cta_policy": cta_policy,
        "compute_intensity_policy": compute_policy,
        "band_policy": {
            mode: [
                {"algo": band_policy[mode][b][0], "proto": band_policy[mode][b][1]}
                for b in range(NUM_BANDS)
            ]
            for mode in ["sequential", "overlap"]
        },
        "cta_compute_interaction_policy": cta_compute_policy,
        "winner_flips": flips,
        "flip_rate": f"{len(flips)}/{len(sizes_both)}",
    }

    policy_path = RESULTS_ROOT / "tuner_policy.json"
    policy_path.write_text(json.dumps(full_policy, indent=2))
    print(f"  Saved policy JSON: {policy_path}")

    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Winner flips between sequential and overlap: {len(flips)}/{len(sizes_both)}")
    print(f"  Key insight: Under overlap, ring_simple dominates for medium-large messages")

    # Auto gap summary
    total_auto_gaps = []
    for mode in ["sequential", "overlap"]:
        for size, info in policy[mode].items():
            if info["auto_gap_pct"] > 0:
                total_auto_gaps.append((mode, size, info["auto_gap_pct"]))

    total_auto_gaps.sort(key=lambda x: -x[2])
    print(f"\n  Top AUTO gaps:")
    for mode, size, gap in total_auto_gaps[:10]:
        print(f"    {mode:>12s} @ {size:<8s}: {gap:+.1f}%")

    # CTA impact
    if cta_policy.get("overlap"):
        print(f"\n  CTA tuning impact (overlap):")
        for size, info in cta_policy["overlap"].items():
            imp = info["improvement_pct"]
            cta = info["optimal_ctas"]
            print(f"    {size:<8s}: optimal={cta:2d} CTAs, improvement={imp:+.1f}% vs default(8)")

    print(f"\n  Policy files generated:")
    print(f"    {policy_path}")
    print(f"    {header_path}")
    print()


if __name__ == "__main__":
    main()
