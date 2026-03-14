"""
Ingest allgather timing .txt files from run_benchmark_modal_allgather.py
and build allgather_multinode_results.json.

Parses files named:
  times_allgather_{sequential|overlap}_{SIZE}_{config}.txt

where SIZE is e.g. 8, 1K, 64K, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB
and config is e.g. auto, ring_simple, ring_ll, ring_ll128, nvls_simple.

Each file contains one float per line (iteration time in ms).
The median is used as the summary statistic per (mode, size, config).

Output JSON structure:
{
  "sequential": {
    "8":    {"auto": 8.42, "ring_simple": 8.31, ...},
    "1K":   {...},
    "256KB": {...},
    ...
  },
  "overlap": {
    ...
  }
}

Usage:
  python ingest_allgather_to_json.py
  python ingest_allgather_to_json.py --results-dir path/to/txt/files
  python ingest_allgather_to_json.py --output custom_output.json
  python ingest_allgather_to_json.py --merge
"""

import argparse
import json
import statistics
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results" / "modal_allgather"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "allgather_multinode_results.json"

VALID_CONFIGS = {
    "auto",
    "ring_simple",
    "ring_ll",
    "ring_ll128",
    "nvls_simple",
}

FILENAME_RE = re.compile(
    r"^times_allgather_(sequential|overlap)_([0-9]+[KMG]?B?)_([a-zA-Z0-9_]+)\.txt$"
)


def parse_times_file(path: Path) -> List[float]:
    """Return list of iteration times in ms, or empty list if invalid."""
    try:
        text = path.read_text()
    except OSError:
        return []

    vals: List[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(float(line))
        except ValueError:
            continue
    return vals


def median(vals: List[float]) -> float:
    vals_sorted = sorted(vals)
    return vals_sorted[len(vals_sorted) // 2]


def ingest_allgather_txt(
    results_dir: Path,
    recursive: bool = True,
    merge_duplicates: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Scan results_dir for times_allgather_*.txt files and build results dict.
    """
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    single_values: Dict[Tuple[str, str, str], float] = {}
    merged_values: Dict[Tuple[str, str, str], List[float]] = {}

    def add_value(mode: str, size_label: str, cfg: str, time_ms: float) -> None:
        key = (mode, size_label, cfg)
        if merge_duplicates:
            merged_values.setdefault(key, []).append(time_ms)
        else:
            if key not in single_values:
                single_values[key] = time_ms

    pattern = "**/times_allgather_*.txt" if recursive else "times_allgather_*.txt"
    for path in results_dir.glob(pattern):
        if not path.is_file():
            continue
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        mode, size_label, cfg = m.groups()
        if cfg not in VALID_CONFIGS:
            continue
        vals = parse_times_file(path)
        if not vals:
            continue
        add_value(mode, size_label, cfg, median(vals))

    data: Dict[str, Dict[str, Dict[str, float]]] = {
        "sequential": {},
        "overlap": {},
    }

    if merge_duplicates:
        items = merged_values.items()
    else:
        items = single_values.items()

    for key, value in items:
        mode, size_label, cfg = key
        if merge_duplicates:
            final_val = round(statistics.mean(value), 3)
        else:
            final_val = round(float(value), 3)

        mode_dict = data.setdefault(mode, {})
        size_dict = mode_dict.setdefault(size_label, {})
        size_dict[cfg] = final_val

    return data


def print_summary_table(data: Dict) -> None:
    """Print a human-readable summary of results."""
    size_order = ["8", "1K", "64K", "256KB", "1MB", "4MB", "16MB", "64MB", "256MB"]

    for mode in ("sequential", "overlap"):
        mode_data = data.get(mode, {})
        if not mode_data:
            continue
        print(f"\n{'='*70}")
        print(f"  {mode.upper()}")
        print(f"{'='*70}")

        sorted_sizes = [s for s in size_order if s in mode_data]
        sorted_sizes += [s for s in mode_data if s not in size_order]

        for size_label in sorted_sizes:
            cfgs = mode_data[size_label]
            valid = {k: v for k, v in cfgs.items() if v > 0}
            if not valid:
                continue
            best_cfg = min(valid, key=valid.get)
            best_t = valid[best_cfg]
            auto_t = valid.get("auto", 0)
            gap = (auto_t - best_t) / auto_t * 100 if auto_t > 0 else 0

            parts = " | ".join(f"{c}: {v:.3f}ms" for c, v in sorted(valid.items()))
            print(f"  {size_label:>8s}  {parts}")
            print(f"           => Best: {best_cfg} ({best_t:.3f}ms), AUTO gap: {gap:+.1f}%")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest allgather timing .txt files and write results JSON."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing times_allgather_*.txt files (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write results JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only look at files directly under --results-dir (no subdirectories).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Average medians from duplicate files for the same (mode, size, config).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable summary table after writing JSON.",
    )

    args = parser.parse_args(argv)

    data = ingest_allgather_txt(
        results_dir=args.results_dir,
        recursive=not args.no_recursive,
        merge_duplicates=args.merge,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))

    num_seq = len(data.get("sequential", {}))
    num_ovl = len(data.get("overlap", {}))
    num_points = sum(
        len(cfgs) for mode_dict in data.values() for cfgs in mode_dict.values()
    )
    print(
        f"Wrote {args.output} with "
        f"{num_points} (mode, size, config) entries "
        f"({num_seq} sequential sizes, {num_ovl} overlap sizes)."
    )

    if args.summary:
        print_summary_table(data)


if __name__ == "__main__":
    main()
