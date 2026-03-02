"""
Ingest multi-node timing .txt files and build multinode_results.json.

We expect two sets of .txt files: one for sequential (no overlap) and one for overlap,
for each protocol/algorithm configuration.

Supports two file formats:

  1. Modal / phase4 multinode: times_mn_{sequential|overlap}_{SIZE}_{config}.txt
     One float per line (iteration time in ms). Mode is in the filename.

  2. AWS / normal NCCL benchmarks: results_{n}gpu_allreduce_{config}_sequential.txt
     and results_{n}gpu_allreduce_{config}_overlap.txt
     Stdout table with "# size(B)  time(us)  ..." and data lines: size (bytes), time (us).
     One file per config per mode; the _sequential or _overlap suffix is required.
     AWS files without this suffix are ignored.

The output JSON matches the structure expected by cross_experiment_analysis.py:

{
  "sequential": {
    "256KB": {"auto": 1.23, "tree_simple": 1.11, ...},
    "1MB":   {...},
    ...
  },
  "overlap": {
    ...
  }
}

Usage examples (from a100-8gpu-new/):

  python ingest_multinode_txt_to_json.py
  python ingest_multinode_txt_to_json.py --results-dir ../aws-multinode-g5.xlarge-2nodes
  python ingest_multinode_txt_to_json.py --merge
  python ingest_multinode_txt_to_json.py --plot
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results" / "multinode_experiment"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "multinode_results.json"

VALID_CONFIGS = {
    "auto",
    "tree_simple",
    "tree_ll128",
    "ring_simple",
    "ring_ll128",
}

# Map NCCL benchmark config names to our canonical keys (for AWS-style filenames).
CONFIG_ALIASES = {
    "tree_ll": "tree_ll128",
    "ring_ll": "ring_ll128",
}

# Figure 6 uses these sizes; map byte counts from benchmark output to labels.
BYTES_TO_SIZE_LABEL = {
    262144: "256KB",
    1048576: "1MB",
    4194304: "4MB",
    16777216: "16MB",
    67108864: "64MB",
    268435456: "256MB",
}

FILENAME_RE = re.compile(
    r"^times_mn_(sequential|overlap)_([0-9]+(?:KB|MB))_([a-zA-Z0-9_]+)\.txt$"
)
# Required (_sequential|_overlap) suffix: we expect two sets of files per config.
AWS_FILENAME_RE = re.compile(
    r"^results_(\d+)gpu_allreduce_([a-zA-Z0-9_]+)(_sequential|_overlap)\.txt$"
)


def parse_times_file(path: Path) -> float:
    """Return median time in ms from a times_mn_*.txt file, or 0.0 if empty/invalid."""
    try:
        text = path.read_text()
    except OSError:
        return 0.0

    vals: List[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(float(line))
        except ValueError:
            continue

    if not vals:
        return 0.0

    vals.sort()
    return float(vals[len(vals) // 2])


def _normalize_config(cfg: str) -> Optional[str]:
    """Return canonical config key or None if not one we use."""
    if cfg in VALID_CONFIGS:
        return cfg
    return CONFIG_ALIASES.get(cfg)


def parse_aws_results_file(path: Path) -> Dict[str, float]:
    """
    Parse AWS / normal NCCL benchmark output: size(B), time(us) table.
    Returns dict mapping size label (e.g. '256KB') to time in ms.
    Only includes sizes in BYTES_TO_SIZE_LABEL; time(us) is converted to ms.
    """
    out: Dict[str, float] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("("):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            size_bytes = int(parts[0])
            time_us = float(parts[1])
        except (ValueError, IndexError):
            continue
        label = BYTES_TO_SIZE_LABEL.get(size_bytes)
        if label is None:
            continue
        out[label] = time_us / 1000.0  # us -> ms
    return out


def ingest_multinode_txt(
    results_dir: Path,
    recursive: bool = True,
    merge_duplicates: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Scan results_dir for multi-node timing files and build multinode_results.json-style data.

    Supports:
      - Modal: times_mn_{sequential|overlap}_{SIZE}_{config}.txt (one median per file).
      - AWS / NCCL benchmarks: results_{n}gpu_allreduce_{config}.txt (table of sizes × time_us).

    Duplicate handling:
      - If merge_duplicates is False: first file found for a (mode, size, config) wins.
      - If merge_duplicates is True: average the per-file values for that key.
    """
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    # When merge_duplicates is False, store a single value per key.
    single_values: Dict[Tuple[str, str, str], float] = {}

    # When merge_duplicates is True, store a list of values per key.
    merged_values: Dict[Tuple[str, str, str], List[float]] = {}

    def add_value(mode: str, size_label: str, cfg: str, time_ms: float) -> None:
        key = (mode, size_label, cfg)
        if merge_duplicates:
            merged_values.setdefault(key, []).append(time_ms)
        else:
            if key not in single_values:
                single_values[key] = time_ms

    # 1) Modal-format: times_mn_*.txt
    pattern_modal = "**/times_mn_*.txt" if recursive else "times_mn_*.txt"
    for path in results_dir.glob(pattern_modal):
        if not path.is_file():
            continue
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        mode, size_label, cfg = m.groups()
        if cfg not in VALID_CONFIGS:
            continue
        median_ms = parse_times_file(path)
        add_value(mode, size_label, cfg, median_ms)

    # 2) AWS / NCCL benchmark format: results_*gpu_allreduce_*_sequential.txt and *_overlap.txt (two sets per config)
    pattern_aws = "**/results_*gpu_allreduce_*.txt" if recursive else "results_*gpu_allreduce_*.txt"
    for path in results_dir.glob(pattern_aws):
        if not path.is_file():
            continue
        m = AWS_FILENAME_RE.match(path.name)
        if not m:
            continue
        cfg_raw = m.group(2)
        suffix = m.group(3)  # _sequential or _overlap (required)
        cfg = _normalize_config(cfg_raw)
        if cfg is None:
            continue
        size_to_ms = parse_aws_results_file(path)
        mode = "sequential" if suffix == "_sequential" else "overlap"
        for size_label, time_ms in size_to_ms.items():
            add_value(mode, size_label, cfg, time_ms)

    # Build final data structure.
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
            avg = statistics.mean(value) if value else 0.0
            final_val = round(float(avg), 3)
        else:
            final_val = round(float(value), 3)

        mode_dict = data.setdefault(mode, {})
        size_dict = mode_dict.setdefault(size_label, {})
        size_dict[cfg] = final_val

    return data


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest multi-node timing .txt files (Modal times_mn_*.txt or AWS results_*gpu_allreduce_*.txt) "
            "and write multinode_results.json compatible with cross_experiment_analysis.paper_figure6."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory containing timing .txt files "
            f"(default: {DEFAULT_RESULTS_DIR})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Path to write multinode_results.json "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only look at files directly under --results-dir (no subdirectories).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "If set, average medians from duplicate files for the same "
            "(mode, size, config) instead of taking the first one."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "If set, call cross_experiment_analysis.paper_figure6 on the "
            "constructed data after writing the JSON."
        ),
    )

    args = parser.parse_args(argv)

    data = ingest_multinode_txt(
        results_dir=args.results_dir,
        recursive=not args.no_recursive,
        merge_duplicates=args.merge,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))

    num_seq_sizes = len(data.get("sequential", {}))
    num_ovl_sizes = len(data.get("overlap", {}))
    num_points = sum(
        len(cfgs) for mode_dict in data.values() for cfgs in mode_dict.values()
    )
    print(
        f"Wrote {args.output} with "
        f"{num_points} (mode, size, config) entries "
        f"({num_seq_sizes} sequential sizes, {num_ovl_sizes} overlap sizes)."
    )

    if args.plot:
        # cross_experiment_analysis.py lives alongside this script.
        sys.path.insert(0, str(HERE))
        try:
            from cross_experiment_analysis import paper_figure6
        except ImportError as exc:
            raise SystemExit(f"Failed to import paper_figure6: {exc}") from exc

        print("Generating Figure 6 (multi-node) using in-memory data...")
        paper_figure6(data)


if __name__ == "__main__":
    main()

