"""
Modal app: dump NCCL tuner header info from the same runtime as the Phase 4 RL tuner.
Same app name as other phases: browser-networking-tests.
Run: modal run get_nccl_tuner_info.py
Output: writes headers to results/nccl_tuner_headers/ and prints paths + struct info.
"""

import os
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Same base image as run_modal.py (CUDA + torch → nvidia-nccl-cu12). No plugin build.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("findutils")
    .run_commands("pip install --upgrade pip")
    .pip_install("torch")
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
)

volume = modal.Volume.from_name("cs244c-nccl-results", create_if_missing=True)
VOLUME_PATH = "/results"
HEADERS_SUBDIR = "nccl_tuner_headers"

app = modal.App("browser-networking-tests")


@app.function(
    name="dump-nccl-tuner-headers",
    image=image,
    timeout=300,
    volumes={VOLUME_PATH: volume},
)
def dump_nccl_tuner_headers():
    """
    Run in the same env as the RL tuner (torch → nvidia-nccl-cu12).
    Finds tuner-related headers, writes them to the volume, returns paths and struct snippet.
    """
    out_dir = Path(VOLUME_PATH) / HEADERS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find nvidia_nccl_cu12 package (bundled with torch)
    try:
        import nvidia_nccl_cu12
        pkg_root = Path(nvidia_nccl_cu12.__file__).resolve().parent
    except ImportError:
        # Fallback: common site-packages locations
        import site
        pkg_root = None
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            d = Path(sp) / "nvidia_nccl_cu12"
            if d.is_dir():
                pkg_root = d
                break
        if pkg_root is None:
            return {
                "error": "nvidia_nccl_cu12 not found",
                "wrote": [],
            }
    pkg_root = Path(pkg_root)

    # Collect all .h files under the package (and one level of include)
    headers_found = []
    for p in pkg_root.rglob("*.h"):
        try:
            rel = p.relative_to(pkg_root)
            headers_found.append((str(rel), p.read_text(errors="replace")))
        except Exception as e:
            headers_found.append((str(p.relative_to(pkg_root)), f"<read error: {e}>"))

    # Also check include dirs that torch/nccl might use
    include_candidates = [
        pkg_root / "include",
        Path("/usr/local/include"),
        Path("/usr/include"),
    ]
    for inc in include_candidates:
        if not inc.is_dir():
            continue
        for p in inc.rglob("*.h"):
            if "nccl" in p.name.lower() or "tuner" in p.name.lower():
                key = f"system/{p.relative_to(inc)}"
                if not any(k == key for k, _ in headers_found):
                    try:
                        headers_found.append((key, p.read_text(errors="replace")))
                    except Exception as e:
                        headers_found.append((key, f"<read error: {e}>"))

    # Write each to volume
    wrote = []
    struct_snippet = None
    for rel_path, content in headers_found:
        safe_name = rel_path.replace("/", "_")
        dest = out_dir / safe_name
        dest.write_text(content, errors="replace")
        wrote.append(str(dest))

        # Capture struct ncclTuner_v5_t or ncclTunerPlugin for plugin ABI
        if "ncclTuner" in content or "tuner" in rel_path.lower():
            # Extract a few lines around the struct definition
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "ncclTuner_v5_t" in line or "ncclTunerPlugin" in line or "getCollInfo" in line and "init" in content:
                    start = max(0, i - 2)
                    end = min(len(lines), i + 15)
                    struct_snippet = {
                        "file": rel_path,
                        "lines": lines[start:end],
                    }
                    break
                if struct_snippet:
                    break

    volume.commit()

    # Return header contents so the local entrypoint can write them to disk
    header_contents = {rel: content for rel, content in headers_found}

    result = {
        "nvidia_nccl_cu12_root": str(pkg_root),
        "wrote": wrote,
        "headers": [rel for rel, _ in headers_found],
        "header_contents": header_contents,
    }
    if struct_snippet:
        result["struct_snippet"] = struct_snippet
    return result


@app.local_entrypoint()
def main():
    """Run the dumper and save headers + summary under results/nccl_tuner_headers/."""
    out = dump_nccl_tuner_headers.remote()

    if out.get("error"):
        print(out["error"], file=sys.stderr)
        sys.exit(1)

    print("nvidia_nccl_cu12_root:", out["nvidia_nccl_cu12_root"])
    print("Headers found:", out["headers"])

    results_dir = Path(__file__).parent / "results" / HEADERS_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)

    for rel, content in out.get("header_contents", {}).items():
        safe_name = rel.replace("/", "_")
        p = results_dir / safe_name
        p.write_text(content, errors="replace")
        print("Wrote", p)

    if out.get("struct_snippet"):
        s = out["struct_snippet"]
        print("\n--- Struct snippet (file: %s) ---" % s["file"])
        print("\n".join(s["lines"]))

    summary_path = results_dir.parent / "nccl_tuner_info.txt"
    lines = [
        "nvidia_nccl_cu12_root: " + out["nvidia_nccl_cu12_root"],
        "headers: " + ", ".join(out["headers"]),
    ]
    if out.get("struct_snippet"):
        lines.append("\nstruct_snippet from " + out["struct_snippet"]["file"] + ":")
        lines.extend(out["struct_snippet"]["lines"])
    summary_path.write_text("\n".join(lines))
    print("\nWrote summary to", summary_path)
    print("Done. Copy needed headers from results/%s/ into phase4-tuner/ to fix the plugin build." % HEADERS_SUBDIR)
