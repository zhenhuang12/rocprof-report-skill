#!/usr/bin/env python3
"""Extract and compare key counters from rocprof-compute output directories.

Produces in `<run_dir>/analysis/`:
    metrics_all_<tag>.json        — every counter sum from pmc_perf.csv
    metrics_key_<tag>.txt/json    — curated MI300X/MI355X key counters
    compare_<tag1>_vs_<tag2>.txt  — side-by-side (when >= 2 reports given)

Usage examples:
    # Single report
    python3 analyze_reports.py --run-dir profile/myrun \\
            --rpc profile/myrun/reports/rpc_<tag> --tag <tag> \\
            --kernel "my_kernel"

    # Multiple reports → side-by-side compare
    python3 analyze_reports.py --run-dir profile/myrun \\
            --rpc profile/myrun/reports/rpc_v1 --tag v1 \\
            --rpc profile/myrun/reports/rpc_v2 --tag v2 \\
            --kernel "my_kernel" --arch gfx942
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rocprof_utils import (  # noqa: E402
    detect_arch, dump_all_counters, dump_key_counters,
    kernel_duration_ns, key_counters_for_arch, load_rpc_dir,
)


_TAG_SAFE = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")


def _safe_tag(tag: str) -> str:
    """Make a tag safe to embed in a filename: keep [A-Za-z0-9._-], else '_'."""
    return "".join(ch if ch in _TAG_SAFE else "_" for ch in tag) or "untagged"


def collect(rpc_dir: Path, tag: str, analysis_dir: Path, kernel_regex, arch) -> dict:
    rpc = load_rpc_dir(rpc_dir, kernel_regex=kernel_regex)
    arch_used = arch or detect_arch(rpc) or "gfx942"

    n_disp = len(rpc["timestamps"])
    dur_ns = kernel_duration_ns(rpc)
    print(f"[{tag}] {rpc_dir}: {n_disp} dispatch(es), total duration {dur_ns/1e3:.2f} µs, arch={arch_used}")

    n = dump_all_counters(rpc, analysis_dir / f"metrics_all_{tag}.json")
    print(f"  -> metrics_all_{tag}.json ({n} counters)")

    key = dump_key_counters(
        rpc, arch_used,
        analysis_dir / f"metrics_key_{tag}.json",
        analysis_dir / f"metrics_key_{tag}.txt",
    )
    print(f"  -> metrics_key_{tag}.{{json,txt}}")
    return key


def compare(collected: dict, analysis_dir: Path, arch):
    tags = list(collected.keys())
    if len(tags) < 2:
        return
    keys = key_counters_for_arch(arch or "gfx942")
    safe_tags = [_safe_tag(t) for t in tags]
    if len(safe_tags) <= 2:
        out_name = f"compare_{'_vs_'.join(safe_tags)}.txt"
    else:
        # Avoid unbounded filename growth; the file header already lists all tags.
        out_name = f"compare_{safe_tags[0]}_vs_{safe_tags[-1]}_and_{len(safe_tags) - 2}_more.txt"
    out_path = analysis_dir / out_name
    with open(out_path, "w") as f:
        col_w = max(20, max(len(t) for t in tags) + 2)
        f.write(f"{'Counter':<60}")
        for t in tags:
            f.write(f"{t:>{col_w}}")
        # "change" always compares LAST vs FIRST; rename to make that explicit
        # when there are 3+ tags.
        change_label = f"{tags[-1]}/{tags[0]}-1"
        f.write(f"{change_label:>14}\n")
        f.write("-" * (60 + col_w * len(tags) + 14) + "\n")

        # Duration row first
        f.write(f"{'__duration_ns__':<60}")
        durs = [collected[t].get("__duration_ns__", 0) for t in tags]
        for d in durs:
            f.write(f"{d:>{col_w}}")
        if durs[0]:
            chg = (durs[-1] - durs[0]) / durs[0] * 100
            f.write(f"{chg:>+9.1f}%\n")
        else:
            f.write("\n")

        for k in keys:
            f.write(f"{k:<60}")
            vals = []
            for t in tags:
                v = collected[t].get(k, "N/A")
                vals.append(v)
                if isinstance(v, (int, float)):
                    f.write(f"{v:>{col_w}.6g}")
                else:
                    f.write(f"{str(v):>{col_w}}")
            if (isinstance(vals[0], (int, float)) and isinstance(vals[-1], (int, float))
                    and vals[0]):
                chg = (vals[-1] - vals[0]) / vals[0] * 100
                f.write(f"{chg:>+9.1f}%\n")
            else:
                f.write("\n")
    print(f"compare -> {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract key AMD PMC counters from rocprof-compute dirs and compare."
    )
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="The profile run directory — outputs go to <run-dir>/analysis/")
    ap.add_argument("--rpc", type=Path, action="append", required=True,
                    help="Path to a rocprof-compute output directory (the `-p` arg). "
                         "Can be passed multiple times.")
    ap.add_argument("--tag", type=str, action="append", required=True,
                    help="Short tag for each --rpc. Must be passed once per --rpc.")
    ap.add_argument("--kernel", type=str, default=None,
                    help="Optional Kernel_Name regex to filter dispatches.")
    ap.add_argument("--arch", type=str, default=None,
                    help="gfx942 / gfx950 / mi300x / mi355x. Auto-detected from sysinfo if omitted.")
    args = ap.parse_args()

    if len(args.rpc) != len(args.tag):
        ap.error("--rpc and --tag counts must match")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    collected = {}
    for rpc, tag in zip(args.rpc, args.tag):
        if not rpc.exists():
            print(f"[skip] {rpc} does not exist", file=sys.stderr)
            continue
        collected[tag] = collect(rpc, tag, analysis_dir, args.kernel, args.arch)

    compare(collected, analysis_dir, args.arch)


if __name__ == "__main__":
    main()
