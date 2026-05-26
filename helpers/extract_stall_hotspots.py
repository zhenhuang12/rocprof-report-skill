#!/usr/bin/env python3
"""Aggregate AMD per-PC stall samples into per-source-line hotspots.

Reads PC-sampling CSV produced by:
    rocprofv3 --pc-sampling-method host-trap \\
        --pc-sampling-interval 1000 --pc-sampling-unit cycles \\
        --kernel-include-regex "<regex>" \\
        -d <pcsamp_dir> -- ./harness [args]

The CSV columns are typically:
    Dispatch_ID, Sample_Time_ns, Instruction_Address,
    Source (file:line; blank without -gline-tables-only),
    Instruction_Comment (ISA mnemonic), Wait_Reason, Sample_Count

Also supports ATT JSON aggregation (one JSON per CU/SE under att_<tag>/).

Produces in `<run-dir>/analysis/`:
    stall_hotspots_<tag>.txt — top lines ranked by total sample count,
                               with per-Wait_Reason breakdown, plus
                               per-Wait_Reason top lines.

Usage:
    # PC sampling (preferred — lower overhead)
    python3 extract_stall_hotspots.py --run-dir profile/myrun \\
            --pcsamp profile/myrun/reports/pcsamp_<tag>/pc_sampling_host_trap_v0.csv \\
            --tag <tag>

    # ATT (heavier, more detail)
    python3 extract_stall_hotspots.py --run-dir profile/myrun \\
            --att-dir profile/myrun/reports/att_<tag> --tag <tag>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rocprof_utils import load_pcsamp_csv  # noqa: E402


def write_report(per_line, totals_by_wait, out_path, tag, top_n=30):
    """per_line: dict[(file, line)] -> dict[wait_reason -> count]."""
    rows = []
    for (fn, ln), waits in per_line.items():
        total = sum(waits.values())
        rows.append((total, fn, ln, dict(waits)))
    rows.sort(key=lambda x: -x[0])

    grand_total = sum(r[0] for r in rows) or 1

    with open(out_path, "w") as f:
        f.write(f"===== Stall hotspots for {tag} =====\n")
        f.write(f"Distinct (file, line) entries: {len(rows)}\n")
        f.write(f"Total samples: {grand_total}\n\n")
        f.write(f"{'Rank':>4} {'Total':>10} {'Pct':>7}  {'File:Line':<60}  Top wait reasons\n")
        f.write("-" * 160 + "\n")
        for i, (total, fn, ln, waits) in enumerate(rows[:top_n]):
            short = fn if fn else "?"
            # Files may already be 'file:line' from the Source column
            if ln and ln != "?":
                disp = f"{short}:{ln}"
            else:
                disp = str(short)
            breakdown = ", ".join(
                f"{w}={c} ({c/total*100:.0f}%)"
                for w, c in sorted(waits.items(), key=lambda x: -x[1])[:4] if c
            )
            f.write(f"{i:>4} {total:>10} {total/grand_total*100:>6.1f}%  {disp:<60}  {breakdown}\n")

        f.write("\n\n===== Per wait-reason top lines =====\n")
        for wr in sorted(totals_by_wait, key=lambda x: -totals_by_wait[x]):
            f.write(f"\n--- {wr} ({totals_by_wait[wr]} samples, "
                    f"{totals_by_wait[wr]/grand_total*100:.1f}%) ---\n")
            items = [((fn, ln), waits.get(wr, 0)) for (fn, ln), waits in per_line.items()]
            items = [it for it in items if it[1] > 0]
            items.sort(key=lambda x: -x[1])
            for (fn, ln), v in items[:10]:
                disp = f"{fn}:{ln}" if ln and ln != "?" else str(fn or "?")
                f.write(f"  {v:>8}  {disp}\n")


def aggregate_pcsamp(csv_path):
    """Load PC-sampling CSV, return (per_line, totals_by_wait).

    per_line[(file, line)] -> dict[wait_reason -> sample_count]
    totals_by_wait[wait_reason] -> total
    """
    df = load_pcsamp_csv(csv_path)
    per_line = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    src_col = "Source" if "Source" in df.columns else None
    wait_col = "Wait_Reason" if "Wait_Reason" in df.columns else None
    count_col = "Sample_Count" if "Sample_Count" in df.columns else None
    if src_col is None or count_col is None:
        raise RuntimeError(
            f"Expected 'Source' and 'Sample_Count' in {csv_path}; got {df.columns.tolist()}"
        )
    if wait_col is None:
        wait_col = "__no_wait_reason__"
        df[wait_col] = "(unknown)"
    for src, wait, cnt in zip(df[src_col].astype(str), df[wait_col].astype(str), df[count_col]):
        # Source is typically "file:line" — split if possible
        if ":" in src:
            fn, _, ln = src.rpartition(":")
        else:
            fn, ln = src, "?"
        try:
            c = int(cnt)
        except Exception:
            c = 0
        if c <= 0:
            continue
        per_line[(fn, ln)][wait] += c
        totals[wait] += c
    return per_line, totals


def aggregate_att_json_dir(att_dir):
    """Best-effort ATT JSON aggregator. Sums any 'wait_reason' / 'sample_count'
    fields keyed by source. Returns (per_line, totals_by_wait)."""
    per_line = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for jp in glob.glob(str(Path(att_dir) / "**" / "*.json"), recursive=True):
        try:
            data = json.loads(Path(jp).read_text())
        except Exception:
            continue
        # ATT JSON formats vary by ROCm version; we walk the tree heuristically.
        def walk(node):
            if isinstance(node, dict):
                src = node.get("source") or node.get("Source")
                wait = node.get("wait_reason") or node.get("Wait_Reason")
                cnt = node.get("sample_count") or node.get("Sample_Count")
                if src and cnt:
                    if ":" in src:
                        fn, _, ln = src.rpartition(":")
                    else:
                        fn, ln = src, "?"
                    try:
                        c = int(cnt)
                    except Exception:
                        c = 0
                    if c > 0:
                        w = wait or "(unknown)"
                        per_line[(fn, ln)][w] += c
                        totals[w] += c
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(data)
    return per_line, totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--pcsamp", type=Path, action="append", default=[],
                    help="Path to PC-sampling CSV. Pass multiple with repeated flag.")
    ap.add_argument("--att-dir", type=Path, action="append", default=[],
                    help="Path to an att_<tag> directory containing per-SE/CU JSON.")
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    sources = list(args.pcsamp) + list(args.att_dir)
    if len(sources) != len(args.tag):
        ap.error("Total --pcsamp + --att-dir count must equal --tag count")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for src, tag in zip(sources, args.tag):
        if not src.exists():
            print(f"[skip] {src} not found", file=sys.stderr)
            continue
        try:
            if src.is_dir():
                per_line, totals = aggregate_att_json_dir(src)
            else:
                per_line, totals = aggregate_pcsamp(src)
        except Exception as e:
            print(f"[error] {tag}: {e}", file=sys.stderr)
            continue
        out = analysis_dir / f"stall_hotspots_{tag}.txt"
        write_report(per_line, totals, out, tag, top_n=args.top)
        print(f"[{tag}] -> {out} ({len(per_line)} distinct lines, "
              f"{sum(totals.values())} samples)")


if __name__ == "__main__":
    main()
