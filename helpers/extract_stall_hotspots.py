#!/usr/bin/env python3
"""Aggregate AMD per-PC stall samples into per-source-line hotspots.

Reads PC-sampling CSV produced by rocprofv3. The two modes produce different
schemas — prefer STOCHASTIC, which is the only mode that emits `Stall_Reason`:

    # Stochastic (preferred — has Stall_Reason):
    rocprofv3 --pc-sampling-beta-enabled \\
        --pc-sampling-method stochastic \\
        --pc-sampling-interval 1048576 --pc-sampling-unit cycles \\
        --kernel-include-regex "<regex>" \\
        -f csv \\
        -d <pcsamp_dir> -- ./harness [args]

    # Host_trap (cheaper, hotspots only — no Stall_Reason column):
    #   --pc-sampling-method host_trap --pc-sampling-interval 1000 --pc-sampling-unit time

See https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html

Output paths (rocprofv3 default `-d <pcsamp_dir>` with no `--output-file`)
nest under `<hostname>/` with a PID prefix:
    <pcsamp_dir>/<hostname>/<pid>_pc_sampling_stochastic.csv
    <pcsamp_dir>/<hostname>/<pid>_pc_sampling_host_trap.csv
Pass `--output-file <prefix>` to rocprofv3 to collapse to a flat
    <pcsamp_dir>/<prefix>_pc_sampling_*.csv
layout. `_resolve_pcsamp_dir` rglobs and handles either form.

Stochastic CSV columns:
    Sample_Timestamp, Exec_Mask, Dispatch_Id, Instruction (PC),
    Instruction_Comment (ISA mnemonic), Correlation_Id,
    Wave_Issued_Instruction (0 = stalled, 1 = issued), Instruction_Type,
    Stall_Reason (populated only when Wave_Issued_Instruction == 0),
    Wave_Count.

Stall_Reason enum values (from
ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_* in
/opt/rocm/include/rocprofiler-sdk/pc_sampling.h): NONE,
NO_INSTRUCTION_AVAILABLE, ALU_DEPENDENCY, WAITCNT, INTERNAL_INSTRUCTION,
BARRIER_WAIT, ARBITER_NOT_WIN, ARBITER_WIN_EX_STALL, OTHER_WAIT, SLEEP_WAIT.

Note: the per-execution-pipe `arb_state_stall_{valu,matrix,lds,lds_direct,
scalar,vmem_tex,flat,exp,misc,brmsg}` bit-fields live in the JSON output
(under the `snapshot` object — use `-f json` instead of `-f csv`), NOT as
CSV columns. This script aggregates only by the CSV `Stall_Reason` column.

Host_trap CSV columns are a strict subset (no Stall_Reason).

Also supports ATT JSON aggregation (one JSON per CU/SE under att_<tag>/).

Produces in `<run-dir>/analysis/`:
    stall_hotspots_<tag>.txt — top lines ranked by total sample count,
                               with per-Stall_Reason breakdown (stochastic only),
                               plus per-Stall_Reason top lines.

Usage:
    # PC sampling (preferred — pass the CSV directly via --pcsamp, or
    # the directory via --pcsamp-dir and let the script glob inside it.
    # The dir glob accepts both the flat layout and the older nested
    # `out/pmc_<N>/<hostname>/...` form as a defensive fallback.)
    python3 extract_stall_hotspots.py --run-dir profile/myrun \\
            --pcsamp-dir profile/myrun/reports/pcsamp_<tag> \\
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

from rocprof_utils import load_pcsamp_csv, safe_tag  # noqa: E402


def write_report(per_line, totals_by_stall, out_path, tag, top_n=30):
    """per_line: dict[(file_or_pc, line)] -> dict[stall_reason -> count]."""
    rows = []
    for (fn, ln), stalls in per_line.items():
        total = sum(stalls.values())
        rows.append((total, fn, ln, dict(stalls)))
    rows.sort(key=lambda x: -x[0])

    grand_total = sum(r[0] for r in rows) or 1

    with open(out_path, "w") as f:
        f.write(f"===== Stall hotspots for {tag} =====\n")
        f.write(f"Distinct entries: {len(rows)}\n")
        f.write(f"Total samples: {grand_total}\n\n")
        f.write(f"{'Rank':>4} {'Total':>10} {'Pct':>7}  {'File:Line (or PC)':<60}  Top stall reasons\n")
        f.write("-" * 160 + "\n")
        for i, (total, fn, ln, stalls) in enumerate(rows[:top_n]):
            short = fn if fn else "?"
            # Files may already be 'file:line' from the Source column
            if ln and ln != "?":
                disp = f"{short}:{ln}"
            else:
                disp = str(short)
            breakdown = ", ".join(
                f"{w}={c} ({c/total*100:.0f}%)"
                for w, c in sorted(stalls.items(), key=lambda x: -x[1])[:4] if c
            )
            f.write(f"{i:>4} {total:>10} {total/grand_total*100:>6.1f}%  {disp:<60}  {breakdown}\n")

        f.write("\n\n===== Per stall-reason top lines =====\n")
        for sr in sorted(totals_by_stall, key=lambda x: -totals_by_stall[x]):
            f.write(f"\n--- {sr} ({totals_by_stall[sr]} samples, "
                    f"{totals_by_stall[sr]/grand_total*100:.1f}%) ---\n")
            items = [((fn, ln), stalls.get(sr, 0)) for (fn, ln), stalls in per_line.items()]
            items = [it for it in items if it[1] > 0]
            items.sort(key=lambda x: -x[1])
            for (fn, ln), v in items[:10]:
                disp = f"{fn}:{ln}" if ln and ln != "?" else str(fn or "?")
                f.write(f"  {v:>8}  {disp}\n")


def aggregate_pcsamp(csv_path):
    """Load PC-sampling CSV, return (per_line, totals_by_stall).

    Handles both the stochastic schema (has `Stall_Reason`) and the host_trap
    schema (per-line hotspots only). When `Source` (file:line) isn't populated,
    falls back to the `Instruction` PC as the grouping key. When `Sample_Count`
    isn't present (the stochastic CSV is one-row-per-sample), counts rows.

    per_line[(file_or_pc, line)] -> dict[stall_reason -> count]
    totals_by_stall[stall_reason] -> total
    """
    df = load_pcsamp_csv(csv_path)
    per_line = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    # Group key: prefer "Source" (file:line) when present, else the PC ("Instruction").
    if "Source" in df.columns:
        src_col = "Source"
        is_source = True
    elif "Instruction" in df.columns:
        src_col = "Instruction"
        is_source = False
    else:
        raise RuntimeError(
            f"Expected 'Source' or 'Instruction' in {csv_path}; got {df.columns.tolist()}"
        )
    # Stall reason: the real column is `Stall_Reason` (stochastic mode).
    # `Wait_Reason` is checked as a back-compat fallback only.
    if "Stall_Reason" in df.columns:
        stall_col = "Stall_Reason"
    elif "Wait_Reason" in df.columns:
        stall_col = "Wait_Reason"  # legacy/back-compat — newer CSVs don't have this
    else:
        stall_col = "__no_stall_reason__"
        df[stall_col] = "(hotspot)"  # host_trap mode: per-line hotspots only
    # Sample weight: rocprof-compute aggregations may have `Sample_Count`; raw
    # stochastic CSVs are one-row-per-sample, so we count rows when absent.
    count_col = "Sample_Count" if "Sample_Count" in df.columns else None

    # Optionally filter to stalled rows when both fields exist.
    if "Wave_Issued_Instruction" in df.columns and stall_col != "__no_stall_reason__":
        df = df[(df["Wave_Issued_Instruction"] == 0) | (df["Wave_Issued_Instruction"] == "0")]

    if count_col is not None:
        iterator = zip(df[src_col].astype(str), df[stall_col].astype(str), df[count_col])
    else:
        iterator = zip(df[src_col].astype(str), df[stall_col].astype(str), [1] * len(df))

    for src, stall, cnt in iterator:
        if is_source:
            fn, ln = _split_source(src)
        else:
            # PC value: keep it intact as the grouping key; no file:line split.
            fn, ln = src, "?"
        if fn == "nan" or fn == "":
            # PC samples with no source attribution would otherwise pile into
            # a single fake "nan" hotspot.
            continue
        try:
            c = int(cnt)
        except (TypeError, ValueError):
            c = 0
        if c <= 0:
            continue
        per_line[(fn, ln)][stall] += c
        totals[stall] += c
    return per_line, totals


_SRC_RE = None


def _split_source(src):
    """Split a "file[:line[:col]]" Source field into (file, line_str).

    Returns ("?", "?") on empty / NaN, and (src, "?") if no trailing numeric
    line component is present.
    """
    global _SRC_RE
    if _SRC_RE is None:
        import re
        _SRC_RE = re.compile(r"^(.*?):(\d+)(?::\d+)?$")
    if src is None:
        return "?", "?"
    s = src.strip()
    if not s:
        return "?", "?"
    m = _SRC_RE.match(s)
    if m:
        return m.group(1), m.group(2)
    return s, "?"


def aggregate_att_json_dir(att_dir):
    """Best-effort ATT JSON aggregator. Sums any 'stall_reason' / 'sample_count'
    fields keyed by source. Returns (per_line, totals_by_stall). Accepts
    `wait_reason` / `Wait_Reason` keys as a back-compat alias for older traces."""
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
                stall = (
                    node.get("stall_reason")
                    or node.get("Stall_Reason")
                    or node.get("wait_reason")      # legacy alias
                    or node.get("Wait_Reason")      # legacy alias
                )
                cnt = node.get("sample_count") or node.get("Sample_Count")
                if src and cnt:
                    fn, ln = _split_source(str(src))
                    if fn in ("?", "", "nan"):
                        for v in node.values():
                            walk(v)
                        return
                    try:
                        c = int(cnt)
                    except (TypeError, ValueError):
                        c = 0
                    if c > 0:
                        w = stall or "(unknown)"
                        per_line[(fn, ln)][w] += c
                        totals[w] += c
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(data)
    return per_line, totals


def _resolve_pcsamp_dir(d):
    """Recursively glob a pcsamp_<tag> directory for the PC-sampling CSV.

    rocprofv3's default `-d <pcsamp_dir>` (no `--output-file`) nests under
    `<hostname>/` with a PID prefix:
        `<pcsamp_dir>/<hostname>/<pid>_pc_sampling_stochastic.csv` or
        `<pcsamp_dir>/<hostname>/<pid>_pc_sampling_host_trap.csv`.
    Passing `--output-file <prefix>` collapses that to a flat
        `<pcsamp_dir>/<prefix>_pc_sampling_*.csv`.
    Both forms carry a `<pid>_` / `<prefix>_` underscore prefix — rocprofv3
    does NOT emit a bare `pc_sampling_*.csv`. We rglob with the underscore-
    prefix glob to catch both layouts regardless of depth.

    When both stochastic and host_trap CSVs are present, we prefer **stochastic**
    — it's the only mode that emits the `Stall_Reason` column needed for a true
    wait-reason breakdown. Caller can pass --pcsamp directly to override.
    """
    base = Path(d)
    # Flat layout (--output-file <prefix>) plus nested default-layout
    # (<hostname>/<pid>_*) covered by a single rglob.
    matches = sorted(base.glob("*_pc_sampling_*.csv"))
    matches += sorted(base.rglob("*_pc_sampling_*.csv"))
    # De-dup while preserving order.
    seen = set()
    matches = [m for m in matches if not (m in seen or seen.add(m))]
    if not matches:
        raise FileNotFoundError(
            f"No pc_sampling_*.csv found under {base}; check rocprofv3 -f csv was set"
        )
    if len(matches) > 1:
        # Prefer stochastic (has Stall_Reason); fall back to host_trap.
        st = [m for m in matches if "stochastic" in m.name]
        if st:
            return st[0]
        ht = [m for m in matches if "host_trap" in m.name]
        return ht[0] if ht else matches[0]
    return matches[0]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate PC-sampling / ATT data into per-line stall hotspots.\n\n"
            "TAG ORDERING: --tag values map to inputs in this fixed order: "
            "ALL --pcsamp first, THEN all --pcsamp-dir, THEN all --att-dir "
            "(regardless of the order you typed the flags on the command line). "
            "Group your --tag flags in the same order or the labels will be wrong."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--pcsamp", type=Path, action="append", default=[],
                    help="Path to PC-sampling CSV. Pass multiple with repeated flag. "
                         "Consumed FIRST when zipping against --tag.")
    ap.add_argument("--pcsamp-dir", type=Path, action="append", default=[],
                    help="Path to a pcsamp_<tag> dir; globs pc_sampling_*.csv inside. "
                         "Consumed SECOND when zipping against --tag.")
    ap.add_argument("--att-dir", type=Path, action="append", default=[],
                    help="Path to an att_<tag> directory containing per-SE/CU JSON. "
                         "Consumed LAST when zipping against --tag.")
    ap.add_argument("--tag", type=str, action="append", required=True,
                    help="One per input. Order must match --pcsamp, then --pcsamp-dir, "
                         "then --att-dir (see ordering note above).")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    # Resolve pcsamp-dir to concrete CSV paths
    pcsamp_resolved = []
    for d in args.pcsamp_dir:
        try:
            pcsamp_resolved.append(_resolve_pcsamp_dir(d))
        except FileNotFoundError as e:
            print(f"[skip] {e}", file=sys.stderr)
            pcsamp_resolved.append(None)

    sources = list(args.pcsamp) + pcsamp_resolved + list(args.att_dir)
    if not sources or any(s is None for s in sources):
        print("[error] no PC-sampling / ATT inputs resolved — nothing to do",
              file=sys.stderr)
        sys.exit(2)
    if len(sources) != len(args.tag):
        ap.error("Total --pcsamp + --pcsamp-dir + --att-dir count must equal --tag count")

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
        out = analysis_dir / f"stall_hotspots_{safe_tag(tag)}.txt"
        write_report(per_line, totals, out, tag, top_n=args.top)
        print(f"[{tag}] -> {out} ({len(per_line)} distinct lines, "
              f"{sum(totals.values())} samples)")


if __name__ == "__main__":
    main()
