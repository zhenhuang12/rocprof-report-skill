#!/usr/bin/env python3
"""ASCII-plot rocprof-compute timeseries and per-CU activity.

Two data sources are supported:

1. **rocprof-compute timeseries** (added in ROCm 6.3): writes one row per
   PMC sample instead of one row per kernel. Enable at collection time:
       rocprof-compute profile --timeseries-sampling-rate 1ms \\
           -p $PROFILE_RUN_DIR/reports/rpc_ts_<tag> -- ./harness
   Output: `pmc_perf_timeseries.csv` with column `Sample_Time_ns` + each
   counter as a column. Minimum interval ~1 ms vs NVIDIA PM ~2 µs — for
   sub-millisecond kernels, use ATT instead.

2. **Per-CU SQ_WAVES distribution** from a regular `pmc_perf.csv`: not a
   true timeseries, but reveals per-CU imbalance and tail effects. Needs a
   per-CU counter column (e.g. `SQ_WAVES_CU<N>`); some ROCm builds only
   expose chip-wide aggregates, in which case this mode is a no-op.

Produces `<run-dir>/analysis/timeline_plots.txt` with ASCII plots for each
requested counter. All `--tag` runs are concatenated into the same file
(under per-tag `###### <tag> ######` headers), so re-invoking overwrites the
previous run; rename the file in between if you want to keep both.

Usage:
    # Timeseries mode
    python3 plot_timeline.py --run-dir profile/myrun \\
            --timeseries profile/myrun/reports/rpc_ts_<tag>/pmc_perf_timeseries.csv \\
            --tag <tag>

    # Per-CU mode (uses per-CU columns in pmc_perf.csv when present)
    python3 plot_timeline.py --run-dir profile/myrun \\
            --rpc profile/myrun/reports/rpc_<tag> --tag <tag> --per-cu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError("pandas is required") from e


DEFAULT_COUNTERS = [
    "SQ_WAVES",
    "SQ_BUSY_CYCLES",
    "SQ_WAIT_INST_VMEM",
    "SQ_WAIT_INST_LDS",
    "SQ_WAIT_BARRIER",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    "TCC_HIT_sum",
    "TCC_MISS_sum",
    "TCC_EA0_RDREQ_32B_sum",
    "TCC_EA1_RDREQ_32B_sum",
    "GRBM_GUI_ACTIVE",
]


def ascii_plot(vals, label, max_rows=20, max_cols=80):
    """Return list of ASCII strings rendering the timeseries."""
    if not vals:
        return [f"{label}: no data"]

    vals = [float(v) if v is not None else 0.0 for v in vals]

    # Trim leading/trailing zeros
    lead = 0
    for v in vals:
        if v > 0:
            break
        lead += 1
    trail = 0
    for v in reversed(vals):
        if v > 0:
            break
        trail += 1
    active = vals[lead:len(vals) - trail] if trail else vals[lead:]
    n = len(active)
    if n == 0:
        return [f"{label}: all zero"]

    ncols = min(max_cols, n)
    bucket_size = max(1, n // ncols)
    buckets = []
    for c in range(ncols):
        s = c * bucket_size
        e = min(n, (c + 1) * bucket_size)
        chunk = active[s:e]
        buckets.append(sum(chunk) / len(chunk) if chunk else 0.0)
    mx = max(buckets) if buckets else 1.0
    if mx == 0:
        mx = 1.0

    lines = [
        f"\n{label}",
        f"  (n={n} active samples, leading_zero={lead}, trailing_zero={trail}, max={mx:.3g})",
    ]
    for r in range(max_rows, 0, -1):
        threshold = mx * r / max_rows
        row = "".join("#" if b >= threshold else " " for b in buckets)
        lines.append(f"  {threshold:10.3g} | {row}")
    lines.append("  " + " " * 12 + "-" * len(buckets))
    lines.append("  " + " " * 12 + " (time →)")
    return lines


def plot_timeseries_csv(csv_path, counters, rows, cols):
    df = pd.read_csv(csv_path)
    lines = [f"\n{'=' * 60}\ntimeseries: {csv_path.name}\n{'=' * 60}"]
    cols_in_df = set(df.columns)
    for c in counters:
        if c not in cols_in_df:
            lines.append(f"\n{c}: not present in CSV (cols={sorted(cols_in_df)[:5]}...)")
            continue
        lines.extend(ascii_plot(df[c].tolist(), c, rows, cols))
    return lines


def plot_per_cu(rpc_dir, counters, rows, cols):
    """Plot per-CU distribution from pmc_perf.csv. Reveals workgroup
    imbalance even without timeseries. Two layouts are supported:

    (a) a single CU index column (e.g. `CU_ID`) plus a counter column —
        one row per CU per dispatch.
    (b) per-CU columns suffixed with the CU index
        (e.g. `SQ_WAVES_CU0`, `SQ_WAVES_CU1`, …) — one row per dispatch.

    If neither shape is present (the build only exposes chip-wide sums),
    returns a single explanatory line.
    """
    rpc = Path(rpc_dir)
    pmc_csv = rpc / "pmc_perf.csv"
    if not pmc_csv.exists():
        return [f"\nper-CU mode: no pmc_perf.csv under {rpc}"]
    df = pd.read_csv(pmc_csv)
    lines = [f"\n{'=' * 60}\nper-CU: {pmc_csv}\n{'=' * 60}"]

    cu_col = None
    for cand in ("CU_ID", "cu_id", "Compute_Unit", "ShaderEngine_CU"):
        if cand in df.columns:
            cu_col = cand
            break

    any_drawn = False
    if cu_col is not None:
        for c in counters:
            if c not in df.columns:
                continue
            per_cu = df.groupby(cu_col)[c].sum().sort_index()
            lines.extend(ascii_plot(per_cu.tolist(), f"{c} per CU", rows, cols))
            any_drawn = True
    else:
        import re as _re
        for c in counters:
            pat = _re.compile(rf"^{_re.escape(c)}_CU(\d+)$")
            per_cu_cols = sorted(
                ((int(m.group(1)), col) for col in df.columns
                 if (m := pat.match(col)) is not None),
                key=lambda x: x[0],
            )
            if not per_cu_cols:
                continue
            vals = [float(df[col].sum()) for _, col in per_cu_cols]
            lines.extend(ascii_plot(vals, f"{c} per CU", rows, cols))
            any_drawn = True

    if not any_drawn:
        lines.append(
            f"\nper-CU mode: no per-CU columns found in {pmc_csv}. "
            "This build may only expose chip-wide aggregates; use "
            "`rocprof-compute analyze --block 23` (workgroup imbalance) "
            "or rocprofv3 --att for per-CU evidence."
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--timeseries", type=Path, action="append", default=[],
                    help="Path to pmc_perf_timeseries.csv. Pass multiple with repeated flag.")
    ap.add_argument("--rpc", type=Path, action="append", default=[],
                    help="Path to a rocprof-compute output dir for per-CU plots.")
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--per-cu", action="store_true",
                    help="When using --rpc, draw per-CU distribution instead of timeseries.")
    ap.add_argument("--counter", type=str, action="append", default=None,
                    help="Override default counter list.")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--cols", type=int, default=80)
    args = ap.parse_args()

    sources = list(args.timeseries) + list(args.rpc)
    if len(sources) != len(args.tag):
        ap.error("Total --timeseries + --rpc count must equal --tag count")

    counters = args.counter or DEFAULT_COUNTERS

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    out_lines = []
    for src, tag in zip(sources, args.tag):
        if not src.exists():
            print(f"[skip] {src} not found", file=sys.stderr)
            continue
        out_lines.append(f"\n\n###### {tag} ######")
        if src.is_dir():
            if args.per_cu:
                out_lines.extend(plot_per_cu(src, counters, args.rows, args.cols))
            else:
                # Look for timeseries CSV under it
                cand = src / "pmc_perf_timeseries.csv"
                if cand.exists():
                    out_lines.extend(plot_timeseries_csv(cand, counters, args.rows, args.cols))
                else:
                    out_lines.append(
                        f"\n{src}: no pmc_perf_timeseries.csv. Re-run with "
                        f"`rocprof-compute profile --timeseries-sampling-rate 1ms ...`, "
                        f"or pass --per-cu to plot per-CU distribution instead."
                    )
        else:
            out_lines.extend(plot_timeseries_csv(src, counters, args.rows, args.cols))

    out_path = analysis_dir / "timeline_plots.txt"
    out_path.write_text("\n".join(out_lines))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
