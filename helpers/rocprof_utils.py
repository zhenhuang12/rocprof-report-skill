"""Shared helpers for parsing rocprofv3 / rocprof-compute outputs.

Designed to work with both:
  - rocprof-compute (ROCm 6.3+): with an explicit `-p <path>` and the
    default `--subpath gpu`, artifacts land FLAT directly under `<-p>`:
    `pmc_perf.csv`, `timestamps.csv` (per-dispatch Start/End_Timestamp),
    `sysinfo.csv` (wide single-row format, NOT param/value),
    `roofline.csv` (when roofline ran; default-on, suppress with
    `--no-roof`), `empirRoof_gpu-0_<datatypes>.pdf` PDF plots (only with
    `--roof-only` / `--kernel-names`), `log.txt`, `profiling_config.yaml`,
    plus raw per-PMC-group CSVs under `out/pmc_<N>/<hostname>/<pid>_*.csv`.
    The default `--subpath "gpu"` matches neither nesting branch in
    `rocprof_compute_base.py`; only `--subpath gpu_model` (adds
    `<gpu_model>/`), `--subpath node_name` (adds `<hostname>/`), or
    omitting `-p` entirely (auto-appends `<name>/<gpu_model>/`) injects a
    child dir. `load_rpc_dir` accepts EITHER layout — see
    `_resolve_workload_dir`.
  - ROCm 7.x: rocprofv3 defaults to a single `.db` per run using the
    rocpd schema (the optional `rocpd` Python helper, or plain sqlite3).

Usage:
    from rocprof_utils import (
        load_rpc_dir, safe_col, key_counters_for_arch,
        dump_all_counters, load_pcsamp_csv, per_kernel_durations_from_db,
    )

This module avoids hard dependencies on `rocpd` — it falls back to plain
`sqlite3` queries against `.db` files when the helper isn't importable.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pandas is required: python3 -m pip install --user pandas"
    ) from e


# --- Filename helpers --------------------------------------------------------

_TAG_SAFE = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")


def safe_tag(tag: str) -> str:
    """Make a tag safe to embed in a filename: keep [A-Za-z0-9._-], else '_'."""
    return "".join(ch if ch in _TAG_SAFE else "_" for ch in tag) or "untagged"


# --- Loading rocprof-compute output dirs ------------------------------------

def _resolve_workload_dir(rpc_dir: Path) -> Path:
    """Resolve the directory that actually contains pmc_perf.csv.

    Default rocprof-compute (explicit `-p`, `--subpath gpu`) writes flat
    directly under `<-p>`, so the common case is `rpc_dir/pmc_perf.csv`.
    If the caller opted into a nested layout (`--subpath gpu_model`,
    `--subpath node_name`, or omitted `-p` entirely so the auto
    `<name>/<gpu_model>/` append fires), we glob one level down to find
    the real workload dir.
    """
    if (rpc_dir / "pmc_perf.csv").exists():
        return rpc_dir
    hits = sorted(rpc_dir.glob("*/pmc_perf.csv"))
    if hits:
        return hits[0].parent
    return rpc_dir


def load_rpc_dir(rpc_dir, kernel_regex=None, kernel_trace_csv=None):
    """Load the standard rocprof-compute output tree into a dict of DataFrames.

    Args:
        rpc_dir: path to a rocprof-compute profile output (the `-p` arg).
            Default layout is flat directly under `-p`; opt-in nested
            layouts (a `<gpu_model>/` or `<hostname>/` child) are also
            accepted and resolved via glob.
        kernel_regex: if given, filter pmc CSVs to dispatches whose
            Kernel_Name matches this regex (column lives in pmc_perf.csv).
        kernel_trace_csv: optional path to a rocprofv3 `kernel_trace.csv`
            produced by a separate `--kernel-trace` run; loaded into
            out['kernel_trace']. rocprof-compute itself emits per-dispatch
            timestamps via the sibling `timestamps.csv`, which we load
            into out['timestamps'] automatically when present.

    Returns:
        dict with keys:
            'pmc'           — DataFrame from pmc_perf.csv (all counters)
            'sysinfo'       — DataFrame from sysinfo.csv (wide single-row format)
            'timestamps'    — DataFrame from rocprof-compute's timestamps.csv (if present)
            'kernel_trace'  — DataFrame from the supplied kernel_trace.csv, if any
            'rpc_dir'       — original input path (pre-resolve)
            'workload_dir'  — resolved dir actually containing pmc_perf.csv
    """
    rpc_input = Path(rpc_dir)
    if not rpc_input.is_dir():
        raise FileNotFoundError(f"rocprof-compute dir not found: {rpc_input}")
    rpc = _resolve_workload_dir(rpc_input)

    out = {"rpc_dir": rpc_input, "workload_dir": rpc}

    pmc_path = rpc / "pmc_perf.csv"
    if pmc_path.exists():
        pmc = pd.read_csv(pmc_path)
        if kernel_regex is not None and "Kernel_Name" in pmc.columns:
            mask = pmc["Kernel_Name"].astype(str).str.contains(
                kernel_regex, regex=True
            )
            pmc = pmc[mask].copy()
        out["pmc"] = pmc
    else:
        out["pmc"] = pd.DataFrame()

    sys_path = rpc / "sysinfo.csv"
    out["sysinfo"] = pd.read_csv(sys_path) if sys_path.exists() else pd.DataFrame()

    ts_path = rpc / "timestamps.csv"
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        if kernel_regex is not None and "Kernel_Name" in ts.columns:
            ts = ts[ts["Kernel_Name"].astype(str).str.contains(
                kernel_regex, regex=True)].copy()
        out["timestamps"] = ts
    else:
        out["timestamps"] = pd.DataFrame()

    if kernel_trace_csv is not None:
        ktp = Path(kernel_trace_csv)
        if ktp.exists():
            kt = pd.read_csv(ktp)
            if kernel_regex is not None and "Kernel_Name" in kt.columns:
                kt = kt[kt["Kernel_Name"].astype(str).str.contains(
                    kernel_regex, regex=True)].copy()
            out["kernel_trace"] = kt
        else:
            out["kernel_trace"] = pd.DataFrame()
    else:
        out["kernel_trace"] = pd.DataFrame()

    return out


# --- Safe counter access ----------------------------------------------------

def safe_col(df, name, default=None):
    """Return df[name] if present, else `default`. Counter names vary by gfx."""
    if df is None or df.empty:
        return default
    return df[name] if name in df.columns else default


def safe_col_sum(df, name, default=0.0):
    """Return float(df[name].sum()) if present, else `default`."""
    s = safe_col(df, name, None)
    if s is None:
        return default
    try:
        return float(s.sum())
    except Exception:
        return default


def first_present(df, *names, default=None):
    """Return df[name] for the first matching name. Useful when a counter has
    multiple historical names across gfx generations."""
    if df is None or df.empty:
        return default
    for n in names:
        if n in df.columns:
            return df[n]
    return default


def enumerate_counters(rpc_dir):
    """Return the set of all column names in pmc_perf.csv."""
    rpc = _resolve_workload_dir(Path(rpc_dir))
    seen = set()
    pmc = rpc / "pmc_perf.csv"
    if pmc.exists():
        try:
            seen.update(pd.read_csv(pmc, nrows=1).columns)
        except Exception:
            pass
    return seen


# --- Kernel duration --------------------------------------------------------

def kernel_duration_ns(rpc):
    """Total duration across selected dispatches, in nanoseconds.

    Prefers rocprof-compute's `timestamps.csv` (loaded into rpc['timestamps']
    by `load_rpc_dir`) when present, else falls back to a rocprofv3
    `kernel_trace.csv` loaded into rpc['kernel_trace']. pmc_perf.csv itself
    does NOT carry per-dispatch Start_Timestamp / End_Timestamp columns.
    """
    if isinstance(rpc, dict):
        ts = rpc.get("timestamps")
        if ts is None or getattr(ts, "empty", True):
            ts = rpc.get("kernel_trace")
    else:
        ts = rpc
    if ts is None or getattr(ts, "empty", True):
        return 0
    if "Start_Timestamp" not in ts.columns or "End_Timestamp" not in ts.columns:
        return 0
    return int((ts["End_Timestamp"] - ts["Start_Timestamp"]).sum())


# --- Curated key counter sets -----------------------------------------------
#
# Lists of PMC names known to exist and return meaningful values on the
# given gfx target with ROCm 6.4 / 7.x. For a fuller list and rationale see
# ../reference/08-mi300x-mi355x-counter-names.md. Other gfx targets and future
# ROCm releases may need alternate names — always verify with
# `enumerate_counters(rpc_dir)` or `rocprofv3 -L` (long form `--list-avail`;
# verified against `rocprofv3 --help` on ROCm 7.x).

_COMMON_GEOMETRY = [
    # pmc_perf.csv per-dispatch columns (not strictly PMCs but useful).
    # Verified column names on rocprof-compute ROCm 7.x:
    #   Arch_VGPR  (per work-item architectural VGPR count; NOT "VGPRs")
    #   Accum_VGPR (per work-item AGPR pool on CDNA3+; NOT "AGPRs")
    #   SGPR       (per wavefront; singular, NOT "SGPRs")
    # Start_Timestamp / End_Timestamp are NOT in pmc_perf.csv — they live
    # in the rocprofv3 kernel_trace.csv from a separate --kernel-trace run.
    "Dispatch_ID", "Kernel_Name", "GPU_ID",
    "Grid_Size", "Workgroup_Size",
    "LDS_Per_Workgroup", "Scratch_Per_Workitem",
    "Arch_VGPR", "Accum_VGPR", "SGPR",
    # Wave_Size is NOT in pmc_perf.csv on gfx942/gfx950 — wave size is fixed at
    # 64 on CDNA and reported in sysinfo.csv. Listing it here produces a
    # spurious "N/A" column in the compare output, so it's omitted.
]

_COMMON_SQ = [
    # Wave / instruction mix
    "SQ_WAVES", "SQ_INSTS", "SQ_INSTS_VALU", "SQ_INSTS_SALU",
    "SQ_INSTS_VMEM", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR",
    "SQ_INSTS_SMEM", "SQ_INSTS_FLAT",
    # NOTE: SQ_INSTS_FLAT_LDS_ONLY does NOT exist on gfx942/gfx950. The real
    # LDS-traffic counters are SQ_INSTS_LDS and SQ_LDS_BANK_CONFLICT below.
    "SQ_INSTS_LDS", "SQ_INSTS_BRANCH", "SQ_INSTS_MFMA",
    "SQ_BUSY_CYCLES",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    # Wait reasons — VERIFIED set on gfx942 / gfx950:
    # ONLY three SQ_WAIT_* PMCs exist (SQ_WAIT_ANY, SQ_WAIT_INST_ANY,
    # SQ_WAIT_INST_LDS) plus SQ_INST_LEVEL_LDS. Granular VMEM / SMEM / FLAT /
    # BARRIER / VMCNT / LGKMCNT / EXPCNT / MISC classification is PC-sampling-
    # only on these gens — see load_pcsamp_csv() / stall_hotspots() below.
    "SQ_WAIT_ANY", "SQ_WAIT_INST_ANY", "SQ_WAIT_INST_LDS",
    "SQ_INST_LEVEL_LDS",
    # LDS
    "SQ_LDS_BANK_CONFLICT", "SQ_LDS_IDX_ACTIVE", "SQ_LDS_ATOMIC_RETURN",
    "SQ_LDS_UNALIGNED_STALL",
]

_COMMON_CACHE = [
    # vL1 (TCP)
    "TCP_TOTAL_CACHE_ACCESSES_sum",
    "TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum",
    "TCP_TCC_ATOMIC_WITH_RET_REQ_sum", "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum",
    "TCP_PENDING_STALL_CYCLES_sum",
    # L2 (TCC) — VERIFIED aggregate counters on gfx942 / gfx950:
    # TCC_HIT_sum, TCC_MISS_sum, TCC_REQ_sum, TCC_ATOMIC_sum.
    # TCC_REQ_READ_sum / TCC_REQ_WRITE_sum are NOT in the verified set on
    # these gens — confirm with `rocprofv3 -L | grep '^TCC_'` on your install.
    "TCC_HIT_sum", "TCC_MISS_sum",
    "TCC_REQ_sum",
    "TCC_ATOMIC_sum",
]

_COMMON_HBM = [
    # HBM read — TCC_EA1_* does NOT exist on gfx942 / gfx950 (single EA
    # channel per XCD). Older gfx906/gfx908 EA0+EA1 formulas do not apply.
    "TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_32B_sum", "TCC_EA0_RDREQ_DRAM_sum",
    # HBM write
    "TCC_EA0_WRREQ_sum", "TCC_EA0_WRREQ_64B_sum", "TCC_EA0_WRREQ_DRAM_sum",
    # Atomic / IO. Bare TCC_EA0_{RD,WR}REQ_IO_sum do NOT exist on gfx942/gfx950
    # — the verified IO-side counters are 32B-granular requests + credit-stall.
    "TCC_EA0_ATOMIC_sum",
    "TCC_EA0_RDREQ_IO_32B_sum",
    "TCC_EA0_RDREQ_IO_CREDIT_STALL_sum",
    "TCC_EA0_WRREQ_IO_CREDIT_STALL_sum",
]

_COMMON_GRBM = [
    # Verified gfx942 / gfx950 set (rocprofv3 -L | grep '^GRBM_'). GRBM_SDMA_BUSY
    # and GRBM_GDS_BUSY are NOT exposed on these gens — they appeared on older
    # gfx (906/908). Use amd-smi / HIP trace for SDMA copy activity.
    "GRBM_GUI_ACTIVE", "GRBM_COUNT", "GRBM_CP_BUSY",
    "GRBM_CPC_BUSY", "GRBM_CPF_BUSY",
    "GRBM_EA_BUSY", "GRBM_SPI_BUSY",
    "GRBM_TA_BUSY", "GRBM_TC_BUSY",
    "GRBM_UTCL2_BUSY",
]

# Per-dtype MFMA op counts on gfx942 / gfx950 use the prefix
#   SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>
# The per-tile-shape names (e.g. SQ_INSTS_MFMA_F32_16X16X16BF16) are NOT a
# stable PMC set across ROCm releases — they're sometimes derived metrics, and
# the exact spelling shifts between gfx942 and gfx950. Always confirm with
# `rocprofv3 -L | grep -i mfma` before extending these lists.
#
# CDNA3 (gfx942 / MI300X / MI300A). VERIFIED via `rocprofv3 -L | grep MFMA`
# on gfx942: F16, BF16, F32, F64, I8, F8 (FNUZ), XF32 (TF32-equivalent).
# There is NO dedicated _BF8 counter — E5M2 inputs (and mixed f8_bf8 shapes)
# are bucketed under SQ_INSTS_VALU_MFMA_MOPS_F8.
MI300X_MFMA = [
    "SQ_INSTS_MFMA",                            # aggregate MFMA op count
    "SQ_INSTS_VALU_MFMA_MOPS_F16",
    "SQ_INSTS_VALU_MFMA_MOPS_BF16",
    "SQ_INSTS_VALU_MFMA_MOPS_F32",
    "SQ_INSTS_VALU_MFMA_MOPS_F64",
    "SQ_INSTS_VALU_MFMA_MOPS_I8",
    "SQ_INSTS_VALU_MFMA_MOPS_F8",               # all FP8 (E4M3+E5M2); FNUZ on CDNA3, OCP on CDNA4
    "SQ_INSTS_VALU_MFMA_MOPS_XF32",             # XF32 = TF32-equivalent; present on BOTH gfx942 and gfx950
]

# CDNA4 (gfx950 / MI355X) — adds only the block-scaled F6F4 family on top of
# the gfx942 set. No separate _F4 / _F6 / _MXFP4 / _MXFP6 / _MXFP8 PMCs exist
# — they roll up into _F6F4.
MI355X_MFMA = MI300X_MFMA + [
    "SQ_INSTS_VALU_MFMA_MOPS_F6F4",
]

MI300X_KEY_COUNTERS = (
    _COMMON_GEOMETRY + _COMMON_SQ + _COMMON_CACHE + _COMMON_HBM
    + _COMMON_GRBM + MI300X_MFMA
)

MI355X_KEY_COUNTERS = (
    _COMMON_GEOMETRY + _COMMON_SQ + _COMMON_CACHE + _COMMON_HBM
    + _COMMON_GRBM + MI355X_MFMA
)

# Peak HBM bandwidth (per package), GB/s.
PEAK_HBM_BW_GBPS = {
    "gfx942": 5300.0,    # MI300X HBM3
    "gfx950": 8000.0,    # MI355X HBM3E
}


def key_counters_for_arch(arch):
    """Return the curated key-counter list for a gfx target.

    Args:
        arch: 'gfx942' / 'mi300x' / 'mi300a' or 'gfx950' / 'mi355x'.
    """
    a = arch.lower().strip()
    if a in ("gfx942", "mi300x", "mi300a"):
        return MI300X_KEY_COUNTERS
    if a in ("gfx950", "mi355x"):
        return MI355X_KEY_COUNTERS
    raise ValueError(f"Unknown arch: {arch}")


def detect_arch(rpc):
    """Best-effort guess of gfx target from rocprof-compute sysinfo.

    sysinfo.csv on current rocprof-compute is a WIDE single-row CSV (column
    names like `gpu_arch`, `num_xcd`, `gpu_model`, etc.), not a param/value
    table. We flatten the row into a string and substring-match.

    Returns one of 'gfx942', 'gfx950', or None.
    """
    sysinfo = rpc.get("sysinfo") if isinstance(rpc, dict) else None
    if sysinfo is None or sysinfo.empty:
        return None
    try:
        # iloc[0] is the single sysinfo row; flatten all cells to lower-cased str.
        row = sysinfo.iloc[0]
        s = " ".join(str(v) for v in row.values).lower()
    except Exception:
        s = sysinfo.astype(str).agg(" ".join, axis=1).str.lower().str.cat(sep=" ")
    if "gfx950" in s or "mi355" in s:
        return "gfx950"
    if "gfx942" in s or "mi300" in s:
        return "gfx942"
    return None


# --- Aggregate counter dumps ------------------------------------------------

# Per-dispatch geometry/identity columns that pmc_perf.csv carries alongside
# the actual hardware counters. Summing these across dispatches produces
# nonsense (e.g. summed Grid_Size or Arch_VGPR), so we report them under a
# distinct "geometry::" namespace and use first/mean rather than sum.
_GEOMETRY_COLS = frozenset({
    "Dispatch_ID", "Kernel_Name", "GPU_ID", "Queue_ID", "PID", "TID",
    "Correlation_ID", "Start_Timestamp", "End_Timestamp",
    "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup",
    "Scratch_Per_Workitem", "Arch_VGPR", "Accum_VGPR", "SGPR", "Wave_Size",
})


def dump_all_counters(rpc, outpath):
    """Dump every counter sum from pmc_perf.csv to a JSON file.

    Hardware counters are summed across dispatches under `pmc_perf::<col>`.
    Per-dispatch geometry/identity columns (Grid_Size, Arch_VGPR, ...) are
    reported separately under `geometry::<col>` as the first row's value —
    summing them would be meaningless.

    Returns the number of entries written.
    """
    rows = {}
    pmc = rpc.get("pmc")
    if pmc is not None and not pmc.empty:
        for col in pmc.columns:
            if col in ("Dispatch_ID", "Kernel_Name"):
                continue
            try:
                if col in _GEOMETRY_COLS:
                    val = pmc[col].iloc[0]
                    rows[f"geometry::{col}"] = (
                        float(val) if pd.api.types.is_numeric_dtype(pmc[col]) else str(val)
                    )
                elif pd.api.types.is_numeric_dtype(pmc[col]):
                    rows[f"pmc_perf::{col}"] = float(pmc[col].sum())
                else:
                    rows[f"pmc_perf::{col}"] = str(pmc[col].iloc[0])
            except Exception as e:
                rows[f"pmc_perf::{col}"] = f"<error: {e}>"
    Path(outpath).write_text(json.dumps(rows, indent=1, default=str))
    return len(rows)


def dump_key_counters(rpc, arch, outpath_json, outpath_txt=None):
    """Dump the curated key-counter set to JSON (+ optional plain-text)."""
    # Resolve arch ONCE so the counter set and the recorded __arch__ value
    # cannot drift. The on-disk artifacts then accurately name the counter
    # list that was actually used.
    resolved_arch = arch or detect_arch(rpc) or "gfx942"
    keys = key_counters_for_arch(resolved_arch)
    out = {}
    pmc = rpc.get("pmc")
    for k in keys:
        v = None
        if pmc is not None and not pmc.empty and k in pmc.columns:
            try:
                if k in _GEOMETRY_COLS:
                    # Per-dispatch geometry — report first-row value, not sum.
                    v = (float(pmc[k].iloc[0]) if pd.api.types.is_numeric_dtype(pmc[k])
                         else str(pmc[k].iloc[0]))
                elif pd.api.types.is_numeric_dtype(pmc[k]):
                    v = float(pmc[k].sum())
                else:
                    v = str(pmc[k].iloc[0])
            except Exception as e:
                v = f"<error: {e}>"
        out[k] = v
    out["__duration_ns__"] = kernel_duration_ns(rpc)
    out["__arch__"] = resolved_arch
    Path(outpath_json).write_text(json.dumps(out, indent=2, default=str))
    if outpath_txt:
        with open(outpath_txt, "w") as f:
            f.write(f"arch: {out['__arch__']}\nduration_ns: {out['__duration_ns__']}\n\n")
            for k, v in out.items():
                if k.startswith("__"):
                    continue
                f.write(f"{k:60s} = {v}\n")
    return out


# --- PC sampling CSV helpers ------------------------------------------------

def load_pcsamp_csv(csv_path):
    """Load a PC-sampling CSV emitted by `rocprofv3 --pc-sampling-method ...`.

    The two PC-sampling modes produce different schemas:

      stochastic (`<pid>_pc_sampling_stochastic.csv`):
        Sample_Timestamp, Exec_Mask, Dispatch_Id, Instruction (PC),
        Instruction_Comment (ISA mnemonic), Correlation_Id,
        Wave_Issued_Instruction (0 = stalled, 1 = issued), Instruction_Type,
        Stall_Reason (populated only when Wave_Issued_Instruction == 0;
        one of NONE, NO_INSTRUCTION_AVAILABLE, ALU_DEPENDENCY, WAITCNT,
        INTERNAL_INSTRUCTION, BARRIER_WAIT, ARBITER_NOT_WIN,
        ARBITER_WIN_EX_STALL, OTHER_WAIT, SLEEP_WAIT),
        Wave_Count.

        Note: the per-execution-pipe `arb_state_stall_*` /
        `arb_state_issue_*` bit-fields are JSON-only (use `-f json` and
        read the `snapshot` object); they are NOT CSV columns.

      host_trap (`<pid>_pc_sampling_host_trap.csv`):
        Sample_Timestamp, Exec_Mask, Dispatch_Id, Instruction,
        Instruction_Comment, Correlation_Id
        (NO Stall_Reason; per-line hotspots only.)

    See https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html
    """
    return pd.read_csv(csv_path)


def _stall_col(df):
    """Return the name of the stall-reason column in df, or None.

    Stochastic-mode CSVs have `Stall_Reason`; host_trap CSVs don't.
    """
    return "Stall_Reason" if "Stall_Reason" in df.columns else None


def _group_col(df):
    """Return the per-PC grouping column (`Source` for file:line, else PC)."""
    if "Source" in df.columns:
        return "Source"
    return "Instruction" if "Instruction" in df.columns else None


def _sample_count_col(df):
    """Return the sample-weight column name if present (else None = use row counts)."""
    return "Sample_Count" if "Sample_Count" in df.columns else None


def stall_hotspots(pcs_df, top=30, stall_reason=None, *, wait_reason=None):
    """Aggregate PC samples by (Source-or-PC, Stall_Reason). Returns DataFrame.

    `wait_reason` is a deprecated alias for `stall_reason` (the real CSV column
    is `Stall_Reason`, not `Wait_Reason`). On a host_trap CSV (no Stall_Reason),
    the grouping degrades to per-source-line counts only.
    """
    if wait_reason is not None and stall_reason is None:
        stall_reason = wait_reason
    df = pcs_df
    stall_col = _stall_col(df)
    grp = _group_col(df)
    cnt = _sample_count_col(df)
    if grp is None:
        return df.iloc[0:0]
    if stall_reason is not None and stall_col is not None:
        df = df[df[stall_col] == stall_reason]
    if df.empty:
        return df

    group_keys = [grp, stall_col] if stall_col is not None else [grp]
    if cnt is not None:
        agg = df.groupby(group_keys)[cnt].sum().sort_values(ascending=False).head(top)
        total = float(pcs_df[cnt].sum())
    else:
        agg = df.groupby(group_keys).size().sort_values(ascending=False).head(top)
        total = float(len(pcs_df))
    return agg.to_frame("samples").assign(
        pct=lambda d: d["samples"] / total * 100 if total else 0.0
    )


def stall_hotspots_per_line(pcs_df, top=30):
    """Per-source-line breakdown across all stall reasons. Returns DataFrame
    with columns: total samples + one column per stall reason. On a host_trap
    CSV (no `Stall_Reason`) it returns a single-column per-line total."""
    if pcs_df.empty:
        return pcs_df
    stall_col = _stall_col(pcs_df)
    grp = _group_col(pcs_df)
    cnt = _sample_count_col(pcs_df)
    if grp is None:
        return pcs_df.iloc[0:0]
    if stall_col is not None:
        if cnt is not None:
            pivot = pcs_df.groupby([grp, stall_col])[cnt].sum().unstack(fill_value=0)
        else:
            pivot = pcs_df.groupby([grp, stall_col]).size().unstack(fill_value=0)
        pivot["__total__"] = pivot.sum(axis=1)
        return pivot.sort_values("__total__", ascending=False).head(top)
    # host_trap: no stall reason; return per-line totals only.
    if cnt is not None:
        agg = pcs_df.groupby(grp)[cnt].sum()
    else:
        agg = pcs_df.groupby(grp).size()
    agg = agg.sort_values(ascending=False).head(top).to_frame("__total__")
    return agg


# --- rocpd / .db helpers ----------------------------------------------------

def find_db_in(trace_dir):
    """Return the path to a .db file in trace_dir, or None.

    Prefers a top-level match, then recurses. Sorts deterministically and
    raises if multiple candidates exist at the same depth.
    """
    base = Path(trace_dir)
    top = sorted(base.glob("*.db"))
    if len(top) > 1:
        raise RuntimeError(
            f"multiple .db files in {base}: {[p.name for p in top]}; "
            "pass the explicit path."
        )
    if top:
        return top[0]
    nested = sorted(base.glob("**/*.db"))
    if len(nested) > 1:
        raise RuntimeError(
            f"multiple .db files under {base}: {[str(p.relative_to(base)) for p in nested]}; "
            "pass the explicit path."
        )
    return nested[0] if nested else None


def per_kernel_durations_from_db(db_path, name_filter=None):
    """Read kernel_dispatch from a rocpd .db, return a DataFrame.

    Falls back to plain sqlite3 if the `rocpd` Python helper isn't importable.
    """
    db_path = str(db_path)
    try:
        import rocpd  # type: ignore
        with rocpd.open(db_path) as db:  # type: ignore
            df = db.kernel_dispatches(name_filter=name_filter)
            return df
    except Exception:
        pass

    con = sqlite3.connect(db_path)
    try:
        where = ""
        params = ()
        if name_filter:
            where = " WHERE n.value LIKE ?"
            params = (f"%{name_filter}%",)
        sql = f"""
            SELECT kd.dispatch_id, n.value AS kernel_name,
                   (kd.end - kd.start) AS duration_ns,
                   kd.workgroup_size_x, kd.workgroup_size_y, kd.workgroup_size_z,
                   kd.grid_size_x, kd.grid_size_y, kd.grid_size_z
            FROM kernel_dispatch kd
            JOIN string n ON kd.kernel_name_id = n.id
            {where}
        """
        return pd.read_sql(sql, con, params=params)
    finally:
        con.close()


def list_tables(db_path):
    """List all table names in a rocpd .db file."""
    con = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", con
        )["name"].tolist()
    finally:
        con.close()


# --- rocprof-compute "analyze" text dump helper ------------------------------

def parse_analyze_text(path):
    """Best-effort parser for the plain-text output of `rocprof-compute analyze`.

    Returns dict[section_id_str] -> list-of-rows. We don't try to type-coerce;
    that's the caller's job. Use this for diffing two `details_<tag>.txt` files.
    """
    # rocprof-compute marks sections by either the legacy dotted form
    # ("2.1.15 Memory — vL1 Cache", with 2+ dots) or the current top-level
    # integer block ID ("15. L1D Cache"). Require either 2+ dots OR a 1-3
    # digit integer FOLLOWED BY a literal period — `\d{1,3}\.?` would also
    # match bare integers like "3 dispatches" / "15 kernels" and silently
    # reclassify data rows as new section headers.
    section_re = re.compile(
        r"^((?:\d+(?:\.\d+){2,})|(?:\d{1,3}\.))\s+\S"
    )
    out = {}
    current = None
    rows = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = section_re.match(stripped)
        if m:
            if current is not None:
                out[current] = rows
            current = m.group(1)
            rows = []
            continue
        rows.append(line)
    if current is not None:
        out[current] = rows
    return out
