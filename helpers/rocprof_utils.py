"""Shared helpers for parsing rocprofv3 / rocprof-compute outputs.

Designed to work with both:
  - rocprof-compute (ROCm 6.3+): writes a directory under the `-p` path with
    a merged `pmc_perf.csv`, `pmc_kernel_top.csv`, `sysinfo.csv` (wide
    single-row format, NOT param/value), `roofline.pdf` (PDF when roofline
    ran; default-on, suppress with `--no-roof`), and `profiling_config.yaml`,
    plus raw per-PMC-group CSVs under `out/pmc_<N>/<hostname>/<pid>_*.csv`.
    There is NO `timestamps.csv` — kernel timing lives in the rocprofv3
    `kernel_trace.csv` of a separate `--kernel-trace` collection.
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


# --- Loading rocprof-compute output dirs ------------------------------------

def load_rpc_dir(rpc_dir, kernel_regex=None, kernel_trace_csv=None):
    """Load the standard rocprof-compute output tree into a dict of DataFrames.

    Args:
        rpc_dir: path to a rocprof-compute profile directory (the `-p` arg).
        kernel_regex: if given, filter pmc CSVs to dispatches whose
            Kernel_Name matches this regex. Filtering uses pmc_perf.csv's
            own `Kernel_Name` column (there is no `timestamps.csv`).
        kernel_trace_csv: optional path to a rocprofv3 `kernel_trace.csv`
            produced by a separate `--kernel-trace` run; loaded into
            out['kernel_trace'] for timing reference (Start_Timestamp /
            End_Timestamp live there, NOT in pmc_perf.csv).

    Returns:
        dict with keys:
            'pmc'           — DataFrame from pmc_perf.csv (all counters)
            'sysinfo'       — DataFrame from sysinfo.csv (wide single-row format)
            'kernel_trace'  — DataFrame from the supplied kernel_trace.csv, if any
            'rpc_dir'       — original path
    """
    rpc = Path(rpc_dir)
    if not rpc.is_dir():
        raise FileNotFoundError(f"rocprof-compute dir not found: {rpc}")

    out = {"rpc_dir": rpc}

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
    rpc = Path(rpc_dir)
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

    Reads from a rocprofv3 `kernel_trace.csv` loaded into rpc['kernel_trace']
    (pmc_perf.csv on current rocprof-compute does NOT carry per-dispatch
    Start_Timestamp / End_Timestamp columns).
    """
    if isinstance(rpc, dict):
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
# `enumerate_counters(rpc_dir)` or `rocprofv3 -L` (a.k.a. `--list-avail`).

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
    "Arch_VGPR", "Accum_VGPR", "SGPR", "Wave_Size",
]

_COMMON_SQ = [
    # Wave / instruction mix
    "SQ_WAVES", "SQ_INSTS", "SQ_INSTS_VALU", "SQ_INSTS_SALU",
    "SQ_INSTS_VMEM", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR",
    "SQ_INSTS_SMEM", "SQ_INSTS_FLAT", "SQ_INSTS_FLAT_LDS_ONLY",
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
    # Atomic / IO
    "TCC_EA0_ATOMIC_sum",
    "TCC_EA0_RDREQ_IO_sum", "TCC_EA0_WRREQ_IO_sum",
]

_COMMON_GRBM = [
    "GRBM_GUI_ACTIVE", "GRBM_COUNT", "GRBM_CP_BUSY", "GRBM_SDMA_BUSY",
]

# Per-dtype MFMA op counts on gfx942 / gfx950 use the prefix
#   SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>
# The per-tile-shape names (e.g. SQ_INSTS_MFMA_F32_16X16X16BF16) are NOT a
# stable PMC set across ROCm releases — they're sometimes derived metrics, and
# the exact spelling shifts between gfx942 and gfx950. Always confirm with
# `rocprofv3 -L | grep -i mfma` before extending these lists.
#
# CDNA3 (gfx942 / MI300X / MI300A) — FNUZ FP8 plus the integer / float dtypes.
MI300X_MFMA = [
    "SQ_INSTS_MFMA",                            # aggregate MFMA op count
    "SQ_INSTS_VALU_MFMA_MOPS_F16",
    "SQ_INSTS_VALU_MFMA_MOPS_BF16",
    "SQ_INSTS_VALU_MFMA_MOPS_F32",
    "SQ_INSTS_VALU_MFMA_MOPS_F64",
    "SQ_INSTS_VALU_MFMA_MOPS_I8",
    "SQ_INSTS_VALU_MFMA_MOPS_F8",               # FNUZ on CDNA3
]

# CDNA4 (gfx950 / MI355X) — adds OCP standard FP8 + the block-scaled F6F4
# family and extended-FP32 (XF32). VERIFIED via `rocprofv3 -L | grep MFMA`
# on gfx950: F16, BF16, F32, F64, I8, F8, XF32, F6F4. No separate _F4 / _F6
# / _MXFP4 / _MXFP6 / _MXFP8 PMCs exist — they roll up into _F6F4.
MI355X_MFMA = MI300X_MFMA + [
    "SQ_INSTS_VALU_MFMA_MOPS_XF32",
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

def dump_all_counters(rpc, outpath):
    """Dump every counter sum from pmc_perf.csv to a JSON file.

    Returns the number of entries written.
    """
    rows = {}
    pmc = rpc.get("pmc")
    if pmc is not None and not pmc.empty:
        for col in pmc.columns:
            if col in ("Dispatch_ID", "Kernel_Name"):
                continue
            try:
                if pd.api.types.is_numeric_dtype(pmc[col]):
                    rows[f"pmc_perf::{col}"] = float(pmc[col].sum())
                else:
                    rows[f"pmc_perf::{col}"] = str(pmc[col].iloc[0])
            except Exception as e:
                rows[f"pmc_perf::{col}"] = f"<error: {e}>"
    Path(outpath).write_text(json.dumps(rows, indent=1, default=str))
    return len(rows)


def dump_key_counters(rpc, arch, outpath_json, outpath_txt=None):
    """Dump the curated key-counter set to JSON (+ optional plain-text)."""
    keys = key_counters_for_arch(arch or detect_arch(rpc) or "gfx942")
    out = {}
    pmc = rpc.get("pmc")
    for k in keys:
        v = None
        if pmc is not None and not pmc.empty and k in pmc.columns:
            try:
                v = float(pmc[k].sum()) if pd.api.types.is_numeric_dtype(pmc[k]) else str(pmc[k].iloc[0])
            except Exception as e:
                v = f"<error: {e}>"
        out[k] = v
    out["__duration_ns__"] = kernel_duration_ns(rpc)
    out["__arch__"] = arch or detect_arch(rpc)
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

    Columns typically include:
        Dispatch_ID, Sample_Time_ns, Instruction_Address,
        Source (file:line; blank without -gline-tables-only),
        Instruction_Comment (ISA mnemonic), Wait_Reason, Sample_Count
    """
    return pd.read_csv(csv_path)


def stall_hotspots(pcs_df, top=30, wait_reason=None):
    """Aggregate PC samples by (Source, Wait_Reason). Returns DataFrame."""
    df = pcs_df
    if wait_reason is not None:
        df = df[df["Wait_Reason"] == wait_reason]
    if df.empty:
        return df
    agg = (
        df.groupby(["Source", "Wait_Reason"])["Sample_Count"]
        .sum()
        .sort_values(ascending=False)
        .head(top)
    )
    total = float(pcs_df["Sample_Count"].sum())
    return agg.to_frame("samples").assign(
        pct=lambda d: d["samples"] / total * 100 if total else 0.0
    )


def stall_hotspots_per_line(pcs_df, top=30):
    """Per-source-line breakdown across all wait reasons. Returns DataFrame
    with columns: total samples + one column per wait reason."""
    if pcs_df.empty:
        return pcs_df
    pivot = (
        pcs_df.groupby(["Source", "Wait_Reason"])["Sample_Count"]
        .sum()
        .unstack(fill_value=0)
    )
    pivot["__total__"] = pivot.sum(axis=1)
    return pivot.sort_values("__total__", ascending=False).head(top)


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
    # integer block ID ("15. L1D Cache" / "15 L1D Cache"). Match both, but
    # require either 2+ dots OR a 1-3 digit integer followed by a period
    # to avoid eating "3.14 ms" / "4.5e-3" data rows.
    section_re = re.compile(
        r"^((?:\d+(?:\.\d+){2,})|(?:\d{1,3}\.?))\s+\S"
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
