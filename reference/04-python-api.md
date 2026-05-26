# Python API for rocprof outputs

There is **no single Python module** equivalent to NVIDIA's `ncu_report`. Instead, the AMD profiling stack writes plain CSVs, JSON, and SQLite (`.rpd` / `rocpd` schema), all of which Python's standard library + `pandas` reads directly. This doc shows the recipes.

The shared helpers in [`../helpers/rocprof_utils.py`](../helpers/rocprof_utils.py) implement these patterns for you — copy/paste from there for production scripts.

---

## Setup

```bash
python3 -m pip install --user pandas    # the only third-party dep most scripts need
python3 -c "import pandas, sqlite3, json; print('OK')"

# Optional: the rocpd helper (ships with ROCprofiler-SDK in ROCm 7+)
python3 -c "import rocpd; print(rocpd.__file__)"
```

The helpers degrade gracefully if `rocpd` isn't importable — they fall back to raw `sqlite3` queries against the `.db` files.

---

## Basic loading — `rocprof-compute profile` output

`rocprof-compute profile -p <dir>` writes a directory; key files:

```
rpc_<tag>/
├── pmc_perf.csv          ← one row per (PMC group × kernel dispatch), columns = counter names
├── timestamps.csv        ← per-dispatch start/end ns + kernel name + grid/block
├── sysinfo.csv           ← gfx arch, ROCm version, CU count, etc.
├── SoC/                  ← per-IP CSVs (one per IP block: SQ, TCP, TCC_EA0, TCC_EA1, ...)
│   ├── SQ.csv
│   ├── TCP.csv
│   ├── TCC_EA0.csv
│   └── ...
└── roofline.csv          ← present if --roofline was used
```

```python
import pandas as pd
from pathlib import Path

RPC_DIR = Path("$PROFILE_RUN_DIR/reports/rpc_<tag>".replace("$PROFILE_RUN_DIR", "/abs/path/profile/myrun"))

pmc = pd.read_csv(RPC_DIR / "pmc_perf.csv")
ts  = pd.read_csv(RPC_DIR / "timestamps.csv")

# Filter to the kernel(s) of interest
mask = ts["Kernel_Name"].str.contains("my_kernel", regex=True)
ts_k  = ts[mask]
pmc_k = pmc[pmc["Dispatch_ID"].isin(ts_k["Dispatch_ID"])]

print(f"Found {len(ts_k)} dispatch(es) of my_kernel")
print(f"Total runtime: {(ts_k['End_Timestamp'] - ts_k['Start_Timestamp']).sum() / 1e3:.2f} µs")
```

`pmc_perf.csv` columns vary by ROCm release — confirm with `pmc.columns.tolist()`. Common: `Dispatch_ID, Kernel_Name, GPU_ID, Queue_ID, PID, TID, Grid_Size, Workgroup_Size, LDS_Per_Workgroup, Scratch_Per_Workitem, VGPRs, SGPRs, Wave_Size, Start_Timestamp, End_Timestamp` + each PMC counter as its own column.

---

## Reading a single counter

```python
def safe_col(df, name, default=None):
    """Return df[name] if present, else default. Counter names vary by release."""
    return df[name] if name in df.columns else default

mfma = safe_col(pmc_k, "SQ_INSTS_MFMA")
valu = safe_col(pmc_k, "SQ_INSTS_VALU")
waves = safe_col(pmc_k, "SQ_WAVES")

if mfma is not None and valu is not None:
    mfma_share = float(mfma.sum()) / max(1.0, float(valu.sum()))
    print(f"MFMA / VALU instruction ratio: {mfma_share*100:.2f}%")
```

**Always wrap counter access.** Counter names change between gfx906 / gfx908 / gfx90a / gfx942 / gfx950, and rocprof-compute occasionally renames columns. See [`08-mi300x-mi355x-counter-names.md`](08-mi300x-mi355x-counter-names.md) for the canonical lists.

---

## Enumerating available counters

```python
# Across pmc_perf.csv + every SoC/*.csv
import glob
seen = set()
for path in [RPC_DIR / "pmc_perf.csv"] + list((RPC_DIR / "SoC").glob("*.csv")):
    df = pd.read_csv(path, nrows=1)     # only need the header
    seen.update(df.columns)

for name in sorted(seen):
    if "SQ_WAIT" in name or "WAIT_INST" in name:
        print(name)
```

This is the AMD analog of `action.metric_names()` — it shows which counters were *collected* in this report. For "which counters *could* be collected on this GPU", run `rocprofv3 -L` (long form `--list-avail`).

---

## Per-instance timeseries

rocprof-compute supports a timeseries mode (added in ROCm 6.3) that writes one row per PMC sample instead of one row per kernel. Enable it at collection time:

```bash
rocprof-compute profile -n <name> --timeseries-sampling-rate 1ms \
    -p $PROFILE_RUN_DIR/reports/rpc_ts_<tag> -- ./harness
```

Then:

```python
import pandas as pd

ts = pd.read_csv("$PROFILE_RUN_DIR/reports/rpc_ts_<tag>/pmc_perf_timeseries.csv")
# Columns include: Sample_Time_ns, plus each counter as a column
import matplotlib  # or use the ASCII plotter in helpers/plot_timeline.py
```

For PC-sampling / ATT-style per-PC data (the AMD analog of NVIDIA's per-correlation-ID per-PC counts) you read **PC-sampling CSV** or **ATT JSON**:

```python
# PC sampling
pcs = pd.read_csv("$PROFILE_RUN_DIR/reports/pcsamp_<tag>/pc_sampling_host_trap_v0.csv")
# Columns: Dispatch_ID, Sample_Time_ns, Instruction_Address, Source, Instruction_Comment, Wait_Reason, Sample_Count, ...
hot = (pcs.groupby(["Source", "Wait_Reason"])["Sample_Count"]
          .sum().sort_values(ascending=False).head(20))
print(hot)
```

`Source` is `file:line` (populated only when compiled with `-gline-tables-only` / `-g`). `Instruction_Comment` is the ISA mnemonic (`global_load_dwordx4`, `v_mfma_f32_16x16x16_bf16`, `s_waitcnt`, …). `Wait_Reason` is one of the AMD wait categories — see the table in [`05-analysis-dimensions.md`](05-analysis-dimensions.md).

---

## Per-PC → per-source-line aggregation

```python
def per_source_line(pcs_df, wait_reason=None):
    df = pcs_df
    if wait_reason:
        df = df[df["Wait_Reason"] == wait_reason]
    agg = (df.groupby("Source")["Sample_Count"]
             .sum().sort_values(ascending=False))
    total = agg.sum()
    return agg.head(20).to_frame("samples").assign(pct=lambda d: d["samples"]/total*100)

print(per_source_line(pcs, wait_reason="WAIT_INST_VMEM"))
```

`extract_stall_hotspots.py` ships a complete implementation that handles both PC-sampling CSV and ATT JSON.

---

## Reading the `rocpd` SQLite (`.db` / `.rpd`)

ROCm 7+ rocprofv3 defaults to a single `.db` per run, using the public `rocpd` schema. Tables include `agents`, `kernel_dispatch`, `hsa_api`, `hip_api`, `memory_copy`, `pmc_sample`, `pc_sample_host_trap`, ...

```python
import sqlite3, pandas as pd

con = sqlite3.connect("$PROFILE_RUN_DIR/reports/trace_<tag>/<run>.db")

# Kernel durations
ker = pd.read_sql("""
    SELECT kd.dispatch_id, n.value AS kernel_name,
           (kd.end - kd.start) AS duration_ns,
           kd.workgroup_size_x, kd.workgroup_size_y, kd.workgroup_size_z,
           kd.grid_size_x, kd.grid_size_y, kd.grid_size_z
    FROM kernel_dispatch kd
    JOIN string n ON kd.kernel_name_id = n.id
    WHERE n.value LIKE '%my_kernel%'
""", con)
print(ker.head())

# Schema discovery
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con))
print(pd.read_sql("PRAGMA table_info(kernel_dispatch)", con))
```

Or, if `rocpd` (the Python helper) is installed:

```python
import rocpd
with rocpd.open("path/to/file.db") as db:
    df = db.kernel_dispatches(name_filter="my_kernel")
```

---

## Discovering counter / column kinds

Most rocprof outputs are plain numeric CSV. Two edge cases:

1. Some counter columns are **already aggregated** (e.g., `..._sum` is sum across all CUs / channels; `..._avg` is average). Don't sum again. Convention is in the suffix.
2. Some columns are **derived per-IP** (e.g., per-TCC channel under `SoC/TCC_EA0.csv`) and have one row per channel × dispatch. Reduce them yourself:

```python
tcc = pd.read_csv(RPC_DIR / "SoC" / "TCC_EA0.csv")
hbm_read = tcc.groupby("Dispatch_ID")["TCC_EA0_RDREQ"].sum()
```

---

## Exploring when you don't know the right counter name

```python
import re

pat = re.compile(r"^(SQ_WAIT|TCC_EA0_).*", re.I)
for path in glob.glob(str(RPC_DIR / "SoC" / "*.csv")) + [str(RPC_DIR / "pmc_perf.csv")]:
    df = pd.read_csv(path, nrows=1)
    for col in df.columns:
        if pat.search(col):
            print(f"{Path(path).name:30s}  {col}")
```

This is how I built [`08-mi300x-mi355x-counter-names.md`](08-mi300x-mi355x-counter-names.md) — by enumerating everything available on gfx942 / gfx950.

---

## Comparing two reports programmatically

```python
def compare(rpc_dir_1, rpc_dir_2, kernel_regex, counters):
    def load(d):
        ts  = pd.read_csv(Path(d) / "timestamps.csv")
        pmc = pd.read_csv(Path(d) / "pmc_perf.csv")
        sel = ts[ts["Kernel_Name"].str.contains(kernel_regex, regex=True)]
        return pmc[pmc["Dispatch_ID"].isin(sel["Dispatch_ID"])], sel
    p1, t1 = load(rpc_dir_1)
    p2, t2 = load(rpc_dir_2)
    print(f"{'Counter':<45} {'v1':>15} {'v2':>15} {'change':>10}")
    dur1 = (t1.End_Timestamp - t1.Start_Timestamp).sum()
    dur2 = (t2.End_Timestamp - t2.Start_Timestamp).sum()
    for c in ["__duration__"] + counters:
        if c == "__duration__":
            v1, v2 = dur1, dur2
        else:
            v1 = p1[c].sum() if c in p1.columns else None
            v2 = p2[c].sum() if c in p2.columns else None
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1:
            chg = (v2 - v1) / v1 * 100
            print(f"{c:<45} {v1:>15.4g} {v2:>15.4g} {chg:>+9.1f}%")
        else:
            print(f"{c:<45} {str(v1):>15} {str(v2):>15}")

compare(
    "$PROFILE_RUN_DIR/reports/rpc_v1",
    "$PROFILE_RUN_DIR/reports/rpc_v2",
    kernel_regex="my_kernel",
    counters=[
        "SQ_WAVES", "SQ_INSTS_VALU", "SQ_INSTS_MFMA",
        "SQ_WAIT_INST_VMEM", "SQ_WAIT_INST_LDS",
        "TCC_EA0_RDREQ_sum", "GRBM_GUI_ACTIVE",
    ],
)
```

---

## Extracting rocprof-compute's section / SoL output as data

rocprof-compute's section reports are designed as human-readable tables, but the underlying CSVs are right there. The "speed-of-light" gaps in section 2.1.1 are derived from `pmc_perf.csv` + the roofline benchmarks; you can recompute them in Python:

```python
# Achieved HBM read BW vs peak (peak from sysinfo.csv if --roofline ran)
sysinfo = pd.read_csv(RPC_DIR / "sysinfo.csv")
peak_hbm_gbps = float(sysinfo[sysinfo["param"]=="peak_hbm_bw_GBps"]["value"].iloc[0])

dur_s = float(((t1.End_Timestamp - t1.Start_Timestamp).sum())) / 1e9
read_bytes = float(pmc[pmc.Dispatch_ID.isin(t1.Dispatch_ID)][["TCC_EA0_RDREQ_32B_sum","TCC_EA1_RDREQ_32B_sum"]].sum().sum()) * 32
ach_gbps = read_bytes / dur_s / 1e9
print(f"HBM read: {ach_gbps:.1f} / {peak_hbm_gbps:.1f} GB/s = {ach_gbps/peak_hbm_gbps*100:.1f}% of peak")
```

If you'd rather just have the table, `rocprof-compute analyze -p ...` prints exactly the same info — pipe to a file and archive it as `details_<tag>.txt`.

---

## Saving everything for later

Archive the full counter dump so future analysis doesn't need to re-open rocprof outputs:

```python
import json
def dump_all(rpc_dir, kernel_regex, outpath):
    rpc = Path(rpc_dir)
    ts  = pd.read_csv(rpc / "timestamps.csv")
    sel = ts[ts["Kernel_Name"].str.contains(kernel_regex, regex=True)]
    rows = {}
    for path in [rpc / "pmc_perf.csv"] + list((rpc / "SoC").glob("*.csv")):
        df = pd.read_csv(path)
        if "Dispatch_ID" in df.columns:
            df = df[df["Dispatch_ID"].isin(sel["Dispatch_ID"])]
        for col in df.columns:
            if col in ("Dispatch_ID", "Kernel_Name"): continue
            rows[f"{path.stem}::{col}"] = df[col].sum() if pd.api.types.is_numeric_dtype(df[col]) else df[col].iloc[0]
    Path(outpath).write_text(json.dumps(rows, indent=1, default=str))

dump_all(RPC_DIR, "my_kernel", "analysis/metrics_all_<tag>.json")
```

This makes future re-analysis cheap: the raw data lives as JSON, you don't need to reopen the report.

---

## Gotchas

- **`KeyError` / missing column** on a counter that "should" exist: the counter has a different name on this gfx target, or the IP block isn't enabled in this build. Check [`08-mi300x-mi355x-counter-names.md`](08-mi300x-mi355x-counter-names.md) or enumerate with the snippet above.
- **`TCC_EA_*` vs `TCC_EA0_*` / `TCC_EA1_*`**: on MI300X the L2 / HBM exposes two memory channels per XCD (`EA0`, `EA1`); use the channel-suffixed counters and sum if you want a per-XCD total.
- **PC-sampling `Source` column is blank**: rebuild with `-gline-tables-only` / `-g`. Same for ATT's source attribution.
- **Per-PC ATT data is split across many JSON files** (one per CU / shader engine): glob `att_<tag>/**/*.json` and aggregate.
- **`SQ_INSTS_MFMA = 0` on a kernel you expect to use MFMA**: either MFMA is not being emitted (check ISA with `llvm-objdump -d`), or the counter group wasn't collected this pass; rerun with `rocprof-compute profile --section 2.1.10` or `--pmc SQ_INSTS_MFMA`.
- **`rocprofv3` replays the application** between PMC groups — any host-side work (init, dataset load) runs N times. Move it out of the profile window.
