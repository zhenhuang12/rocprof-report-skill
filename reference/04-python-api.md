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
rpc_<tag>/                ← when you pass `-p <dir>`, files land FLAT under it
├── pmc_perf.csv          ← all collected PMCs land here, one row per dispatch
├── pmc_kernel_top.csv    ← top-K kernels by dispatch count / time (when present)
├── sysinfo.csv           ← wide single-row: gfx arch, ROCm version, CU count, partition, etc.
├── log.txt               ← rocprof-compute invocation log
├── profiling_config.yaml ← captured invocation config
├── roofline.pdf          ← PDF when roofline ran (default-on; `--no-roof` to skip)
├── perfmon/              ← per-PMC-group .txt/.yaml input files (not analysis data)
└── out/pmc_<N>/<host>/<pid>_{kernel_trace,counter_collection,agent_info}.csv
                           ← raw per-pass dumps before merge
```

Notes:
- There is **no `timestamps.csv`** in current rocprof-compute. Per-kernel wall-clock
  duration is in `pmc_perf.csv` columns and in rocprofv3's `kernel_trace.csv` (run
  a separate `rocprofv3 --kernel-trace -f csv -d <path>`).
- There is **no top-level `roofline.csv`**. When roofline runs, the artifact is a PDF.
- There is no `SoC/` subdir — every counter lives in `pmc_perf.csv`.
- When `-p` is omitted, output defaults to `<cwd>/workloads/<name>/` (no `<gpu_model>/`
  injection in current 7.x).

```python
import os, pandas as pd
from pathlib import Path

RUN = os.environ["PROFILE_RUN_DIR"]
RPC_DIR = Path(f"{RUN}/reports/rpc_<tag>")

pmc = pd.read_csv(RPC_DIR / "pmc_perf.csv")

# Filter to the kernel(s) of interest — Kernel_Name lives in pmc_perf.csv directly.
mask = pmc["Kernel_Name"].astype(str).str.contains("my_kernel", regex=True)
pmc_k = pmc[mask]
print(f"Found {len(pmc_k)} dispatch(es) of my_kernel")

# For wall-clock durations, parse rocprofv3 kernel_trace.csv (has Start/End_Timestamp).
# rocprof-compute profile no longer emits its own timestamps.csv.
ktrace_glob = list(Path(f"{RUN}/reports/trace_<tag>").rglob("*_kernel_trace.csv"))
if ktrace_glob:
    kt = pd.concat([pd.read_csv(p) for p in ktrace_glob], ignore_index=True)
    kt_k = kt[kt["Kernel_Name"].astype(str).str.contains("my_kernel", regex=True)]
    print(f"Total runtime: {(kt_k['End_Timestamp'] - kt_k['Start_Timestamp']).sum() / 1e3:.2f} µs")
```

`pmc_perf.csv` columns vary by ROCm release — confirm with `pmc.columns.tolist()`. Core launch columns present on all recent releases: `Dispatch_ID, Kernel_Name, GPU_ID, Queue_ID, PID, TID, Grid_Size, Workgroup_Size, LDS_Per_Workgroup, Scratch_Per_Workitem, Arch_VGPR, Accum_VGPR, SGPR, Wave_Size` + each PMC counter as its own column. Some releases also expose `Kernel_ID` and `Correlation_ID`; treat both as optional. **There are no `VGPRs`/`SGPRs`/`AGPRs` plural columns and no `Start_Timestamp`/`End_Timestamp` columns** — use `Arch_VGPR`/`Accum_VGPR`/`SGPR` (singular) and read durations from `kernel_trace.csv`.

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
# pmc_perf.csv holds every collected PMC; no SoC/ subdir to walk.
seen = set(pd.read_csv(RPC_DIR / "pmc_perf.csv", nrows=1).columns)

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
import os, pandas as pd
RUN = os.environ["PROFILE_RUN_DIR"]
ts = pd.read_csv(f"{RUN}/reports/rpc_ts_<tag>/pmc_perf_timeseries.csv")
# Columns include: Sample_Time_ns, plus each counter as a column
import matplotlib  # or use the ASCII plotter in helpers/plot_timeline.py
```

For PC-sampling / ATT-style per-PC data (the AMD analog of NVIDIA's per-correlation-ID per-PC counts) you read **PC-sampling CSV** or **ATT JSON**:

```python
# PC sampling — rocprofv3 writes the CSV at a nested PID-prefixed path like
# pcsamp_<tag>/pmc_1/<host>/<pid>_pc_sampling_host_trap_v0.csv. Glob both the
# bare and PID-prefixed forms so this works regardless of how rocprofv3 was
# invoked (standalone vs through rocprof-compute).
import os, glob, pandas as pd
RUN = os.environ["PROFILE_RUN_DIR"]
csvs = sorted(set(
    glob.glob(f"{RUN}/reports/pcsamp_<tag>/**/pc_sampling_*.csv", recursive=True) +
    glob.glob(f"{RUN}/reports/pcsamp_<tag>/**/*_pc_sampling_*.csv", recursive=True)
))
if not csvs:
    raise FileNotFoundError(f"no pc_sampling CSV under {RUN}/reports/pcsamp_<tag>")
pcs = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
# Columns: Dispatch_ID, Sample_Time_ns, Instruction_Address, Source, Instruction_Comment, Wait_Reason, Sample_Count, ...
hot = (pcs.groupby(["Source", "Wait_Reason"])["Sample_Count"]
          .sum().sort_values(ascending=False).head(20))
print(hot)
```

For production use, prefer `helpers/extract_stall_hotspots.py --pcsamp-dir ...` — it handles both layouts and degrades cleanly on missing input.

`Source` is `file:line` (populated only when compiled with `-gline-tables-only` / `-g`). `Instruction_Comment` is the ISA mnemonic (`global_load_dwordx4`, `v_mfma_f32_16x16x16bf16_1k`, `s_waitcnt`, …; AMDGPU MFMA dtype glues onto the tile shape with no underscore between them, while suffixes like `_1k` use an underscore). `Wait_Reason` is one of the AMD wait categories — see the table in [`05-analysis-dimensions.md`](05-analysis-dimensions.md).

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

**Primary path — raw `sqlite3`** (works on any ROCm 7.x install without extra packages):

```python
import os, sqlite3, pandas as pd
RUN = os.environ["PROFILE_RUN_DIR"]

con = sqlite3.connect(f"{RUN}/reports/trace_<tag>/<run>.db")

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

If your install ships the `rocpd` python helper (NOT guaranteed — it's missing from
some ROCm 7.2 containers), it can be more ergonomic than raw sqlite3:

```python
import rocpd  # ImportError-prone; fall back to sqlite3 above if missing
with rocpd.open("path/to/file.db") as db:
    df = db.kernel_dispatches(name_filter="my_kernel")
```

---

## Discovering counter / column kinds

Most rocprof outputs are plain numeric CSV. One edge case:

1. Some counter columns are **already aggregated** (e.g., `..._sum` is sum across all CUs / channels; `..._avg` is average). Don't sum again. Convention is in the suffix. Use the unsuffixed name only when summing yourself across channels (e.g., grouping `TCC_EA0_RDREQ` by `Dispatch_ID` if rocprof-compute exposed it that way on your build).

---

## Exploring when you don't know the right counter name

```python
import re

pat = re.compile(r"^(SQ_WAIT|TCC_EA0_).*", re.I)
df = pd.read_csv(RPC_DIR / "pmc_perf.csv", nrows=1)
for col in df.columns:
    if pat.search(col):
        print(col)
```

This is how I built [`08-mi300x-mi355x-counter-names.md`](08-mi300x-mi355x-counter-names.md) — by enumerating everything available on gfx942 / gfx950.

---

## Comparing two reports programmatically

```python
import os, glob, pandas as pd
from pathlib import Path
RUN = os.environ["PROFILE_RUN_DIR"]

def compare(rpc_dir_1, rpc_dir_2, kernel_regex, counters, trace_dir_1=None, trace_dir_2=None):
    def load_rpc(d):
        pmc = pd.read_csv(Path(d) / "pmc_perf.csv")
        return pmc[pmc["Kernel_Name"].astype(str).str.contains(kernel_regex, regex=True)]
    def load_trace_duration(trace_dir):
        if not trace_dir:
            return 0.0
        csvs = glob.glob(f"{trace_dir}/**/*_kernel_trace.csv", recursive=True)
        if not csvs: return 0.0
        kt = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
        kt = kt[kt["Kernel_Name"].astype(str).str.contains(kernel_regex, regex=True)]
        return float((kt["End_Timestamp"] - kt["Start_Timestamp"]).sum())
    p1 = load_rpc(rpc_dir_1); p2 = load_rpc(rpc_dir_2)
    dur1 = load_trace_duration(trace_dir_1); dur2 = load_trace_duration(trace_dir_2)
    print(f"{'Counter':<45} {'v1':>15} {'v2':>15} {'change':>10}")
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
    f"{RUN}/reports/rpc_v1",
    f"{RUN}/reports/rpc_v2",
    kernel_regex="my_kernel",
    counters=[
        "SQ_WAVES", "SQ_INSTS_VALU", "SQ_INSTS_MFMA",
        "SQ_WAIT_INST_ANY", "SQ_WAIT_INST_LDS",
        "TCC_EA0_RDREQ_sum", "GRBM_GUI_ACTIVE",
    ],
    trace_dir_1=f"{RUN}/reports/trace_v1",
    trace_dir_2=f"{RUN}/reports/trace_v2",
)
```

---

## Extracting rocprof-compute's section / SoL output as data

rocprof-compute's section reports are designed as human-readable tables, but the underlying CSVs are right there. The "speed-of-light" gaps in the SoL block (`-b 2`, formerly §2.1.1) are derived from `pmc_perf.csv` + the roofline benchmarks; you can recompute them in Python:

```python
# sysinfo.csv is a WIDE single-row format (not param/value pairs). Inspect with
# `sysinfo.columns.tolist()`. Useful columns include: workload_name, command,
# compute_partition, memory_partition, gpu_model, gpu_arch, cu_per_gpu,
# simd_per_cu, num_xcd, num_hbm_channels, lds_banks_per_cu, ...
# Peak HBM BW is NOT in sysinfo — hard-code per-arch.
PEAK_HBM_BW_GBPS = {"gfx942": 5300.0, "gfx950": 8000.0}
sysinfo = pd.read_csv(RPC_DIR / "sysinfo.csv")
arch = str(sysinfo.iloc[0]["gpu_arch"]).strip()
peak_hbm_gbps = PEAK_HBM_BW_GBPS.get(arch, 5300.0)
num_xcd = int(sysinfo.iloc[0]["num_xcd"])

# Wall-clock duration: from rocprofv3 kernel_trace.csv (rocprof-compute has no timestamps.csv).
import glob
ktrace_csvs = glob.glob(f"{RUN}/reports/trace_<tag>/**/*_kernel_trace.csv", recursive=True)
kt = pd.concat([pd.read_csv(p) for p in ktrace_csvs], ignore_index=True)
kt_k = kt[kt["Kernel_Name"].astype(str).str.contains("my_kernel", regex=True)]
dur_s = float((kt_k["End_Timestamp"] - kt_k["Start_Timestamp"]).sum()) / 1e9

# gfx942/gfx950 only expose TCC_EA0_*, not TCC_EA1_*.
read_bytes = float(pmc[["TCC_EA0_RDREQ_32B_sum"]].sum().sum()) * 32
ach_gbps = read_bytes / dur_s / 1e9
print(f"HBM read: {ach_gbps:.1f} / {peak_hbm_gbps:.1f} GB/s = {ach_gbps/peak_hbm_gbps*100:.1f}% of peak")
```

If you'd rather just have the table, `rocprof-compute analyze -p ...` prints exactly the same info — pipe to a file and archive it as `details_<tag>.txt`.

---

## Saving everything for later

Archive the full counter dump so future analysis doesn't need to re-open rocprof outputs:

```python
import os, json
from pathlib import Path
import pandas as pd
RUN = os.environ["PROFILE_RUN_DIR"]

def dump_all(rpc_dir, kernel_regex, outpath):
    rpc = Path(rpc_dir)
    rows = {}
    pmc = rpc / "pmc_perf.csv"
    if pmc.exists():
        df = pd.read_csv(pmc)
        df = df[df["Kernel_Name"].astype(str).str.contains(kernel_regex, regex=True)]
        for col in df.columns:
            if col in ("Dispatch_ID", "Kernel_Name"): continue
            rows[f"pmc_perf::{col}"] = df[col].sum() if pd.api.types.is_numeric_dtype(df[col]) else df[col].iloc[0]
    Path(outpath).write_text(json.dumps(rows, indent=1, default=str))

dump_all(f"{RUN}/reports/rpc_<tag>", "my_kernel", f"{RUN}/analysis/metrics_all_<tag>.json")
```

This makes future re-analysis cheap: the raw data lives as JSON, you don't need to reopen the report.

---

## Gotchas

- **`KeyError` / missing column** on a counter that "should" exist: the counter has a different name on this gfx target, or the IP block isn't enabled in this build. Check [`08-mi300x-mi355x-counter-names.md`](08-mi300x-mi355x-counter-names.md) or enumerate with the snippet above.
- **`TCC_EA_*` vs `TCC_EA0_*`**: on gfx942 / gfx950 only `TCC_EA0_*` is exposed per XCD — there is **no** `TCC_EA1_*` (the two-channel form was gfx906 / gfx908). Don't sum a nonexistent second channel; verify with `rocprofv3 -L | grep TCC_EA`.
- **PC-sampling `Source` column is blank**: rebuild with `-gline-tables-only` / `-g`. Same for ATT's source attribution.
- **Per-PC ATT data is split across many JSON files** (one per CU / shader engine): glob `att_<tag>/**/*.json` and aggregate.
- **`SQ_INSTS_MFMA = 0` on a kernel you expect to use MFMA**: either MFMA is not being emitted (check ISA with `llvm-objdump -d`), or the counter group wasn't collected this pass; rerun with `rocprof-compute analyze -p <dir> -b 10` (Compute Pipe) / `-b 11` (Instruction Mix), or `rocprofv3 --pmc SQ_INSTS_MFMA`.
- **`rocprofv3` replays the application** between PMC groups — any host-side work (init, dataset load) runs N times. Move it out of the profile window.
