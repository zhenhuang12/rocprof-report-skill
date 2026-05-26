# Python API for rocprof outputs

There is **no single Python module** equivalent to NVIDIA's `ncu_report`. Instead, the AMD profiling stack writes plain CSVs, JSON, and SQLite (`.rpd` / `rocpd` schema), all of which Python's standard library + `pandas` reads directly. This doc shows the recipes.

The shared helpers in [`../helpers/rocprof_utils.py`](../helpers/rocprof_utils.py) implement these patterns for you — copy/paste from there for production scripts.

> **Notation in this file:** `<tag>` is a **literal placeholder** inside all path strings below (e.g. `rpc_<tag>/pmc_perf.csv`) — substitute your actual tag name (`v1`, `baseline`, etc.) by hand before running. The `<tag>` token is not Python interpolation and will produce a `FileNotFoundError` if pasted verbatim.

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

`rocprof-compute profile -p <dir>` writes a directory; with an explicit `-p` and
the default `--subpath gpu`, key files land **flat directly under `-p`** (the
default `"gpu"` matches neither of the two `subpath` branches in
`rocprof_compute_base.py`). Opt-in nested layouts (`--subpath gpu_model` →
`<gpu_model>/` child, `--subpath node_name` → `<hostname>/` child, or omitting
`-p` entirely → auto `<name>/<gpu_model>/`) are also handled by the helpers via
a one-level glob fallback. `RPC_DIR` below means *the dir that actually holds
`pmc_perf.csv`*:

```
rpc_<tag>/                       ← the `-p` value you passed
├── pmc_perf.csv                 ← all collected PMCs land here, one row per dispatch
├── timestamps.csv               ← per-dispatch Start/End_Timestamp (rocprof-compute does emit this)
├── sysinfo.csv                  ← wide single-row: gfx arch, ROCm version, CU count, partition, etc.
├── log.txt                      ← rocprof-compute invocation log
├── profiling_config.yaml        ← captured invocation config
├── roofline.csv                 ← roofline benchmark results (when roofline ran; default-on, --no-roof to skip)
├── empirRoof_gpu-0_*.pdf        ← roofline PDF plots (only with `--roof-only` / `--kernel-names`)
├── perfmon/                     ← per-PMC-group .txt/.yaml input files (not analysis data)
└── out/pmc_<N>/<host>/<pid>_{kernel_trace,counter_collection,agent_info}.csv
                                  ← raw per-pass dumps before merge
```

Notes:
- Default `--subpath gpu` is flat: artifacts live directly under `<-p>/`. If you
  opted into a nested layout (`--subpath gpu_model` or omitted `-p`), they sit
  one level deeper under `<gpu_model>/`; the helpers in `$SKILL/helpers/` glob
  for either form, so you don't have to know which one you used.
- There is no `SoC/` subdir — every counter lives in `pmc_perf.csv`.

```python
import os, pandas as pd
from pathlib import Path

RUN = os.environ["PROFILE_RUN_DIR"]
RPC_TOP = Path(f"{RUN}/reports/rpc_<tag>")
# Default flat layout: `pmc_perf.csv` is directly under `-p`. If you opted into
# `--subpath gpu_model` (or omitted `-p`), fall back to the one-level glob.
RPC_DIR = RPC_TOP if (RPC_TOP / "pmc_perf.csv").exists() else \
          (sorted(RPC_TOP.glob("*/pmc_perf.csv"))[0].parent
           if list(RPC_TOP.glob("*/pmc_perf.csv")) else RPC_TOP)

pmc = pd.read_csv(RPC_DIR / "pmc_perf.csv")

# Filter to the kernel(s) of interest — Kernel_Name lives in pmc_perf.csv directly.
mask = pmc["Kernel_Name"].astype(str).str.contains("my_kernel", regex=True)
pmc_k = pmc[mask]
print(f"Found {len(pmc_k)} dispatch(es) of my_kernel")

# Wall-clock durations: either `timestamps.csv` (rocprof-compute) or rocprofv3
# `kernel_trace.csv` (separate `--kernel-trace -f csv` run).
ts_path = RPC_DIR / "timestamps.csv"
if ts_path.exists():
    ts = pd.read_csv(ts_path)
    ts_k = ts[ts["Kernel_Name"].astype(str).str.contains("my_kernel", regex=True)]
    print(f"Total runtime: {(ts_k['End_Timestamp'] - ts_k['Start_Timestamp']).sum() / 1e3:.2f} µs")

ktrace_glob = list(Path(f"{RUN}/reports/trace_<tag>").rglob("*_kernel_trace.csv"))
if ktrace_glob:
    kt = pd.concat([pd.read_csv(p) for p in ktrace_glob], ignore_index=True)
    kt_k = kt[kt["Kernel_Name"].astype(str).str.contains("my_kernel", regex=True)]
    print(f"Total runtime: {(kt_k['End_Timestamp'] - kt_k['Start_Timestamp']).sum() / 1e3:.2f} µs")
```

`pmc_perf.csv` columns vary by ROCm release — confirm with `pmc.columns.tolist()`. Core launch columns present on all recent releases: `Dispatch_ID, Kernel_Name, GPU_ID, Queue_ID, PID, TID, Grid_Size, Workgroup_Size, LDS_Per_Workgroup, Scratch_Per_Workitem, Arch_VGPR, Accum_VGPR, SGPR` + each PMC counter as its own column. (Wave size is fixed at 64 on CDNA gfx9 and reported in `sysinfo.csv`, not as a per-dispatch `pmc_perf.csv` column on gfx942/gfx950.) Some releases also expose `Kernel_ID` and `Correlation_ID`; treat both as optional. **There are no `VGPRs`/`SGPRs`/`AGPRs` plural columns** — use `Arch_VGPR`/`Accum_VGPR`/`SGPR` (singular). Per-dispatch wall-clock timestamps live in the sibling `timestamps.csv` (rocprof-compute) and/or rocprofv3's `kernel_trace.csv`, not in `pmc_perf.csv` itself.

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

This is the AMD analog of `action.metric_names()` — it shows which counters were *collected* in this report. For "which counters *could* be collected on this GPU", run `rocprofv3 -L` (long form `--list-avail`; verified against `rocprofv3 --help` on ROCm 7.x. The legacy `--list-counters` / `--list-metrics` / `--list-basic` / `--list-derived` flags are all gone).

---

## Per-window PMC timeseries

There is **no** `rocprof-compute profile --timeseries-sampling-rate` flag — that name was a fabrication from earlier drafts of this skill. Verify with `rocprof-compute profile --help`. The supported windowed-PMC primitive is `rocprofv3 -P`, which produces N separate per-window CSVs (see Recipe 2b in `03-collection.md`):

```bash
rocprofv3 --pmc SQ_BUSY_CYCLES SQ_INSTS_VALU TCC_EA0_RDREQ_sum GRBM_GUI_ACTIVE \
    -P 0:1:50 --collection-period-unit msec \
    --kernel-include-regex "my_kernel" -f csv \
    -d $PROFILE_RUN_DIR/reports/rpc_ts_<tag> -- ./harness
```

Then stitch the per-window CSVs in pandas (window index = synthetic time axis):

```python
import os, glob, pandas as pd
RUN = os.environ["PROFILE_RUN_DIR"]
# Default rocprofv3 layout nests as rpc_ts_<tag>/<hostname>/<pid>_counter_collection.csv;
# --output-file <prefix> collapses to a flat rpc_ts_<tag>/<prefix>_counter_collection.csv.
# Glob both.
csvs = sorted(set(
    glob.glob(f"{RUN}/reports/rpc_ts_<tag>/*_counter_collection.csv") +
    glob.glob(f"{RUN}/reports/rpc_ts_<tag>/**/*_counter_collection.csv", recursive=True)
))
frames = []
for i, p in enumerate(csvs):
    df = pd.read_csv(p)
    df["window_idx"] = i
    frames.append(df)
ts = pd.concat(frames, ignore_index=True)
# Each row is one window × kernel-dispatch. Pivot or aggregate as needed.
```

For PC-sampling / ATT-style per-PC data (the AMD analog of NVIDIA's per-correlation-ID per-PC counts) you read **PC-sampling CSV** or **ATT JSON**:

```python
# PC sampling — rocprofv3's default `-d <dir>` writes
#   pcsamp_<tag>/<hostname>/<pid>_pc_sampling_stochastic.csv   (preferred: has Stall_Reason)
#   pcsamp_<tag>/<hostname>/<pid>_pc_sampling_host_trap.csv    (PCs only, NO Stall_Reason column)
# Pass `--output-file <prefix>` to collapse to a flat
# `pcsamp_<tag>/<prefix>_pc_sampling_*.csv` (no <hostname>/<pid>_ default).
# Glob accepts both the default <hostname>/ nesting, an explicit flat layout,
# and the defensive rocprof-compute-wrapped form (pcsamp_<tag>/out/pmc_<N>/<hostname>/...).
# (Note: current rocprof-compute has no PC-sampling subcommand, so this third
# layout is defensive — covered in case a future wrapper emits it.)
import os, glob, pandas as pd
RUN = os.environ["PROFILE_RUN_DIR"]
csvs = sorted(set(
    glob.glob(f"{RUN}/reports/pcsamp_<tag>/*_pc_sampling_*.csv") +
    glob.glob(f"{RUN}/reports/pcsamp_<tag>/**/*_pc_sampling_*.csv", recursive=True) +
    glob.glob(f"{RUN}/reports/pcsamp_<tag>/**/pc_sampling_*.csv", recursive=True)
))
if not csvs:
    raise FileNotFoundError(f"no pc_sampling CSV under {RUN}/reports/pcsamp_<tag>")
pcs = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
# Stochastic columns: Sample_Timestamp, Exec_Mask, Dispatch_Id, Instruction (PC),
#   Instruction_Comment, Correlation_Id, Wave_Issued_Instruction, Instruction_Type,
#   Stall_Reason, Wave_Count.
# Stall_Reason values: NONE, NO_INSTRUCTION_AVAILABLE, ALU_DEPENDENCY, WAITCNT,
#   INTERNAL_INSTRUCTION, BARRIER_WAIT, ARBITER_NOT_WIN, ARBITER_WIN_EX_STALL,
#   OTHER_WAIT, SLEEP_WAIT.
# Note: per-pipe `arb_state_stall_*` / `arb_state_issue_*` fields are JSON-only
# (use `-f json` and read the `snapshot` object); they are NOT CSV columns.
# Host_trap columns are a strict subset (no Stall_Reason).
if "Stall_Reason" in pcs.columns:
    # Only stalled rows carry a Stall_Reason; productive issues have Wave_Issued_Instruction == 1.
    stalled = pcs[pcs["Wave_Issued_Instruction"] == 0]
    hot = (stalled.groupby(["Instruction", "Stall_Reason"]).size()
                 .sort_values(ascending=False).head(20))
else:
    # host_trap: hotspots only, no breakdown
    hot = pcs.groupby("Instruction").size().sort_values(ascending=False).head(20)
print(hot)
```

For production use, prefer `helpers/extract_stall_hotspots.py --pcsamp-dir ...` — it handles both layouts and degrades cleanly on missing input or missing `Stall_Reason` (host_trap).

`Instruction_Comment` is the ISA mnemonic (`global_load_dwordx4`, `v_mfma_f32_16x16x16bf16_1k`, `s_waitcnt`, …; AMDGPU MFMA mnemonics: legacy FP16/BF16/I8/F64 forms concatenate the dtype with no underscore (e.g. `v_mfma_f32_16x16x16bf16_1k`, `v_mfma_i32_32x32x16i8`); FP8/BF8/F8F6F4/XF32 forms use an underscore separator (e.g. `v_mfma_f32_16x16x32_fp8_fp8`, `v_mfma_f32_16x16x128_f8f6f4`, `v_mfma_f32_*_xf32`). Always confirm with `rocprofv3 -L | grep MFMA`). Source attribution (`file:line`) is populated in the `Source` column by rocprofv3 when the binary was built with `-gline-tables-only` / `-g`. When that column is blank, resolve raw `Instruction` PCs yourself with `llvm-addr2line -e <binary> 0x<pc>`; `extract_stall_hotspots.py` aggregates by `Source` when populated and falls back to the raw PC otherwise — it does not shell out to addr2line. `Stall_Reason` (stochastic mode only) is one of the AMD stall categories — see the table in [`05-analysis-dimensions.md`](05-analysis-dimensions.md). See also AMD's docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html

---

## Per-PC → per-source-line aggregation

```python
def per_source_line(pcs_df, stall_reason=None, *, wait_reason=None):
    # `wait_reason` is a deprecated alias for `stall_reason` (kept for back-compat).
    if wait_reason is not None and stall_reason is None:
        stall_reason = wait_reason
    df = pcs_df
    col = "Stall_Reason" if "Stall_Reason" in df.columns else None
    if stall_reason and col:
        df = df[df[col] == stall_reason]
    group_key = "Source" if "Source" in df.columns else "Instruction"
    count_col = "Sample_Count" if "Sample_Count" in df.columns else None
    if count_col:
        agg = df.groupby(group_key)[count_col].sum().sort_values(ascending=False)
    else:
        agg = df.groupby(group_key).size().sort_values(ascending=False)
    total = agg.sum()
    return agg.head(20).to_frame("samples").assign(pct=lambda d: d["samples"]/total*100)

print(per_source_line(pcs, stall_reason="WAITCNT"))
```

`extract_stall_hotspots.py` ships a complete implementation that handles both PC-sampling CSV (stochastic + host_trap) and ATT JSON.

---

## Reading the `rocpd` SQLite (`.db` / `.rpd`)

ROCm 7+ rocprofv3 defaults to a single `.db` per run, using the public `rocpd` schema. Tables include `agents`, `kernel_dispatch`, `hsa_api`, `hip_api`, `memory_copy`, `pmc_sample`, `pc_sample_host_trap`, ...

**Primary path — raw `sqlite3`** (works on any ROCm 7.x install without extra packages):

```python
import os, sqlite3, pandas as pd
RUN = os.environ["PROFILE_RUN_DIR"]

from pathlib import Path
# rocprofv3 default `-d <dir>` nests the .db at `<dir>/<hostname>/<pid>_results.db`;
# `--output-file <prefix>` collapses it flat to `<dir>/<prefix>_results.db`. Use rglob
# to cover both (PID and hostname are unstable per run).
db_path = next(Path(f"{RUN}/reports/trace_<tag>").rglob("*_results.db"))
con = sqlite3.connect(db_path)

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

1. Some counter columns are **already aggregated** (e.g., `..._sum` is sum across all CUs / channels; `..._avr` is per-instance average; `..._max` is per-instance max). Don't sum again. Convention is in the suffix — see `reference/08-mi300x-mi355x-counter-names.md` for the canonical `_sum` / `_avr` / `_max` definitions. Use the unsuffixed name only when summing yourself across channels (e.g., grouping `TCC_EA0_RDREQ` by `Dispatch_ID` if rocprof-compute exposed it that way on your build).

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

# Wall-clock duration: either rocprof-compute's sibling timestamps.csv (preferred)
# or rocprofv3's kernel_trace.csv. We use kernel_trace.csv here for the regex-filter example.
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
