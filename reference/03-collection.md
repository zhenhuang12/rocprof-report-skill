# Profile Collection Commands

This document lists the exact `rocprofv3` and `rocprof-compute` commands you should run, in what order, and what each flag does.

---

## Prerequisites recap

- `-gline-tables-only` (or `-g`) in the compile flags (see `02-harness-guide.md`).
- `rocprofv3` (ROCm 6.2+) and `rocprof-compute` (ROCm 6.3+, formerly Omniperf) on PATH.
- User in `render` group (and `video` on some distros); ATT and PC sampling may additionally need `CAP_PERFMON` or `kfd_admin_group` — check `getfacl /dev/kfd`.
- Kernel name known (check with `llvm-objdump --syms --demangle <code-object>` if unsure — see `02-harness-guide.md`).
- **`$PROFILE_RUN_DIR` exported** (Phase 0 in [`01-workflow.md`](01-workflow.md) and the Quickstart in [`../SKILL.md`](../SKILL.md)). Every recipe below writes to `$PROFILE_RUN_DIR/reports/...` and silently misfires (output lands at `/reports/...`) if the variable is unset.

Quick smoke test:
```bash
rocprofv3 --kernel-trace \
    --kernel-include-regex "YOUR_KERNEL_NAME" \
    -d /tmp/rocprof_smoke \
    -- ./harness [args]
ls /tmp/rocprof_smoke   # should contain at least *_kernel_trace.csv (and *.db on ROCm 7+)
```

If `kernel_trace.csv` is empty, the regex didn't match — verify with `llvm-objdump --syms --demangle`. If you get a permission error, see `09-common-issues.md`.

---

## Recipe 1: Kernel-trace overview (first pass)

Cheapest pass — runtime APIs, HSA dispatch records, kernel begin/end timestamps. No PMC overhead. Use this to confirm the regex matches and to get a wall-time map.

```bash
rocprofv3 --kernel-trace --hip-trace --hsa-trace \
    --kernel-include-regex "KERNEL_REGEX" \
    -f csv \
    -d $PROFILE_RUN_DIR/reports/trace_<tag> \
    -- ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--kernel-trace` | Record kernel dispatch records (name, grid, block, agent, queue, duration). |
| `--hip-trace` | Record HIP runtime API calls. |
| `--hsa-trace` | Record HSA runtime API calls (lower-level, useful when you want to see queue / signal activity). |
| `--kernel-include-regex` | Only trace kernels whose demangled name matches. Reduces output volume. Itanium-ABI demangled symbol — same as `llvm-objdump --syms --demangle`. |
| `-d` | Output directory (rocprofv3 writes multiple files inside, under `%hostname%/%pid%/` by default). |
| `-f` / `--output-format` | `{csv, json, pftrace, otf2, rocpd}`. Pass `csv` to get the human-/pandas-friendly CSVs; the default depends on the build. |
| `--` | Separates rocprofv3 options from the command to launch. |

Output: `*_kernel_trace.csv`, `*_hip_api_trace.csv`, `*_hsa_api_trace.csv` when `-f csv`; with `-f rocpd` you get a single `.db` SQLite (the `rocpd` schema, default on ROCm 7+). Add `-f pftrace` for a Perfetto trace, or `-f json` for the rocprofv3 JSON.

Replay count: 1 pass. Wall time: kernel duration + ~tens of ms init.

---

## Recipe 2: Section-based perf metrics (second pass) — analog of `ncu --set full`

This is the bread-and-butter run. `rocprof-compute profile` collects the PMC groups that back rocprof-compute's ~24 perf sections (block `2` SoL, `5` CS / wavefront, `7` wavefront launch, `10` compute pipe, `11` instruction mix, `15` L1D, `16` L2 cache, `17` L2-fabric / HBM, `18` scratch / spill — see `--list-metrics gfx942` for the full list; older docs use dotted IDs like `2.1.10` for the same blocks) and the roofline model.

```bash
rocprof-compute profile \
    -n <run_name>_<tag> \
    -k "KERNEL_SUBSTRING" \
    -p $PROFILE_RUN_DIR/reports/rpc_<tag> \
    -- ./harness [args]
```

> **Roofline is ON by default** in current rocprof-compute (7.x). Pass `--no-roof` to skip the empirical roofline benchmarks; pass `--roof-only` to run them without the regular PMC pass. **There is no `--roofline` flag** — invoking it crashes. `--no-roof` cannot be combined with `--set` or `--roof-only` (per-help), but `--no-roof` *is* compatible with `-b`.

| Flag | Meaning |
|---|---|
| `-n` / `--name` | Workload name used in report titles (and, only when `-p` is **omitted**, in the default output path). |
| `--no-roof` | Skip the empirical roofline benchmarks (otherwise they run by default and add ~30 s). |
| `-k` / `--kernel` | Filter on demangled kernel names (substring, accepts multiple values). Limits the kernels measured per PMC group. (Note: the flag is `--kernel`, not `--kernel-name`.) |
| `-p` / `--path` | Output directory. When you pass `-p`, rocprof-compute writes **flat** under that directory: `pmc_perf.csv`, `sysinfo.csv`, `log.txt`, `profiling_config.yaml`, plus a `perfmon/<group>.{txt,yaml}` subdir and an `out/pmc_<N>/<hostname>/<pid>_{kernel_trace,counter_collection,agent_info}.csv` raw-per-pass subdir. There is no `timestamps.csv` and no top-level `roofline.csv` (when roofline runs, the artifact is a PDF). When `-p` is omitted, output defaults to `<cwd>/workloads/<name>/`. |
| `-b` / `--block` | Filter to specific metric IDs (e.g. `12.1.1`), block IDs (e.g. `12`), or block aliases (e.g. `lds`, `l1i`, `sl1d`). |

Replay count: ~15-30 passes (one per PMC group; rocprofv3 replays the whole binary, not just the kernel). Wall time: kernel time × number of groups + init.

After collection, render the section reports:

```bash
# Print all sections to stdout (use for diff & archive)
rocprof-compute analyze -p $PROFILE_RUN_DIR/reports/rpc_<tag> > \
    $PROFILE_RUN_DIR/analysis/details_<tag>.txt

# A single section (e.g., `-b 15` = L1D cache; older dotted form `2.1.15`)
rocprof-compute analyze -p $PROFILE_RUN_DIR/reports/rpc_<tag> -b 15

# List all kernels & dispatches (NOT section IDs)
rocprof-compute analyze -p $PROFILE_RUN_DIR/reports/rpc_<tag> --list-stats

# List all section / metric IDs for this arch (gfx942 for MI300X, gfx950 for MI355X)
rocprof-compute analyze --list-metrics gfx942
```

**Always read `details_<tag>.txt` first.** Each section has a "Speed-of-Light" line that names the bottleneck subsystem and a numeric gap to peak — this is the AMD analog of NCU's `Est. Speedup` rule.

For a browsable GUI, use the **ROCprof Compute Viewer** (the AMD analog of `ncu-ui`). RGP does *not* support CDNA / Instinct GPUs; do not try to open these reports with it.

---

## Recipe 2b: Timeseries collection (optional, required for CU timeline / tail-effect analysis)

Recipe 2 averages PMCs over the whole kernel. To see the *shape* of utilization over time — pipeline bubbles, tail effect, ramp-up / ramp-down — collect a separate timeseries pass. Dimension 5 (CU timeline), Pattern B (tail effect from variable-length inputs), and Pattern M (pipeline bubbles) in the diagnosis playbook all consume this CSV. Skip if you don't need timeline analysis.

```bash
rocprof-compute profile -n <run_name>_<tag>_ts \
    -k "KERNEL_SUBSTRING" \
    --timeseries-sampling-rate 1ms \
    -p $PROFILE_RUN_DIR/reports/rpc_ts_<tag> \
    -- ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--timeseries-sampling-rate` | Sample PMC counters at this interval. `1ms` is a sensible default for kernels in the 10 ms - 1 s range; drop to `100us` (= 0.1 ms) for sub-ms kernels; raise to `10ms` for very long ones. rocprof-compute's effective floor is ~1 ms (vs Nsight Compute / PM ~2 µs), so for very-short kernels prefer ATT instead. |

Output: in addition to the usual `pmc_perf.csv`, a `pmc_perf_timeseries.csv` lands under `$PROFILE_RUN_DIR/reports/rpc_ts_<tag>/`. Use `plot_timeline.py --timeseries <path-to-csv>` to render it.

Why a separate run, not `-p` pointing at the same dir as Recipe 2: the timeseries pass adds substantial overhead and writes a different schema. Keeping the two runs separate also lets Recipe 2 stay cheap when you don't need a timeline.

---

## Recipe 3: Per-line stall sampling (third pass) — analog of `ncu --set source`

Two options. Prefer PC sampling (lower overhead) when available; fall back to ATT.

### 3a) PC sampling

```bash
rocprofv3 --pc-sampling-beta-enabled \
    --pc-sampling-method host_trap \
    --pc-sampling-interval 1000 \
    --pc-sampling-unit time \
    --kernel-include-regex "KERNEL_REGEX" \
    -f csv \
    -d $PROFILE_RUN_DIR/reports/pcsamp_<tag> \
    -- ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--pc-sampling-beta-enabled` | **Required in ROCm 6.4+** — PC sampling is still a beta feature; sets `ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1` internally. |
| `--pc-sampling-method` | `host_trap` (works on MI200+) is the most portable; `stochastic` is lower-overhead on MI300+ if your ROCm build enables it. **Note the underscore — not `host-trap`.** |
| `--pc-sampling-interval` | Sample every N units (per `--pc-sampling-unit`). For `host_trap` + `time`, units are **microseconds** (the rocprof-compute default is 1048576 µs ≈ 1 s, which is FAR too coarse for short kernels). `1000` = 1 ms is a sensible starting point; drop to `100` for sub-ms kernels. |
| `--pc-sampling-unit` | **`host_trap` only accepts `time`** — passing `cycles` or `instructions` with `host_trap` is rejected at runtime as "PC sampling configuration is not supported". `cycles` and `instructions` are for `stochastic`. |
| `--kernel-include-regex` | Limits sampling to matching kernels. |
| `-f` / `--output-format` | Format of output: `{csv, json, pftrace, otf2, rocpd}`. Use `csv` for downstream pandas parsing; default writes to `%hostname%/%pid%/` structure. |

Output: per-kernel CSV with `Instruction_Address`, `Source` (file:line, populated only when compiled with `-gline-tables-only`/`-g`), `Instruction_Comment` (the SASS-equivalent text on AMD: the ISA mnemonic), `Wait_Reason`, `Sample_Count`.

**Stochastic alternative (MI300+; lower overhead, only if your ROCm build enables it):**

```bash
rocprofv3 --pc-sampling-beta-enabled \
    --pc-sampling-method stochastic \
    --pc-sampling-unit cycles \
    --pc-sampling-interval 1048576 \
    --kernel-include-regex "KERNEL_REGEX" \
    -f csv \
    -d $PROFILE_RUN_DIR/reports/pcsamp_<tag> \
    -- ./harness [args]
```

For `stochastic` the unit MUST be `cycles` or `instructions` (NOT `time`). `1048576` (= 2^20) cycles is a sensible default; lower the value for short kernels but expect higher overhead. If your build rejects stochastic with "PC sampling configuration is not supported", fall back to `host_trap`.

Use `extract_stall_hotspots.py` to aggregate these by `(file, line)` and by wait reason.

### 3b) ATT (Advanced Thread Trace / SQTT)

Heavier — captures every wave's instruction stream on the targeted CU(s). Default capture is **1 kernel × 1 CU per SE**, so plan accordingly.

```bash
# --att-target-cu 0 picks the CU at index 0 within each enabled SE. There is
# no special meaning to either 0 or the default of 1 — both are just CU
# indices. The examples in this skill use 0 throughout for reproducibility;
# omit the flag to fall back to the rocprofv3 default (1).
rocprofv3 --att \
    --att-target-cu 0 \
    --att-buffer-size 0x10000000 \
    --att-shader-engine-mask 0xF \
    --kernel-include-regex "KERNEL_REGEX" \
    -d $PROFILE_RUN_DIR/reports/att_<tag> \
    -- ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--att` | Enable Advanced Thread Trace. |
| `--att-target-cu N` | Capture this CU index (within each enabled SE). Default is `1`; `0` is equally valid. Both are plain indices — there's no special-case meaning. To cover more CUs, run multiple invocations or script around it. |
| `--att-shader-engine-mask` | Bitmask of SEs to enable. `0xF` = first 4 SEs. |
| `--att-buffer-size` | Per-SE trace buffer in bytes. Bump if traces are getting truncated. |

Output: per-SE JSON / binary traces; open with ROCprof Compute Viewer or process programmatically via the `att_tool` JSON. Source attribution requires `-gline-tables-only`/`-g`.

---

## Recipe 4: Targeted PMC only (fast)

If you already know which counters you want (e.g., re-running after a code change to check the fix), collect just those:

```bash
# Inline list
rocprofv3 --pmc SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_MFMA,SQ_WAIT_INST_ANY,TCP_TCC_READ_REQ_sum,TCC_EA0_RDREQ_sum,GRBM_GUI_ACTIVE \
    --kernel-include-regex "KERNEL_REGEX" \
    -f csv \
    -d $PROFILE_RUN_DIR/reports/pmc_<tag> \
    -- ./harness [args]

# Or a YAML/JSON job file (preferred for reproducibility). Each `jobs` entry
# mirrors a single rocprofv3 CLI invocation; there is no `name:` field at the
# job level (use the file name or comments to label).
# Note: gfx942 / gfx950 only expose `TCC_EA0_*` (no `TCC_EA1_*`) and only the
# three SQ_WAIT_* PMCs `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_WAIT_INST_LDS`.
cat > /tmp/pmc.yaml <<'EOF'
jobs:
  - pmc:
      - SQ_WAVES
      - SQ_INSTS_VALU
      - SQ_WAIT_INST_ANY
      - SQ_WAIT_INST_LDS
      - SQ_LDS_BANK_CONFLICT
      - TCC_EA0_RDREQ_sum
      - TCC_EA0_RDREQ_32B_sum
      - GRBM_GUI_ACTIVE
    kernel_include_regex: "KERNEL_REGEX"
EOF
rocprofv3 -i /tmp/pmc.yaml -f csv -d $PROFILE_RUN_DIR/reports/pmc_<tag> -- ./harness [args]
```

A single `pmc:` list must fit in one hardware pass — rocprofv3 will **fail** the job if the counters don't fit, it does not auto-split. To collect more than one pass' worth, add multiple `- pmc: ...` entries under `jobs:` (one entry = one extra pass), or use `pmc_groups:` for explicit grouping. Counters that share a unit (SQ_*, TCP_*, TCC_*) often fit in one group; check `rocprofv3 -L` output for the conflict list.

The full list of available counters: `rocprofv3 -L` (long form `--list-supported-counters`; the older `--list-avail` is rocprof v1 and does NOT exist in rocprofv3). On newer ROCm builds, `rocprofv3-avail list --pmc` is the dedicated companion. The legacy `--list-counters` / `--list-metrics` flags are from rocprof v1/v2 and are not part of rocprofv3.

---

## Recipe 5: A/B comparison (before vs after optimization)

```bash
# Before
rocprof-compute profile -n v1 -k my_kernel \
    -p $PROFILE_RUN_DIR/reports/rpc_v1 -- ./harness_v1 [args]

# After
rocprof-compute profile -n v2 -k my_kernel \
    -p $PROFILE_RUN_DIR/reports/rpc_v2 -- ./harness_v2 [args]

# Side-by-side from the CLI
rocprof-compute analyze \
    -p $PROFILE_RUN_DIR/reports/rpc_v1 \
    -p $PROFILE_RUN_DIR/reports/rpc_v2 \
    > $PROFILE_RUN_DIR/analysis/compare_v1_vs_v2.txt
```

Or in Python (see `04-python-api.md`):

```python
import os
from pathlib import Path
import pandas as pd
RUN = os.environ["PROFILE_RUN_DIR"]
# rocprof-compute profile does NOT write timestamps.csv. Per-kernel wall-clock
# duration comes from rocprofv3's kernel_trace.csv (run a separate
# `rocprofv3 --kernel-trace -f csv -d <path>` to produce it). rocprofv3 nests
# the CSV under <host>/<pid>/, so rglob the trace dir rather than guessing.
def _load_trace(tag):
    paths = sorted(Path(f"{RUN}/reports/trace_{tag}").rglob("*_kernel_trace.csv"))
    if not paths:  # standalone `rocprofv3 --kernel-trace` form has no PID prefix
        paths = sorted(Path(f"{RUN}/reports/trace_{tag}").rglob("kernel_trace.csv"))
    if not paths:
        raise FileNotFoundError(f"no kernel_trace.csv under {RUN}/reports/trace_{tag}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

d1, d2 = _load_trace("v1"), _load_trace("v2")
t1 = (d1["End_Timestamp"] - d1["Start_Timestamp"]).sum()
t2 = (d2["End_Timestamp"] - d2["Start_Timestamp"]).sum()
print(f"Speedup: {t1/t2:.2f}x")
```

---

## What rocprof-compute sections cover

```bash
# List all kernels & dispatches captured in this run (NOT section IDs).
rocprof-compute analyze -p <dir> --list-stats

# List all section / metric IDs available for this arch (source of truth).
rocprof-compute analyze --list-metrics gfx942        # MI300X
rocprof-compute analyze --list-metrics gfx950        # MI355X
```

Top-level block / section IDs (verified from `--list-metrics gfx942` on rocprof-compute 7.x). `-b` accepts EITHER a block ID like `12` OR a metric ID like `12.1.1` OR a block alias like `lds`/`l1i`/`sl1d`.

| ID | Block | What it tells you |
|---|---|---|
| 2  | Speed-of-Light (SoL)          | % of peak for compute, HBM, vL1, L2, scratch — the headline |
| 5  | CS / Wavefront                | Waves/wkg, achieved occupancy, wavefront launch |
| 7  | Wavefront Launch Stats        | Per-dispatch grid, workgroup, registers, LDS, occupancy limiter |
| 10 | Compute Pipe                  | VALU / SALU / Matrix-core busy %, IPC, FMA |
| 11 | Instruction Mix               | VALU / SALU / MFMA / VMEM / LDS / FLAT instruction counts |
| 12 | Pipe SoL                      | Per-pipe Speed-of-Light |
| 13 | Pipe Stats                    | Per-pipe stall / activity stats |
| 14 | Cache (overview)              | Cache summary across vL1 / L2 |
| 15 | L1D Cache (TCP)               | vL1 hit rate, sectors per request, bytes per wave |
| 16 | L2 Cache (TCC)                | L2 hit rate, atomics, bytes |
| 17 | L2 – Fabric / HBM (TCC_EA)    | HBM read/write bytes, achieved BW |
| 18 | Scratch / Spill               | Scratch reads/writes (= register spill on AMD) |
| 21 | Misc / PC sampling derived    | Other derived metrics (incl. PC-sample-driven aggregates) |

(Run `--list-metrics gfx942` on your actual install to confirm; the IDs above are top-level *block* IDs. Sub-metric IDs use decimal nesting like `12.1.1`.)

---

## Profiling multiple kernel launches

If the kernel is called multiple times and you want different iterations:

```bash
# Limit to the first N matches via the include-regex (rocprofv3 doesn't have a direct
# "skip" flag like ncu -s); instead, use --kernel-iteration-range:
rocprofv3 --kernel-trace \
    --kernel-include-regex "my_kernel" \
    --kernel-iteration-range "[6:9]" \
    -d $PROFILE_RUN_DIR/reports/trace_iter6_8 -- ./harness
```

`--kernel-iteration-range "[N:M]"` selects the Nth through Mth match (zero-indexed, half-open on some builds — check `rocprofv3 --help`). Useful for ignoring warmup or focusing on a steady-state iteration.

---

## GPU clock / power state (for reproducibility)

```bash
# Show current clocks and SCLK/MCLK states
rocm-smi --showclocks
rocm-smi --showperflevel

# Lock to a deterministic perf level (sudo required)
sudo rocm-smi --setperflevel high      # or "manual" + setsclk/setmclk

# Restore
sudo rocm-smi --resetclocks
```

For MI300X / MI355X the GPU normally reaches steady-state during rocprof-compute replays because the binary is rerun once per PMC group. If results jitter between runs, lock to `high` perf level. Note that on MI300A APU SKUs (gfx942 APU variant) thermal headroom is shared with CPU cores; rocm-smi may report throttling even at "high".

---

## Gotchas

- **`rocprof-compute profile` wall time blows up**: that's normal — each PMC group replays the whole binary. Profile a smaller representative workload if the kernel is expensive.
- **`--kernel-include-regex` matches nothing**: check with `llvm-objdump --syms --demangle <code-object>` and make sure you're matching the demangled name. Templates produce names like `my_kernel<8, 256>(...)`.
- **Output dir is empty / 0 KB**: profile terminated before the kernel launched. Usually means the regex didn't match, or the harness crashed.
- **ATT JSON is empty or "Source" column blank**: rebuild with `-gline-tables-only` (or `-g`), and confirm symbols weren't stripped at link time.
- **PC sampling silently no-ops**: not all GPU/driver/ROCm combinations expose PC sampling. Make sure you passed `--pc-sampling-beta-enabled`, and try `--pc-sampling-method host_trap` first (most portable) before `stochastic`.
- **`rocprofv3` replays the *application*, not just the kernel**: any expensive host-side init (data loading, NCCL init, hipBLASLt warmup) is paid on every replay. Move it outside the profile window — or just shrink the harness, which is the whole point.
- **MI355X support requires ROCm 7+**. ROCm 6.x will refuse `--offload-arch=gfx950` and may also refuse to enumerate gfx950 counters in `rocprofv3 -L` output.
